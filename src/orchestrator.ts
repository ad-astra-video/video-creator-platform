/**
 * Livepeer orchestrator client (TypeScript port of the Python backend's
 * `livepeer_client.py`). Because the orchestrator is stood up AFTER the Worker
 * (architecture.md Decision 5), all HTTP goes through a `fetchImpl` seam and a
 * base-URL constructor arg, both DEFAULTED to the real thing — so it works for
 * real and can be unit-/smoke-tested against an in-memory mock.
 *
 * Uses only standard Web APIs available in Workers (fetch, URL, TextEncoder).
 */

export interface RunnerCapabilities {
  /** Task types this runner can execute. */
  tasks: string[];
  /** Minimum level for capable task tags (e.g. "restyle" needing sam3). */
  extra?: Record<string, unknown>;
}

export interface RunnerInfo {
  id: string;
  name: string;
  url: string;
  status: "ready" | "busy" | "offline";
  capabilities: RunnerCapabilities;
  priceUsdMicrosPerSec?: number;
  /** go-livepeer PriceInfo (see github.com/livepeer/go-livepeer net/lp_rpc): pricePerUnit (wei) + pixelsPerUnit. */
  priceInfo?: {
    pricePerUnit?: number;
    pixelsPerUnit?: number;
    /** Raw discovery price_info { price, currency, unit } surfaced verbatim (our runner quotes wei). */
    raw?: { price?: number; currency?: string; unit?: string };
  };
  location?: string;
  gpu?: { name?: string; vram_mb?: number };
  /** Runner-advertised video model specs (resolution/fps/duration) from its metadata. */
  modelSpecs?: unknown;
}

export interface JobPayload {
  type: string;
  /** Worker-assigned canonical job id (the orchestrator keys runner work by it). */
  jobId?: string;
  /** Model id from the catalog (resolved by the dispatch route). */
  modelId?: string;
  request: unknown;
}

export interface SubmittedJob {
  id: string;
  runnerId: string;
  status: string;
}

export interface JobStatusResult {
  id: string;
  status: "queued" | "running" | "completed" | "failed" | "cancelled";
  phase?: string;
  percent?: number;
  frame?: number;
  error?: string;
  output?: unknown;
}

export interface OrchestratorOptions {
  baseUrl: string;
  /** Injectable fetch for tests; defaults to the global `fetch`. */
  fetchImpl?: typeof fetch;
}

const DEFAULT_HEADERS = { "content-type": "application/json" };

/** Parse heartbeat/discovery `metadata` which may be an object or a JSON string. */
function parseMeta(v: unknown): Record<string, unknown> {
  if (!v) return {};
  if (typeof v === "object") return v as Record<string, unknown>;
  try {
    const p = JSON.parse(String(v));
    return p && typeof p === "object" ? (p as Record<string, unknown>) : {};
  } catch {
    return {};
  }
}

/** Livepeer orchestrator client. */
export class OrchestratorClient {
  private baseUrl: string;
  private fetchImpl: typeof fetch;

  constructor(opts: OrchestratorOptions) {
    this.baseUrl = opts.baseUrl.replace(/\/+$/, "");
    this.fetchImpl = opts.fetchImpl ?? fetch.bind(globalThis);
  }

  private endpoint(path: string): string {
    return `${this.baseUrl}${path.startsWith("/") ? path : `/${path}`}`;
  }

  /**
   * GET the orchestrator discovery endpoint and filter to runners that are
   * `ready` and advertise the requested capability (task) set.
   */
  /**
   * GET the orchestrator discovery endpoint and normalize to RunnerInfo[].
   *
   * The stored discovery URL is tried verbatim first (the desktop uses it
   * verbatim), then with the /discovery and /api/discovery suffixes and the
   * `app=video-creator` query param, so a bare host (e.g.
   * `https://orchestrator-5090-3.dpn.gg`) resolves against the real service.
   * Handles both the go-livepeer shape `[{address, runners:[{url, gpu,
   * metadata, ...}]}]` and the flat `[{runner_id, runner_url, ...}]` mock shape.
   * Prices are read from the orchestrator payload ONLY — never synthesized.
   */
  async discoverRunners(requiredCapabilities: string[] = []): Promise<RunnerInfo[]> {
    const caps = new Set(requiredCapabilities);
    let lastErr: unknown = null;
    for (const url of this.discoveryCandidates()) {
      try {
        const res = await this.fetchImpl(url, {
          method: "GET",
          headers: { accept: "application/json" },
        });
        if (!res.ok) {
          lastErr = new Error(`discovery ${res.status}: ${await res.text()}`);
          continue;
        }
        const data: unknown = await res.json();
        const parsed = this.parseDiscovery(data);
        if (parsed.length > 0) {
          return parsed.filter(
            (r) => r && (caps.size === 0 || [...caps].every((c) => (r.capabilities?.tasks ?? []).includes(c))),
          );
        }
      } catch (e) {
        lastErr = e;
      }
    }
    void lastErr;
    return [];
  }

  private discoveryCandidates(): string[] {
    const base = this.baseUrl;
    const withApp = (u: string): string =>
      u.includes("app=") ? u : u + (u.includes("?") ? "&app=video-creator" : "?app=video-creator");
    return [base, withApp(base), withApp(`${base}/discovery`), withApp(`${base}/api/discovery`)];
  }

  parseDiscovery(data: unknown): RunnerInfo[] {
    if (!Array.isArray(data)) return [];
    const out: RunnerInfo[] = [];
    for (const entry of data as Record<string, unknown>[]) {
      if (!entry) continue;
      if (Array.isArray(entry.runners)) {
        for (const r of entry.runners as Record<string, unknown>[]) out.push(this.normalizeRunner(r));
      } else if (entry.runner_id !== undefined || entry.url !== undefined || entry.runner_url !== undefined) {
        out.push(this.normalizeRunner(entry));
      }
    }
    return out;
  }

  normalizeRunner(raw: Record<string, unknown>): RunnerInfo {
    const url = String(raw.url ?? raw.runner_url ?? "");
    let id = raw.runner_id ? String(raw.runner_id) : "";
    if (!id && url) {
      const seg = url.replace(/\/+$/, "").split("/");
      id = seg.length >= 2 ? seg[seg.length - 2] : seg[seg.length - 1] || url;
    }
    if (!id) id = "runner";
    const gpuRaw = raw.gpu && typeof raw.gpu === "object" ? (raw.gpu as Record<string, unknown>) : {};
    const gpu: { name?: string; vram_mb?: number } = {};
    if (gpuRaw.name !== undefined) gpu.name = String(gpuRaw.name);
    if (typeof gpuRaw.vram_mb === "number") gpu.vram_mb = gpuRaw.vram_mb;
    const label = raw.label !== undefined ? String(raw.label) : "";
    const name = gpu.name || label || id;
    const meta = parseMeta(raw.metadata);
    let caps: string[] = [];
    if (Array.isArray(raw.capabilities)) caps = (raw.capabilities as unknown[]).map(String);
    if (caps.length === 0) {
      if (Array.isArray(meta.capabilities)) caps = (meta.capabilities as unknown[]).map(String);
    }
    const modelSpecs = Array.isArray(meta.model_specs) ? meta.model_specs : undefined;
    // Price is taken from the upstream payload only. And only when it is
    // actually present — we never invent one.
    //
    // priceUsdMicrosPerSec: trusted as pre-normalized USD micros/sec.
    // raw discovery price_info {price,currency,unit}: our live-runner quotes a
    //   price denominated in `currency` (wei or usd) for `unit`. We surface it
    //   verbatim so the client can convert wei->USD with its own ETH/USD feed
    //   (the orchestrator republishes a USD price as wei). We must NOT fold a
    //   wei price into priceUsdMicrosPerSec as if it were USD — that produced
    //   an absurd USD figure from the raw wei integer.
    let micros: number | undefined;
    if (raw.priceUsdMicrosPerSec !== undefined) {
      micros = Number(raw.priceUsdMicrosPerSec);
    }
    let rawPrice: { price?: number; currency?: string; unit?: string } | undefined;
    if (raw.price_info && typeof raw.price_info === "object") {
      const pi = raw.price_info as Record<string, unknown>;
      if (typeof pi.price === "number") {
        rawPrice = {
          price: pi.price,
          currency: String(pi.currency ?? ""),
          unit: String(pi.unit ?? ""),
        };
        const cur = (rawPrice?.currency ?? "").toLowerCase();
        if (cur === "usd" || cur === "") {
          micros = (rawPrice?.unit ?? "").toLowerCase().includes("sec")
            ? Math.round((rawPrice?.price ?? 0) * 1_000_000)
            : Math.round(((rawPrice?.price ?? 0) / 60) * 1_000_000);
        }
      }
    }
    // go-livepeer OrchestratorInfo priceInfo / price_info { pricePerUnit (wei), pixelsPerUnit }.
    const glSource =
      raw.priceInfo && typeof raw.priceInfo === "object"
        ? (raw.priceInfo as Record<string, unknown>)
        : raw.price_info && typeof raw.price_info === "object"
          ? (raw.price_info as Record<string, unknown>)
          : undefined;
    const glPpu = glSource ? glSource.pricePerUnit : raw.pricePerUnit;
    const glPix = glSource ? glSource.pixelsPerUnit : raw.pixelsPerUnit;
    const glPrice: { pricePerUnit?: number; pixelsPerUnit?: number } | undefined =
      glPpu !== undefined || glPix !== undefined
        ? {
            ...(glPpu !== undefined ? { pricePerUnit: Number(glPpu) } : {}),
            ...(glPix !== undefined ? { pixelsPerUnit: Number(glPix) } : {}),
          }
        : undefined;
    const priceInfo =
      glPrice || rawPrice
        ? { ...(glPrice ?? {}), ...(rawPrice ? { raw: rawPrice } : {}) }
        : undefined;

    const status = raw.status === "busy" ? "busy" : raw.status === "offline" ? "offline" : "ready";
    return {
      id,
      name,
      url,
      status,
      capabilities: { tasks: caps },
      ...(micros !== undefined && Number.isFinite(micros) ? { priceUsdMicrosPerSec: micros } : {}),
    ...(priceInfo ? { priceInfo } : {}),
      ...(gpu.name !== undefined || gpu.vram_mb !== undefined ? { gpu } : {}),
      ...(modelSpecs !== undefined ? { modelSpecs } : {}),
    };
  }

  /** Pick the cheapest eligible runner by price; fall back to load-round-robin. */
  selectRunner(runners: RunnerInfo[], _caps: string[] = []): RunnerInfo | null {
    if (runners.length === 0) return null;
    const priced = runners.filter((r) => Number.isFinite(r.priceUsdMicrosPerSec));
    if (priced.length > 0) {
      return [...priced].sort((a, b) => (a.priceUsdMicrosPerSec! || 0) - (b.priceUsdMicrosPerSec! || 0))[0];
    }
    return runners[0];
  }

  /**
   * POST a job payload DIRECTLY to a runner's API endpoint
   * (`runner.url + <task endpoint>`). There is NO job API between runners — the
   * webapp/WWW targets the runner itself and the task endpoint resolves the
   * task (e.g. `/video-creator/v1/{endpoint}`). Returns the HTTP status + body.
   */
  async postToRunner(
    runner: RunnerInfo,
    endpoint: string,
    body: unknown,
    timeoutMs = 120000,
  ): Promise<{ status: number; data: any }> {
    const url = runner.url.replace(/\/+$/, "") + endpoint;
    const ctrl = new AbortController();
    const t = setTimeout(() => ctrl.abort(), timeoutMs);
    try {
      const res = await this.fetchImpl(url, {
        method: "POST",
        headers: DEFAULT_HEADERS,
        body: JSON.stringify(body),
        signal: ctrl.signal,
      });
      const text = await res.text();
      let data: any = null;
      if (text) {
        try { data = JSON.parse(text); } catch { data = null; }
      }
      return { status: res.status, data };
    } finally {
      clearTimeout(t);
    }
  }

  /** Submit a job to a specific runner; returns the orchestrator's job id. */
  async submitJob(runner: RunnerInfo, payload: JobPayload): Promise<SubmittedJob> {
    const res = await this.fetchImpl(this.endpoint(`/api/jobs`), {
      method: "POST",
      headers: DEFAULT_HEADERS,
      body: JSON.stringify({ runnerId: runner.id, ...payload }),
    });
    const text = await res.text();
    let data: any = null;
    if (text) {
      try {
        data = JSON.parse(text);
      } catch {
        data = null;
      }
    }
    if (!res.ok) throw new Error(`orchestrator submit ${res.status}: ${text}`);
    if (!data?.jobId) throw new Error(`orchestrator submit: missing jobId in response: ${text}`);
    return { id: String(data.jobId), runnerId: runner.id, status: data.status || "queued" };
  }

  /** Poll a job's status from the orchestrator/runner. */
  async getJobStatus(jobId: string): Promise<JobStatusResult> {
    const res = await this.fetchImpl(this.endpoint(`/api/jobs/${encodeURIComponent(jobId)}`), {
      method: "GET",
      headers: { accept: "application/json" },
    });
    const text = await res.text();
    let data: any = null;
    if (text) {
      try {
        data = JSON.parse(text);
      } catch {
        data = null;
      }
    }
    if (!res.ok) throw new Error(`orchestrator getJobStatus ${res.status}: ${text}`);
    return data as JobStatusResult;
  }

  /** Cancel a job (best-effort). */
  async cancelJob(jobId: string): Promise<boolean> {
    const res = await this.fetchImpl(this.endpoint(`/api/jobs/${encodeURIComponent(jobId)}/cancel`), {
      method: "POST",
      headers: DEFAULT_HEADERS,
    });
    return res.ok;
  }
}
