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
  location?: string;
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
  async discoverRunners(
    requiredCapabilities: string[] = [],
  ): Promise<RunnerInfo[]> {
    const res = await this.fetchImpl(this.endpoint("/api/discovery"), {
      method: "GET",
      headers: { accept: "application/json" },
    });
    if (!res.ok) throw new Error(`orchestrator discovery ${res.status}: ${await res.text()}`);
    const data = (await res.json()) as { runners?: RunnerInfo[] };
    const runners = Array.isArray(data.runners) ? data.runners : [];
    const caps = new Set(requiredCapabilities);
    return runners.filter(
      (r) =>
        r &&
        r.status === "ready" &&
        r.capabilities?.tasks &&
        (caps.size === 0 || caps.size <= new Set(r.capabilities.tasks).size) &&
        [...caps].every((c) => r.capabilities.tasks.includes(c)),
    );
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
