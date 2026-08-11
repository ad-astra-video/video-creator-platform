/**
 * Providers (api-contract.md): backed by REAL orchestrator runner discovery. The
 * response shape matches the desktop contract the frontend's ApiClient.getProviders()
 * expects:
 *
 *   { providers: [{ runner_id, url, status, gpu?, price_info?, selected, excluded, capabilities }],
 *     total, online, chosenId, error? }
 *
 * Discovery failures are non-fatal here (the tab should render, not 500). When the
 * orchestrator isn't reachable yet (architecture.md Decision 5) and DEMO_RUNNERS=1
 * (wrangler dev / local only), we seed a clearly-labelled demo runner set from the
 * platform model/price catalog so the Models tab is demonstrable. Production (no
 * DEMO_RUNNERS) shows real discovery (possibly empty).
 */
import { z } from "zod";
import { err, ok } from "../utils";
import { makeOrchestrator, parseBody, resolveUserFromRequest } from "./lib";
import { deleteProvider, getProvider, setProvider, getSettings } from "../jobs";
import type { Env } from "../types";
import { OrchestratorClient, type RunnerInfo } from "../orchestrator";
import { DEFAULT_ORCHESTRATOR_URL } from "./lib";

const capsSchema = z.object({ capabilities: z.array(z.string()).optional() });
const selectSchema = z.object({ runnerId: z.string().min(1) });

const TASK_LABELS: Record<string, string> = {
  t2v: "Text-to-Video",
  image: "Image",
  extend: "Extend",
  retake: "Retake",
  restyle: "Restyle",
  "ic-lora": "IC-LoRA",
  "ic-lora-generate": "IC-LoRA",
  sam3: "Segment",
  prompt: "Prompt Enhance",
  "prompt-enhance": "Prompt Enhance",
  "suggest-gap-prompt": "Script Gap Fill",
  i2v: "Image-to-Video",
  edit: "Masked Edit",
  "extract-conditioning": "Extract Conditioning",
  chat: "Chat",
};

/** Demo runners used when the orchestrator isn't stood up yet (DEMO_RUNNERS=1, local). */
function demoRunners(caps: string[]): RunnerInfo[] {
  const want = new Set(caps);
  const ok = (tasks: string[]) => want.size === 0 || [...want].every((t) => tasks.includes(t));
  const out: RunnerInfo[] = [];
  if (ok(["t2v", "extend", "retake", "prompt"])) {
    out.push({
      id: "demo-rtx-4090-1",
      name: "RTX 4090 runner (demo)",
      url: "livepeer://runner.demo/demo-rtx-4090-1",
      status: "ready",
      capabilities: { tasks: ["t2v", "extend", "retake", "prompt"] },
      location: "demo",
    });
  }
  if (ok(["t2v", "image", "restyle", "extend", "ic-lora", "sam3"])) {
    out.push({
      id: "demo-rtx-5090-1",
      name: "RTX 5090 runner (demo)",
      url: "livepeer://runner.demo/demo-rtx-5090-1",
      status: "ready",
      capabilities: { tasks: ["t2v", "image", "restyle", "extend", "ic-lora", "sam3"] },
      location: "demo",
    });
  }
  return out;
}

function buildPrice(r: RunnerInfo): { usdPerSec?: number; pricePerUnit?: number; pixelsPerUnit?: number } | null {
  // Never invent a price. Surface USD/sec when advertised, else the canonical
  // go-livepeer PriceInfo (pricePerUnit wei + pixelsPerUnit) verbatim. A value of 0
  // is a real "free" price and is shown, never blanked.
  if (typeof r.priceUsdMicrosPerSec === "number" && Number.isFinite(r.priceUsdMicrosPerSec)) {
    return { usdPerSec: r.priceUsdMicrosPerSec / 1_000_000 };
  }
  if (r.priceInfo && (r.priceInfo.pricePerUnit !== undefined || r.priceInfo.pixelsPerUnit !== undefined)) {
    return { ...(r.priceInfo.pricePerUnit !== undefined ? { pricePerUnit: r.priceInfo.pricePerUnit } : {}),
             ...(r.priceInfo.pixelsPerUnit !== undefined ? { pixelsPerUnit: r.priceInfo.pixelsPerUnit } : {}) };
  }
  return null;
}

function toProviderDto(
  r: RunnerInfo,
  chosenId: string | null,
  demo: boolean,
): Record<string, unknown> {
  return {
    runner_id: r.id,
    url: r.url,
    status: r.status,
    gpu: r.gpu ?? null,
    price_info: buildPrice(r),
    capabilities: (r.capabilities?.tasks ?? []).map((t) => ({
      id: t,
      label: TASK_LABELS[t] ?? t,
    })),
    selected: r.id === chosenId,
    excluded: false,
    demo,
  };
}

/** GET /api/providers — discover ready runners (user's saved choice surfaced first). */
/**
 * Build an orchestrator client from the user's configured discovery URL (drives
 * which orchestrator we discover), falling back to env then the local default.
 */
async function orchestratorFor(env: Env, userId: string): Promise<OrchestratorClient> {
  let base = env.ORCHESTRATOR_BASE_URL || DEFAULT_ORCHESTRATOR_URL;
  if (env.DB) {
    try {
      const st = await getSettings(env.DB, userId);
      const du = st && typeof (st as Record<string, unknown>).livepeerDiscoveryUrl === "string"
        ? String((st as Record<string, unknown>).livepeerDiscoveryUrl).trim()
        : "";
      if (du) base = du;
    } catch { /* fall back to env/default */ }
  }
  return new OrchestratorClient({ baseUrl: base });
}

export async function getProviders(request: Request, env: Env): Promise<Response> {
  const u = await resolveUserFromRequest(request, env);
  if (!u.ok) return u.response;
  const orch = await orchestratorFor(env, u.userId);
  const caps = (new URL(request.url).searchParams.get("capabilities") || "").split(",").filter(Boolean);
  let runners: RunnerInfo[] = [];
  let discoveryError: string | null = null;
  try {
    runners = await orch.discoverRunners(caps);
  } catch (e) {
    discoveryError = (e as Error).message;
  }
  let demo = false;
  if (runners.length === 0 && env.DEMO_RUNNERS === "1") {
    runners = demoRunners(caps);
    demo = true;
  }
  const chosen = env.DB ? await getProvider(env.DB, u.userId) : null;
  const chosenId = chosen?.id ? String(chosen.id) : null;
  const ordered = [...runners].sort((a, b) => (a.id === chosenId ? -1 : b.id === chosenId ? 1 : 0));
  return ok({
    providers: ordered.map((r) => toProviderDto(r, chosenId, demo)),
    total: ordered.length,
    online: ordered.filter((r) => r.status === "ready").length,
    chosenId,
    ...(discoveryError ? { error: discoveryError } : {}),
  });
}

/** POST /api/providers/discover — run discovery now and return fresh ready runners. */
export async function postDiscoverProviders(request: Request, env: Env): Promise<Response> {
  const u = await resolveUserFromRequest(request, env);
  if (!u.ok) return u.response;
  const body = await parseBody(request, capsSchema);
  const caps = body.ok ? body.data.capabilities ?? [] : [];
  const orch = await orchestratorFor(env, u.userId);
  let runners: RunnerInfo[] = [];
  let discoveryError: string | null = null;
  try {
    runners = await orch.discoverRunners(caps);
  } catch (e) {
    discoveryError = (e as Error).message;
  }
  let demo = false;
  if (runners.length === 0 && env.DEMO_RUNNERS === "1") {
    runners = demoRunners(caps);
    demo = true;
  }
  return ok({
    providers: runners.map((r) => toProviderDto(r, null, demo)),
    total: runners.length,
    online: runners.filter((r) => r.status === "ready").length,
    ...(discoveryError ? { error: discoveryError } : {}),
  });
}

/** POST /api/providers/select — persist the user's chosen provider in D1. */
export async function postSelectProvider(request: Request, env: Env): Promise<Response> {
  const u = await resolveUserFromRequest(request, env);
  if (!u.ok) return u.response;
  const body = await parseBody(request, selectSchema);
  if (!body.ok) return body.response;
  if (!env.DB) return err("Server error", 500);
  await setProvider(env.DB, u.userId, { id: body.data.runnerId, selectedAt: new Date().toISOString() });
  return ok({ ok: true, runnerId: body.data.runnerId });
}

/** POST /api/providers/exclude — clear the user's saved provider choice. */
export async function postExcludeProvider(request: Request, env: Env): Promise<Response> {
  const u = await resolveUserFromRequest(request, env);
  if (!u.ok) return u.response;
  if (!env.DB) return err("Server error", 500);
  await deleteProvider(env.DB, u.userId);
  return ok({ ok: true });
}
