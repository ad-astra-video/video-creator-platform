/**
 * Catalog routes (keep (Worker/catalog) per api-contract.md). All funnel through
 * catalogService() so the backing store can become data-driven later.
 */

import { ok } from "../utils";
import { catalogService, getVideoGenerationModelSpecs } from "../catalog";
import { resolveUserFromRequest, DEFAULT_ORCHESTRATOR_URL } from "./lib";
import { getSettings } from "../jobs";
import { OrchestratorClient } from "../orchestrator";
import type { Env } from "../types";

/**
 * Build an orchestrator client from the user's configured discovery URL (drives
 * which orchestrator we discover), falling back to env then the local default.
 * Mirrors providers.ts so model specs and runner discovery agree on the source.
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

/** Canonicalize a runner-advertised spec item into a stable comparable key. */
function specKey(item: Record<string, unknown>): string {
  const p = typeof item.pipeline === "string" ? item.pipeline : "";
  const spec = (item.spec && typeof item.spec === "object" ? item.spec : {}) as Record<string, unknown>;
  const dn = typeof spec.display_name === "string" ? spec.display_name : "";
  return `${p}::${dn}`;
}

/** Merge ready runners' advertised model specs into local_models (de-duped). */
function mergeLocalModels(specs: unknown[]): unknown[] {
  const seen = new Set<string>();
  const out: unknown[] = [];
  for (const s of specs) {
    if (!Array.isArray(s)) continue;
    for (const item of s) {
      if (!item || typeof item !== "object") continue;
      const rec = item as Record<string, unknown>;
      const spec = rec.spec && typeof rec.spec === "object" ? rec.spec : null;
      if (!spec) continue;
      const k = specKey(rec);
      if (seen.has(k)) continue;
      seen.add(k);
      out.push(item);
    }
  }
  return out;
}

export async function getModels(): Promise<Response> {
  // GET /api/models == InstalledModelsResponse (frontend `listModels`): models/loras present on
  // local disk. The web app has none (everything is remote via Worker/orchestrator), so this is
  // genuinely empty; must keep the InstalledModelResponse shape so the frontend's listModels works.
  return ok({ models: [] });
}

/** GET /api/generate/models-specs — available video model specs. */
export async function getModelSpecs(request: Request, env: Env): Promise<Response> {
  const base = getVideoGenerationModelSpecs();
  // The webapp's video path runs on Livepeer runners (not the LTX API), so
  // `local_models` must come from what the runner ADVERTISES in its discovery
  // metadata — that's the data that makes video generation available in-app.
  let localModels: unknown[] = [];
  try {
    const u = await resolveUserFromRequest(request, env);
    if (u.ok) {
      const orch = await orchestratorFor(env, u.userId);
      // No capability filter — a runner may serve t2v/i2v/etc.; take all specs.
      const runners = await orch.discoverRunners();
      const advertised = runners
        .filter((r) => r.status === "ready" && r.modelSpecs !== undefined)
        .map((r) => r.modelSpecs);
      localModels = mergeLocalModels(advertised);
    }
  } catch {
    localModels = []; // discovery failure is non-fatal: fall back to empty
  }
  return ok({ ...base, local_models: localModels });
}

export async function getLtxVersions(): Promise<Response> {
  const svc = catalogService();
  return ok({ versions: svc.ltxVersions() });
}

export async function getLtxRecommendation(): Promise<Response> {
  return ok({ recommendation: catalogService().recommend("ltx") });
}

export async function getIcLoraRecommendation(): Promise<Response> {
  return ok({ recommendation: catalogService().recommend("ltx-ic-lora") });
}

export async function getImgGenRecommendation(): Promise<Response> {
  return ok({ recommendation: catalogService().recommend("img-gen") });
}

export async function getTextEncoderRecommendation(): Promise<Response> {
  return ok({ recommendation: catalogService().recommend("text-encoder") });
}

export async function getLoras(): Promise<Response> {
  // LoraListResponse: never downloaded on-device (the web app has no local weights).
  return ok({ loras: catalogService().loras().map((l) => ({ downloaded: false, downloaded_variant_ids: [], lora: l })) });
}

export async function getIcLoras(): Promise<Response> {
  // IcLoraListResponse: `ic_loras` + IcLoraListItem with the ic_lora key.
  return ok({ ic_loras: catalogService().icLoras().map((l) => ({ downloaded: false, downloaded_variant_ids: [], ic_lora: l })) });
}

/** Local catalog snapshot (used by the smoke test + future cached fetch). */
export async function getLocalCatalog(): Promise<Response> {
  const svc = catalogService();
  return ok({ ...svc.getCatalog() });
}
