/**
 * Catalog routes (keep (Worker/catalog) per api-contract.md). All funnel through
 * catalogService() so the backing store can become data-driven later.
 */

import { ok } from "../utils";
import { catalogService, getVideoGenerationModelSpecs } from "../catalog";

export async function getModels(): Promise<Response> {
  // GET /api/models == InstalledModelsResponse (frontend `listModels`): models/loras present on
  // local disk. The web app has none (everything is remote via Worker/orchestrator), so this is
  // genuinely empty; must keep the InstalledModelResponse shape so the frontend's listModels works.
  return ok({ models: [] });
}

/** GET /api/generate/models-specs — available t2v model specs. */
export async function getModelSpecs(): Promise<Response> {
  return ok(getVideoGenerationModelSpecs());
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
