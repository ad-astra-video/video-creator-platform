/**
 * Catalog routes (keep (Worker/catalog) per api-contract.md). All funnel through
 * catalogService() so the backing store can become data-driven later.
 */

import { ok } from "../utils";
import { catalogService } from "../catalog";

export async function getModels(): Promise<Response> {
  const svc = catalogService();
  return ok({ models: svc.models() });
}

/** GET /api/generate/models-specs — available t2v model specs. */
export async function getModelSpecs(): Promise<Response> {
  const svc = catalogService();
  return ok({ specs: svc.modelSpecs() });
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
  return ok({ loras: catalogService().loras() });
}

export async function getIcLoras(): Promise<Response> {
  return ok({ icLoras: catalogService().icLoras() });
}

/** Local catalog snapshot (used by the smoke test + future cached fetch). */
export async function getLocalCatalog(): Promise<Response> {
  const svc = catalogService();
  return ok({ ...svc.getCatalog() });
}
