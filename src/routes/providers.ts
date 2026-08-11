/**
 * Providers (rework per api-contract.md): backed by REAL orchestrator runner
 * discovery. GET /api/providers returns discovered runners for the user (their
 * saved choice surfaced first); POST /discover re-runs discovery; POST /select
 * persists the chosen provider per-user in D1; POST /exclude removes it.
 */

import { z } from "zod";
import { err, ok } from "../utils";
import { makeOrchestrator, parseBody, resolveUserFromRequest } from "./lib";
import { deleteProvider, getProvider, setProvider } from "../jobs";
import type { Env } from "../types";

const capsSchema = z.object({
  capabilities: z.array(z.string()).optional(),
});
const selectSchema = z.object({
  runnerId: z.string().min(1),
});

/** GET /api/providers — discover ready runners, surface the user's saved choice first. */
export async function getProviders(request: Request, env: Env): Promise<Response> {
  const u = await resolveUserFromRequest(request, env);
  if (!u.ok) return u.response;
  const orch = makeOrchestrator(env);
  const caps = (new URL(request.url).searchParams.get("capabilities") || "").split(",").filter(Boolean);
  let runners;
  try {
    runners = await orch.discoverRunners(caps);
  } catch (e) {
    return err(`discovery failed: ${(e as Error).message}`, 502);
  }
  const chosen = env.DB ? await getProvider(env.DB, u.userId) : null;
  const chosenId = chosen?.id ? String(chosen.id) : null;
  const ordered = [...runners].sort((a, b) => (a.id === chosenId ? -1 : b.id === chosenId ? 1 : 0));
  return ok({ providers: ordered, chosenId });
}

/** POST /api/providers/discover — run discovery now and return fresh ready runners. */
export async function postDiscoverProviders(request: Request, env: Env): Promise<Response> {
  const u = await resolveUserFromRequest(request, env);
  if (!u.ok) return u.response;
  const body = await parseBody(request, capsSchema);
  const caps = body.ok ? body.data.capabilities ?? [] : [];
  const orch = makeOrchestrator(env);
  let runners;
  try {
    runners = await orch.discoverRunners(caps);
  } catch (e) {
    return err(`discovery failed: ${(e as Error).message}`, 502);
  }
  return ok({ providers: runners });
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
