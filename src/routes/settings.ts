/**
 * Settings (keep (Worker/D1, per-user)): per-user app settings stored as JSON in D1.
 *
 * The vendored desktop frontend expects the desktop backend's FLAT settings contract:
 * GET /api/settings returns the settings object at the top level
 * (e.g. `{ livepeerDiscoveryUrl, remoteInferenceEnabled, ... })`, and updates POST the
 * same flat object. We match that exactly so settings save and re-load through the UI.
 */

import { err, ok } from "../utils";
import { getSettings, setSettings } from "../jobs";
import { resolveUserFromRequest } from "./lib";
import type { Env } from "../types";

export async function getSettingsRoute(request: Request, env: Env): Promise<Response> {
  const u = await resolveUserFromRequest(request, env);
  if (!u.ok) return u.response;
  if (!env.DB) return err("Server error", 500);
  const stored = await getSettings(env.DB, u.userId);
  return ok({
    ...stored,
    // Serverless web app: inference is ALWAYS remote (Worker -> orchestrator -> runners),
    // so report remote mode so the desktop's local-install / API-key first-run gates are skipped.
    livepeerDiscoveryUrl: (stored.livepeerDiscoveryUrl as string) ?? "",
    remoteInferenceEnabled: true,
    hasLivepeerDiscoveryUrl: Boolean(stored.livepeerDiscoveryUrl),
    hasLtxApiKey: false,
    hasFalApiKey: false,
  });
}

export async function postSettingsRoute(request: Request, env: Env): Promise<Response> {
  const u = await resolveUserFromRequest(request, env);
  if (!u.ok) return u.response;
  if (!env.DB) return err("Server error", 500);

  let body: unknown;
  try {
    body = await request.json();
  } catch {
    return err("Invalid JSON body", 400);
  }
  if (!body || typeof body !== "object" || Array.isArray(body)) {
    return err("settings must be a JSON object", 400);
  }

  // Merge so a partial save (e.g. just the Discovery URL) never wipes other settings.
  const stored = await getSettings(env.DB, u.userId);
  const merged: Record<string, unknown> = { ...stored, ...(body as Record<string, unknown>) };
  await setSettings(env.DB, u.userId, merged);
  return ok(merged);
}

/**
 * GET /api/settings/fal-key — return the raw FAL API key to its authenticated owner.
 *
 * Under the direct-transport design (plans/20260811_direct_transport.md) the browser calls FAL
 * (fal.run) DIRECTLY for image generation, so it needs the raw key. The webapp threat model
 * already keeps the per-user platform key client-side (same trust as desktop), so returning the
 * key to the authenticated owner is acceptable and keeps the Worker the single source of config.
 * This endpoint never returns the key to anyone but the owner (per-user Bearer key auth).
 */
export async function getSettingsFalKey(request: Request, env: Env): Promise<Response> {
  const u = await resolveUserFromRequest(request, env);
  if (!u.ok) return u.response;
  if (!env.DB) return err("Server error", 500);
  const stored = await getSettings(env.DB, u.userId);
  const key = (stored as Record<string, unknown>).falApiKey;
  return ok({ falApiKey: typeof key === "string" ? key : "", hasFalApiKey: typeof key === "string" && key.length > 0 });
}
