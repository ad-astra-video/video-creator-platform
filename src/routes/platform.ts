/**
 * Platform-credits surface for the web CreditsPanel (/api/platform/* contract).
 *
 * These are the shapes the frontend CreditsPanel already expects (the desktop app
 * used to get them from a Python backend that proxied the Worker). In the direct
 * web build there is no desktop backend, so the Worker serves the same contract
 * directly:
 *
 *   GET  /api/platform/status         -> { userId, configured, hasApiKey, baseUrl }
 *   GET  /api/platform/balance        -> Balance + { configured }
 *   POST /api/platform/checkout       -> { url, configured }        (body { tier: cents })
 *   POST /api/platform/link-email     -> { configured }             (body { email })
 *   POST /api/platform/recover/request  -> { status }               (public, body { email })
 *   POST /api/platform/recover/confirm  -> { hasApiKey, configured }(public, body { email, code })
 *
 * All of them stay graceful when PymtHouse / Stripe / email are not configured in the
 * local environment: they report `configured: false` (or a valid empty result) instead
 * of 502/404, so the panel renders a clean "not configured" state.
 */

import { err, ok } from "../utils";
import { PymtHouseClient } from "../pymthouse";
import { resolveUserFromRequest } from "./lib";
import type { Env } from "../types";

export async function getPlatformStatus(request: Request, env: Env): Promise<Response> {
  const u = await resolveUserFromRequest(request, env);
  if (!u.ok) return u.response;

  let balance = null;
  try {
    balance = await new PymtHouseClient(env).getBalance(u.userId);
  } catch {
    // PymtHouse not configured / unreachable — report "not configured" instead of 502.
  }

  return ok({
    ok: true,
    userId: u.userId,
    configured: balance !== null,
    hasApiKey: true,
    baseUrl: new URL(request.url).origin,
  });
}
