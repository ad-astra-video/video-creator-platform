/**
 * Hugging Face OAuth (worker OAuth per api-contract.md): a redirect-based flow
 * served entirely by the Worker.
 *
 *   POST /api/auth/huggingface/login   -> { url: huggingface.co/oauth/authorize }
 *   GET  /api/auth/huggingface/callback -> exchange ?code for a token (fetch to
 *                                         hf.co/oauth/token), store per-user
 *   GET  /api/auth/huggingface/status   -> boolean (token present), never the raw value
 *   POST /api/auth/huggingface/logout   -> delete the stored token
 *
 * Secrets: HF_CLIENT_ID / HF_CLIENT_SECRET / HF_REDIRECT_URI (set with
 * `wrangler secret put`; commented in wrangler.toml).
 */

import { cryptoRandomHex, err, ok, readJson, sha256Hex } from "../utils";
import { clearHfToken, getHfTokenRow, getOwnedJob, storeHfToken } from "../jobs";
import { getExternalUserByKeyHash } from "../ledger";
import { resolveUserFromRequest } from "./lib";
import type { Env } from "../types";

const HF_AUTHORIZE = "https://huggingface.co/oauth/authorize";
const HF_TOKEN = "https://huggingface.co/oauth/token";

/** Start the OAuth flow: return a redirect URL to huggingface.co/oauth/authorize. */
export async function postHfLogin(request: Request, env: Env): Promise<Response> {
  const u = await resolveUserFromRequest(request, env);
  if (!u.ok) return u.response;
  const conf = makeHfConf(env);
  if (!conf) return err("Hugging Face OAuth is not configured (HF_CLIENT_ID / HF_REDIRECT_URI missing)", 503);

  const state = cryptoRandomHex(16); // CSRF: bound to this user's key in the callback
  const params = new URLSearchParams({
    client_id: conf.clientId,
    redirect_uri: conf.redirectUri,
    response_type: "code",
    scope: "read",
    state: `${u.userId}:${state}`,
  });
  return ok({ url: `${HF_AUTHORIZE}?${params.toString()}`, state });
}

/** Exchange the ?code for a token and store it per-user. */
export async function getHfCallback(request: Request, env: Env): Promise<Response> {
  const url = new URL(request.url);
  const code = url.searchParams.get("code");
  const state = url.searchParams.get("state");
  const conf = makeHfConf(env);
  if (!conf) return err("Hugging Face OAuth is not configured (HF_CLIENT_ID / HF_REDIRECT_URI missing)", 503);

  // state encodes `userId:nonce`. Resolve the user from their key (the caller
  // authed when starting the flow), then exchange the code server-side.
  if (!code || !state) return err("missing code or state", 400);
  const [userId] = state.split(":");
  if (!userId) return err("invalid state", 400);

  const tokenRes = await fetch(HF_TOKEN, {
    method: "POST",
    headers: { "content-type": "application/x-www-form-urlencoded" },
    body: new URLSearchParams({
      grant_type: "authorization_code",
      code,
      redirect_uri: conf.redirectUri,
      client_id: conf.clientId,
      client_secret: conf.clientSecret || "",
    }).toString(),
  });
  if (!tokenRes.ok) {
    return err(`Hugging Face token exchange failed: ${tokenRes.status} ${await tokenRes.text()}`, 502);
  }
  const tokenData = (await tokenRes.json()) as { access_token?: string; user?: { name?: string } };
  if (!tokenData.access_token) return err("Hugging Face token exchange: no access_token", 502);

  if (!env.DB) return err("Server error", 500);
  await storeHfToken(env.DB, userId, tokenData.access_token, tokenData.user?.name);

  // Redirect the browser back to the app with a success marker (no tokens in URL).
  const origin = new URL(request.url).origin;
  return Response.redirect(`${origin}/settings?hf=connected`, 302);
}

/** GET /api/auth/huggingface/status — boolean presence, never the raw token. */
export async function getHfStatus(request: Request, env: Env): Promise<Response> {
  const u = await resolveUserFromRequest(request, env);
  if (!u.ok) return u.response;
  if (!env.DB) return err("Server error", 500);
  const row = await getHfTokenRow(env.DB, u.userId);
  return ok({ connected: !!row, user: row?.user_name ?? null, connectedAt: row?.updated_at ?? null });
}

/** POST /api/auth/huggingface/logout — delete the stored token. */
export async function postHfLogout(request: Request, env: Env): Promise<Response> {
  const u = await resolveUserFromRequest(request, env);
  if (!u.ok) return u.response;
  if (!env.DB) return err("Server error", 500);
  await clearHfToken(env.DB, u.userId);
  return ok({ ok: true });
}

function makeHfConf(env: Env): { clientId: string; clientSecret?: string; redirectUri: string } | null {
  if (!env.HF_CLIENT_ID || !env.HF_REDIRECT_URI) return null;
  return { clientId: env.HF_CLIENT_ID, clientSecret: env.HF_CLIENT_SECRET, redirectUri: env.HF_REDIRECT_URI };
}

// keep import referenced for API completeness (readJson used by parity / future body routes)
export { readJson, sha256Hex, getExternalUserByKeyHash, getOwnedJob };
