import { createDirectSignerProxyHandler, mintUserSignerToken } from "@pymthouse/builder-sdk/signer/server";
import { getExternalUserByKeyHash } from "./ledger";
import { PymtHouseClient } from "./pymthouse";
import { err, json, ok, readJson, sha256Hex } from "./utils";
import type { AuthzSessionRow, Env } from "./types";

// ───────────────────────────────────────────────────────────────────────────
// INVARIANT: every balance debit MUST be authorized through PymtHouse.
// This Worker NEVER debits a balance itself — no direct D1 balance writes,
// no local ledger decrements. The only mechanism that moves money out of a
// user's allowance is PymtHouse: the DMZ signs a ticket on
// `/generate-live-payment`, and PymtHouse's metering performs the debit
// (reserving the expected value) — gated by the allowance and the `sign:job`
// JWT this module mints. Routes here only (a) auth the user, (b) request
// PymtHouse to sign, and (c) gate the sign on a *read* of the PymtHouse
// balance. If you add any code that decreases a stored balance outside
// PymtHouse, you are breaking this invariant — route the debit through
// PymtHouse instead.
// ───────────────────────────────────────────────────────────────────────────

// ---------------------------------------------------------------------------
// Per-user key auth (shared with index.ts) — resolves the Bearer key to a user.
// ---------------------------------------------------------------------------
export async function authUserFromKey(req: Request, env: Env): Promise<{ externalUserId: string } | null> {
  const auth = req.headers.get("authorization") || "";
  const secret = auth.startsWith("Bearer ") ? auth.slice(7) : "";
  if (!secret || !env.DB) return null;
  const hash = await sha256Hex(secret);
  const externalUserId = await getExternalUserByKeyHash(env.DB, hash);
  return externalUserId ? { externalUserId } : null;
}

// ---------------------------------------------------------------------------
// POST /sign-ticket — the PymtHouse DMZ signing proxy.
//
// This is `createDirectSignerProxyHandler`: the Worker mints a `sign:job` JWT
// for the authenticated user, forwards the incoming request to the remote
// signer DMZ `POST /generate-live-payment`, and relays the response back —
// i.e. the exact `{ payment, segCreds, state }` the browser attaches as
// `Livepeer-Payment` / `Livepeer-Segment`. PymtHouse sizes/signs the tickets
// and reserves the allowance (expected value) via its own metering.
// ---------------------------------------------------------------------------

const _proxyCache = new WeakMap<Env, ReturnType<typeof createDirectSignerProxyHandler>>();

// Resolve the direct signer DMZ base URL: explicit REMOTE_SIGNER_URL wins; otherwise
// derive it from PymtHouse /signer/routing (remoteDmzUrl === signerApiUrl === the DMZ
// that serves /generate-live-payment and /sign-orchestrator-info). Cached 5 min.
const _dmzCache = new WeakMap<Env, { url: string; fetchedAt: number }>();
async function resolveRemoteSignerUrl(env: Env): Promise<string> {
  if (typeof env.REMOTE_SIGNER_URL === "string" && env.REMOTE_SIGNER_URL.trim()) {
    return env.REMOTE_SIGNER_URL.trim();
  }
  const cached = _dmzCache.get(env);
  if (cached && Date.now() - cached.fetchedAt < 5 * 60_000) return cached.url;
  const routing = await new PymtHouseClient(env).getSignerRouting();
  const url = (routing.dmzUrl || "").trim();
  if (!url) {
    throw new Error("No signer DMZ URL: set REMOTE_SIGNER_URL or resolve it from PymtHouse /signer/routing");
  }
  _dmzCache.set(env, { url, fetchedAt: Date.now() });
  return url;
}



export async function getSignerProxy(env: Env) {
  const remoteSignerUrl = await resolveRemoteSignerUrl(env);
  let proxy = _proxyCache.get(env);
  if (!proxy) {
    proxy = createDirectSignerProxyHandler({
      pymthouseIssuerUrl: env.PYMTHOUSE_ISSUER_URL,
      pymthouseClientId: env.PYMTHOUSE_PUBLIC_CLIENT_ID,
      pymthouseM2MClientId: env.PYMTHOUSE_M2M_CLIENT_ID,
      pymthouseM2MClientSecret: env.PYMTHOUSE_M2M_CLIENT_SECRET,
      remoteSignerUrl,
      // Browser POSTs to /sign-ticket; empty suffix → defaults to the DMZ's
      // /generate-live-payment. The body (paymentParams/type/manifestId/state)
      // is relayed verbatim to the DMZ.
      proxyPathPrefix: "/sign-ticket",
      defaultRemotePath: "/generate-live-payment",
      authenticate: (req) => authUserFromKey(req, env),
      resolveExternalUserId: async (session) => {
        if (!session || typeof (session as { externalUserId?: unknown }).externalUserId !== "string") {
          throw new Error("Unauthorized: missing externalUserId");
        }
        return (session as { externalUserId: string }).externalUserId;
      },
      // Balance gate: block signing (and thus debit) when the user has no allowance.
      beforeSign: async ({ token }) => {
        const balance = Number(token.balanceUsdMicros ?? "0");
        if (balance <= 0) {
          return {
            status: 402,
            body: {
              ok: false,
              error: "Insufficient balance — add credits to continue",
              balanceUsdMicros: token.balanceUsdMicros ?? "0",
            },
          };
        }
      },
    });
    _proxyCache.set(env, proxy);
  }
  return proxy;
}

// ---------------------------------------------------------------------------
// GET /signer/address — the payer/broadcaster address the browser sends as
// `Livepeer-Payer-Address` so the orchestrator issues the right challenge.
// ---------------------------------------------------------------------------
export async function getPayerAddress(env: Env, externalUserId: string): Promise<Response> {
  let signerBase: string;
  try {
    signerBase = (await resolveRemoteSignerUrl(env)).replace(/\/+$/, "");
  } catch (e) {
    return err(`signer address error: ${(e as Error).message}`, 502);
  }
  try {
    const token = await mintUserSignerToken({
      issuerUrl: env.PYMTHOUSE_ISSUER_URL,
      m2mClientId: env.PYMTHOUSE_M2M_CLIENT_ID,
      m2mClientSecret: env.PYMTHOUSE_M2M_CLIENT_SECRET,
      externalUserId,
    });
    const res = await fetch(`${signerBase}/sign-orchestrator-info`, {
      method: "POST",
      headers: { authorization: `Bearer ${token.jwt}`, "content-type": "application/json" },
      body: "{}",
    });
    if (!res.ok) {
      return err(`signer /sign-orchestrator-info -> ${res.status}`, 502);
    }
    const data = (await res.json()) as { address?: string; signature?: string };
    if (!data.address) return err("signer returned no address", 502);
    return ok({ address: data.address });
  } catch (e) {
    return err(`signer address error: ${(e as Error).message}`, 502);
  }
}

// ---------------------------------------------------------------------------
// POST /authorize — go-livepeer remote-signer identity webhook.
//
// The go-livepeer DMZ calls this (configured via -remoteSignerWebhookUrl +
// -remoteSignerWebhookSecret) on every signing request to verify the end-user
// before it signs. Envelope: { authorization, payload } → 200 { auth_id }.
//
// NOTE: `@livepeer/clearinghouse-identity-webhook` is not published to the
// public npm registry, so this reimplements the contract using Web Crypto
// (RS256 JWT verification against the PymtHouse OIDC JWKS) with no extra deps.
// The shared-secret transport and the exact envelope must be confirmed against
// the go-livepeer remote-signer implementation before go-live.
// ---------------------------------------------------------------------------

export async function handleAuthorize(env: Env, request: Request): Promise<Response> {
  // 1. Shared secret with the go-livepeer DMZ (-remoteSignerWebhookSecret).
  const provided = (request.headers.get("authorization") || "").replace(/^Bearer\s+/i, "");
  if (!env.WEBHOOK_SECRET || provided !== env.WEBHOOK_SECRET) {
    return err("Unauthorized", 401);
  }

  let body: { authorization?: string; payload?: unknown } = {};
  try {
    body = await readJson<typeof body>(request);
  } catch {
    return err("invalid JSON body", 400);
  }

  const jwt = body.authorization?.trim();
  if (!jwt) return err("missing authorization JWT", 400);

  // 2. Verify the end-user signer JWT against the PymtHouse OIDC JWKS.
  let payload: Record<string, unknown>;
  try {
    payload = await verifyEndUserJwt(env, jwt);
  } catch (e) {
    return err(`JWT verification failed: ${(e as Error).message}`, 401);
  }

  if (env.JWT_AUDIENCE && payload.aud !== env.JWT_AUDIENCE) {
    return err("JWT audience mismatch", 401);
  }

  // 3. Resolve the usage identity: auth_id (the externalUserId the JWT is scoped to).
  const authId =
    (payload.sub as string) || (payload.external_user_id as string) || (payload.externalUserId as string) || "";
  if (!authId) return err("JWT missing usage subject", 401);

  return ok({ auth_id: authId, externalUserId: authId });
}

// ---------------------------------------------------------------------------
// RS256 JWT verification against the PymtHouse OIDC JWKS (no external deps).
// ---------------------------------------------------------------------------
// TS's JsonWebKey lacks `kid`; extend it for JWKS lookup.
interface PmthJwk extends JsonWebKey {
  kid?: string;
}

const _jwksCache = new Map<string, { keys: PmthJwk[]; fetchedAt: number }>();

async function verifyEndUserJwt(env: Env, jwt: string): Promise<Record<string, unknown>> {
  const parts = jwt.split(".");
  if (parts.length !== 3) throw new Error("malformed JWT");

  const header = JSON.parse(_b64url(parts[0])) as { kid?: string; alg?: string };
  if (header.alg && !/^RS|RS256|RSASSA/i.test(header.alg)) {
    throw new Error(`unsupported alg ${header.alg}`);
  }
  const kid = header.kid || "";

  const jwks = await _fetchJwks(env, kid);
  let cryptoKey: CryptoKey;
  try {
    cryptoKey = await crypto.subtle.importKey(
      "jwk",
      jwks as JsonWebKey,
      { name: "RSASSA-PKCS1-v1-5", hash: "SHA-256" },
      false,
      ["verify"],
    );
  } catch {
    throw new Error("JWKS key import failed");
  }

  const data = new TextEncoder().encode(`${parts[0]}.${parts[1]}`);
  const sig = _b64urlToBytes(parts[2]);
  const okSig = await crypto.subtle.verify("RSASSA-PKCS1-v1-5", cryptoKey, sig, data);
  if (!okSig) throw new Error("signature verification failed");

  return JSON.parse(_b64url(parts[1])) as Record<string, unknown>;
}

async function _fetchJwks(env: Env, kid: string): Promise<JsonWebKey> {
  const jwksUrl = `${env.PYMTHOUSE_ISSUER_URL.replace(/\/+$/, "")}/jwks`;
  const cached = _jwksCache.get(jwksUrl);
  if (cached && Date.now() - cached.fetchedAt < 5 * 60_000) {
    const hit = cached.keys.find((k) => k.kid === kid);
    if (hit) return hit;
  }
  const res = await fetch(jwksUrl);
  if (!res.ok) throw new Error(`JWKS fetch failed: ${res.status}`);
  const body = (await res.json()) as { keys: PmthJwk[] };
  _jwksCache.set(jwksUrl, { keys: body.keys, fetchedAt: Date.now() });
  const key = body.keys.find((k) => k.kid === kid);
  if (!key) throw new Error(`JWKS missing kid ${kid}`);
  return key;
}

function _b64url(input: string): string {
  const b64 = input.replace(/-/g, "+").replace(/_/g, "/");
  const padded = b64.padEnd(Math.ceil(b64.length / 4) * 4, "=");
  return new TextDecoder().decode(Uint8Array.from(atob(padded), (c) => c.charCodeAt(0)));
}

function _b64urlToBytes(input: string): Uint8Array {
  const b64 = input.replace(/-/g, "+").replace(/_/g, "/");
  const padded = b64.padEnd(Math.ceil(b64.length / 4) * 4, "=");
  return Uint8Array.from(atob(padded), (c) => c.charCodeAt(0));
}

// Silence unused-import lint for the audit row type (kept for future reconciliation).
export type { AuthzSessionRow };
