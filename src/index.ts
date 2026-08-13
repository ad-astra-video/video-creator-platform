import { findTier, publicTiers } from "./config";
import { PymtHouseClient } from "./pymthouse";
import {
  consumeRecoveryCode,
  getAccountByEmail,
  upsertAccount,
  verifyAccountEmail,
  alreadyApplied,
  markApplied,
  storeRecoveryCode,
  listPayments,
  listAccounts,
  logPayment,
  createApiKey,
  getExternalUserByKeyHash,
  rotateApiKey,
  revokeApiKey,
  listApiKeys,
  setPendingEmail,
  getPendingEmail,
  clearPendingEmail,
  setBackupCode,
  useBackupCode,
} from "./ledger";
import { sendCodeEmail } from "./recovery";
import { createCheckoutSession, verifyWebhook } from "./stripe";
import type { Env } from "./types";
import { getPayerAddress, getSignerProxy, handleAuthorize } from "./signer";
import {
  err,
  generateRecoveryCode,
  generateBackupCode,
  hashSecret,
  isValidEmail,
  json,
  ok,
  readJson,
  requireAdminKey,
  sha256Hex,
  cryptoRandomHex,
} from "./utils";
import { getHealth } from "./routes/health";
import { handleWebSocket } from "./ws";
import {
  postGenerate,
  postGenerateImage,
  postEnhancePrompt,
  postExtend,
  postRetake,
  postRestyle,
  postRestyleExtractFirstFrame,
  postRestyleSegmentSubject,
  postRestyleStyleFrame,
  postIcLoraGenerate,
  postIcLoraExtractConditioning,
  postCancel,
  getGenerationProgress,
  getDownloadProgress,
} from "./routes/generation";
import {
  getModels,
  getModelSpecs,
  getLtxVersions,
  getLtxRecommendation,
  getIcLoraRecommendation,
  getImgGenRecommendation,
  getTextEncoderRecommendation,
  getLoras,
  getIcLoras,
  getLocalCatalog,
} from "./routes/catalog";
import { getProviders, postDiscoverProviders, postSelectProvider, postExcludeProvider } from "./routes/providers";
import { getSettingsRoute, postSettingsRoute, getSettingsFalKey } from "./routes/settings";
import { getPlatformStatus } from "./routes/platform";
import { postHfLogin, getHfCallback, getHfStatus, postHfLogout } from "./routes/hf-auth";

async function handle(request: Request, env: Env): Promise<Response> {
    const url = new URL(request.url);
    const path = url.pathname;
    const method = request.method;

    if (method === "OPTIONS") return handleOptions(env);

    // Liveness (keep-remote).
    if (method === "GET" && path === "/health") return await getHealth();

    // WebSocket relay (authenticates itself via ?token= or Authorization).
    if (path === "/ws") return await handleWebSocket(request, env);

    // Stripe webhook (signature-verified, no API key)
    if (method === "POST" && path === "/webhook/stripe") {
      const raw = await request.text();
      try {
        return await handleStripeWebhook(raw, request.headers.get("stripe-signature"), env);
      } catch (e) {
        return err(`Webhook rejected: ${(e as Error).message}`, 400);
      }
    }

    // Phase B: go-livepeer remote-signer identity webhook (DMZ, shared WEBHOOK_SECRET).
    if (method === "POST" && path === "/authorize") {
      try {
        return await handleAuthorize(env, request);
      } catch (e) {
        return err(`authorize error: ${(e as Error).message}`, 401);
      }
    }

    // ---- Admin routes: use the ADMIN key independently ----
    if (path.startsWith("/admin/")) {
      try {
        const adminFail = requireAdminKey(request, env);
        if (adminFail) return adminFail;
        if (method === "GET" && path === "/admin/payments") return await adminPayments(request, env);
        if (method === "GET" && path === "/admin/users") return await adminUsers(request, env);
        if (method === "GET" && path === "/admin/balance") return await adminBalance(request, env);
        if (method === "POST" && path === "/admin/grant") return await adminGrant(request, env);
        if (method === "GET" && path === "/admin/api-keys") return await adminApiKeys(env);
        if (method === "POST" && path === "/admin/revoke-key") return await adminRevokeKey(request, env);
        return err("Not found", 404);
      } catch (e) {
        return err((e as Error).message || "Server error", 500);
      }
    }

    // ---- Public, identity-free routes ----
    if (method === "GET" && path === "/tiers") return ok({ tiers: publicTiers(env) });
    if (method === "POST" && path === "/provision") return await postProvision(request, env);
    if (method === "POST" && path === "/recover/request") return await postRecoverRequest(request, env);
    if (method === "POST" && path === "/recover/confirm") return await postRecoverConfirm(request, env);
    if (method === "POST" && path === "/recover/backup") return await postRecoverBackup(request, env);
    // Platform-credits contract — public recovery (no current key required).
    if (method === "POST" && path === "/api/platform/recover/request") return await postPlatformRecoverRequest(request, env);
    if (method === "POST" && path === "/api/platform/recover/confirm") return await postPlatformRecoverConfirm(request, env);
    // HF OAuth callback is a redirect target (no per-user key header present).
    if (method === "GET" && path === "/api/auth/huggingface/callback") return await getHfCallback(request, env);

    // ---- User-key authenticated routes (user resolved FROM the key) ----
    const user = await authUser(request, env);
    if (!user) return err("Unauthorized", 401);

    try {
      const uid = user.externalUserId;

      // Runtime policy + GPU info (web app: no local GPU; generation is Worker-dispatched).
      if (method === "GET" && path === "/api/runtime-policy")
        return ok({ force_api_generations: false, wait_for_model_download: false, allow_cuda_backend: true });
      if (method === "GET" && path === "/api/gpu-info")
        return ok({ pretty_name: "none", vram_bytes: 0, torch_backend: "none", cuda_available: false });

      // Worker-reimplemented API surface (api-contract.md).
      // Catalog:
      if (method === "GET" && path === "/api/models") return await getModels();
      if (method === "GET" && path === "/api/generate/models-specs") return await getModelSpecs(request, env);
      if (method === "GET" && path === "/api/models/ltx-versions") return await getLtxVersions();
      if (method === "GET" && path === "/api/models/ltx-recommendation") return await getLtxRecommendation();
      if (method === "GET" && path === "/api/models/ltx-ic-lora-recommendation") return await getIcLoraRecommendation();
      if (method === "GET" && path === "/api/models/img-gen-recommendation") return await getImgGenRecommendation();
      if (method === "GET" && path === "/api/models/text-encoder-recommendation") return await getTextEncoderRecommendation();
      if (method === "GET" && path === "/api/loras") return await getLoras();
      if (method === "GET" && path === "/api/ic-loras") return await getIcLoras();
      if (method === "GET" && path === "/api/catalog") return await getLocalCatalog();

      // Dispatch (worker-dispatch):
      if (method === "POST" && path === "/api/generate") return await postGenerate(request, env);
      if (method === "POST" && path === "/api/generate/cancel") return await postCancel(request, env);
      if (method === "POST" && path === "/api/generate-image") return await postGenerateImage(request, env);
      if (method === "POST" && path === "/api/enhance-prompt") return await postEnhancePrompt(request, env);
      if (method === "POST" && path === "/api/extend") return await postExtend(request, env);
      if (method === "POST" && path === "/api/retake") return await postRetake(request, env);
      if (method === "POST" && path === "/api/restyle") return await postRestyle(request, env);
      if (method === "POST" && path === "/api/restyle/extract-first-frame") return await postRestyleExtractFirstFrame(request, env);
      if (method === "POST" && path === "/api/restyle/segment-subject") return await postRestyleSegmentSubject(request, env);
      if (method === "POST" && path === "/api/restyle/style-frame") return await postRestyleStyleFrame(request, env);
      if (method === "POST" && path === "/api/ic-lora/generate") return await postIcLoraGenerate(request, env);
      if (method === "POST" && path === "/api/ic-lora/extract-conditioning") return await postIcLoraExtractConditioning(request, env);

      // REST progress fallbacks (WS supersedes):
      if (method === "GET" && path === "/api/generation/progress") return await getGenerationProgress(request, env);
      if (
        method === "GET" &&
        (path === "/api/loras/download/progress" ||
          path === "/api/ic-loras/download/progress" ||
          path === "/api/models/download/progress")
      ) {
        return await getDownloadProgress(request, env);
      }

      // Providers (orchestrator discovery):
      if (method === "GET" && path === "/api/providers") return await getProviders(request, env);
      if (method === "POST" && path === "/api/providers/discover") return await postDiscoverProviders(request, env);
      if (method === "POST" && path === "/api/providers/select") return await postSelectProvider(request, env);
      if (method === "POST" && path === "/api/providers/exclude") return await postExcludeProvider(request, env);

      // Settings:
      if (method === "GET" && path === "/api/settings") return await getSettingsRoute(request, env);
      if (method === "POST" && path === "/api/settings") return await postSettingsRoute(request, env);
      // Raw FAL key for the DIRECT fal.run path (direct-transport design).
      if (method === "GET" && path === "/api/settings/fal-key") return await getSettingsFalKey(request, env);

      // Platform:
      if (method === "GET" && path === "/api/platform/status") return await getPlatformStatus(request, env);
      if (method === "GET" && path === "/api/platform/balance") return await getPlatformBalance(env, uid);
      if (method === "POST" && path === "/api/platform/checkout") return await postPlatformCheckout(request, env, uid);
      if (method === "POST" && path === "/api/platform/link-email") return await postPlatformLinkEmail(request, env, uid);

      // Hugging Face auth (authed endpoints; callback handled above):
      if (method === "POST" && path === "/api/auth/huggingface/login") return await postHfLogin(request, env);
      if (method === "GET" && path === "/api/auth/huggingface/status") return await getHfStatus(request, env);
      if (method === "POST" && path === "/api/auth/huggingface/logout") return await postHfLogout(request, env);

      // Existing money/auth routes (unchanged):
      if (method === "POST" && path === "/checkout") return await postCheckout(request, env, uid);
      if (method === "GET" && path === "/balance") return await getBalance(env, uid);
      if (method === "GET" && path === "/usage") return await getUsage(env, uid);
      if (method === "POST" && path === "/sign-ticket") return await (await getSignerProxy(env))(request);
      if (method === "GET" && path === "/signer/address") return await getPayerAddress(env, uid);
      if (method === "POST" && path === "/link-email") return await postLinkEmail(request, env, uid);
      if (method === "POST" && path === "/link-email/verify") return await verifyLinkEmail(request, env, uid);

      return err("Not found", 404);
    } catch (e) {
      const msg = (e as Error).message;
      if (/unauthorized|invalid|not found/i.test(msg)) return err(msg, 400);
      return err(msg || "Server error", 500);
    }
}

// CORS: apply allow-origin on ALL responses (not just the OPTIONS preflight) so a
// browser app served on a different origin can call this Worker directly.
function withCors(request: Request, env: Env, res: Response): Response {
  const origin = request.headers.get("origin");
  if (!origin) return res;
  // Clone into a NEW Response: a proxied response pulled from an upstream
  // fetch() (e.g. the PymtHouse DMZ in createDirectSignerProxyHandler) has
  // IMMUTABLE headers, so mutating res.headers would throw "Can't modify
  // immutable headers" and turn every such route into a bare 500 with no ACAO.
  const headers = new Headers(res.headers);
  headers.set("access-control-allow-origin", origin);
  headers.set("access-control-expose-headers", "content-type, content-length");
  headers.set("vary", "Origin");
  return new Response(res.body, {
    status: res.status,
    statusText: res.statusText,
    headers,
  });
}

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    const res = await handle(request, env);
    return withCors(request, env, res);
  },
};

// ---------------------------------------------------------------------------
// Auth middleware (per-user key)
// ---------------------------------------------------------------------------

async function authUser(req: Request, env: Env): Promise<{ externalUserId: string } | null> {
  const auth = req.headers.get("authorization") || "";
  const secret = auth.startsWith("Bearer ") ? auth.slice(7) : "";
  if (!secret || !env.DB) return null;
  const hash = await sha256Hex(secret);
  const externalUserId = await getExternalUserByKeyHash(env.DB, hash);
  return externalUserId ? { externalUserId } : null;
}

// ---------------------------------------------------------------------------
// Provisioning
// ---------------------------------------------------------------------------

type ProvisionBody = { externalUserId: string };
/** First run of an install: mint a 256-bit per-user key, return it ONCE. */
async function postProvision(request: Request, env: Env): Promise<Response> {
  const body = await readJson<ProvisionBody>(request);
  if (!body?.externalUserId) return err("externalUserId required", 400);
  if (!env.DB) return err("Server error", 500);

  await upsertAccount(env.DB, body.externalUserId);
  const provisionedKey = cryptoRandomHex(32); // 64 hex chars, 256 bits
  const inserted = await createApiKey(env.DB, body.externalUserId, await sha256Hex(provisionedKey));
  if (!inserted) {
    // Already provisioned — refuse to hand out a new key for this user (prevents
    // an attacker from seizing a balance by re-provisioning a known UUID).
    return err("API key already provisioned for this externalUserId; rotate via /recover", 409);
  }
  // Optional no-email recovery: mint a one-time backup code, store only its hash,
  // and return the plaintext ONCE so the user can save it (never emailed/sent again).
  const backupCode = generateBackupCode();
  await setBackupCode(env.DB, body.externalUserId, await sha256Hex(backupCode));

  return ok({ apiKey: provisionedKey, externalUserId: body.externalUserId, backupCode });
}

// ---------------------------------------------------------------------------
// User handlers
// ---------------------------------------------------------------------------

type CheckoutBody = { tier?: number; micros?: string; successUrl?: string; cancelUrl?: string };
async function postCheckout(request: Request, env: Env, externalUserId: string): Promise<Response> {
  const body = await readJson<CheckoutBody>(request);
  const tier = body.micros ? findTier(env, Math.round(Number(body.micros) / 10000)) : findTier(env, body.tier ?? NaN);
  if (!tier) return err("Unknown top-up tier", 400);

  await upsertAccount(env.DB, externalUserId);
  const { url } = await createCheckoutSession(env, externalUserId, tier, {
    successUrl: body.successUrl || `${new URL(request.url).origin}/success`,
    cancelUrl: body.cancelUrl || `${new URL(request.url).origin}/cancel`,
  });
  return ok({ url });
}

async function getBalance(env: Env, externalUserId: string): Promise<Response> {
  return ok(await new PymtHouseClient(env).getBalance(externalUserId));
}

async function getUsage(env: Env, externalUserId: string): Promise<Response> {
  return ok({ externalUserId, balance: await new PymtHouseClient(env).getBalance(externalUserId) });
}

type LinkEmailBody = { email: string };
async function postLinkEmail(request: Request, env: Env, externalUserId: string): Promise<Response> {
  const body = await readJson<LinkEmailBody>(request);
  if (!body?.email || !isValidEmail(body.email)) return err("valid email required", 400);
  const email = body.email.trim().toLowerCase();

  await setPendingEmail(env.DB, externalUserId, email);
  const code = generateRecoveryCode();
  await storeRecoveryCode(env.DB, email, await hashSecret(code), "link");
  await sendCodeEmail(env, email, "link", code);
  return ok({ ok: true });
}

type VerifyLinkBody = { code: string };
async function verifyLinkEmail(request: Request, env: Env, externalUserId: string): Promise<Response> {
  const body = await readJson<VerifyLinkBody>(request);
  const pending = await getPendingEmail(env.DB, externalUserId);
  if (!body?.code || !pending) return err("Invalid or expired code", 400);

  const { ok: valid } = await consumeRecoveryCode(env.DB, pending, body.code, "link");
  if (!valid) return err("Invalid or expired code", 400);
  await verifyAccountEmail(env.DB, externalUserId, pending);
  await clearPendingEmail(env.DB, externalUserId);
  return ok({ ok: true, email: pending });
}

type RecoverRequestBody = { email: string };
async function postRecoverRequest(request: Request, env: Env): Promise<Response> {
  const body = await readJson<RecoverRequestBody>(request);
  if (!body?.email || !isValidEmail(body.email)) return err("valid email required", 400);
  const email = body.email.trim().toLowerCase();

  const account = await getAccountByEmail(env.DB, email);
  if (account) {
    const code = generateRecoveryCode();
    await storeRecoveryCode(env.DB, email, await hashSecret(code), "recover");
    await sendCodeEmail(env, email, "recover", code);
  }
  return ok({ ok: true }); // generic — no account enumeration
}

type RecoverConfirmBody = { email: string; code: string };
async function postRecoverConfirm(request: Request, env: Env): Promise<Response> {
  const body = await readJson<RecoverConfirmBody>(request);
  if (!body?.email || !body.code) return err("email and code required", 400);
  const email = body.email.trim().toLowerCase();

  const { ok: valid, account } = await consumeRecoveryCode(env.DB, email, body.code, "recover");
  if (!valid || !account) return err("Invalid or expired code", 400);

  const secret = cryptoRandomHex(32);
  await rotateApiKey(env.DB, account.external_user_id, await sha256Hex(secret));
  return ok({ externalUserId: account.external_user_id, apiKey: secret });
}

type RecoverBackupBody = { code: string };
/**
 * No-email recovery: the user presents the one-time backup code they saved at sign-up.
 * (It is never emailed.) On success: rotate to a fresh API key AND mint a NEW backup
 * code (returned once) so the user can save the next one. The presented code is
 * consumed, so it cannot be reused.
 */
async function postRecoverBackup(request: Request, env: Env): Promise<Response> {
  const body = await readJson<RecoverBackupBody>(request);
  if (!body?.code || !env.DB) return err("code required", 400);

  const secret = cryptoRandomHex(32);
  const keyHash = await sha256Hex(secret);
  const { ok: matched, account } = await useBackupCode(env.DB, body.code, keyHash);
  if (!matched || !account) return err("Invalid recovery code", 400);

  // Rotate succeeded and the used code was cleared. Mint the next one-time code.
  const nextBackup = generateBackupCode();
  await setBackupCode(env.DB, account.external_user_id, await sha256Hex(nextBackup));
  return ok({ externalUserId: account.external_user_id, apiKey: secret, backupCode: nextBackup });
}

// ---------------------------------------------------------------------------
// Platform credits contract (/api/platform/*) — thin shapes over the worker-native
// money/auth logic so the web CreditsPanel works without a desktop backend. The
// handlers stay graceful when PymtHouse / Stripe / email are not configured (local
// dev): they report `configured: false` (or a valid empty result) instead of 502.
// ---------------------------------------------------------------------------

async function getPlatformBalance(env: Env, externalUserId: string): Promise<Response> {
  try {
    const balance = await new PymtHouseClient(env).getBalance(externalUserId);
    return ok({ ...balance, configured: true });
  } catch {
    return ok({
      hasAccess: false,
      balanceUsdMicros: "0",
      remainingUsdMicros: "0",
      consumedUsdMicros: "0",
      lifetimeGrantedUsdMicros: "0",
      configured: false,
    });
  }
}

type PlatformCheckoutBody = { tier?: number };
async function postPlatformCheckout(request: Request, env: Env, externalUserId: string): Promise<Response> {
  const body = await readJson<PlatformCheckoutBody>(request);
  const tier = findTier(env, body?.tier ?? NaN);
  if (!tier) return err("Unknown top-up tier", 400);
  await upsertAccount(env.DB, externalUserId);
  const { url } = await createCheckoutSession(env, externalUserId, tier, {});
  return ok({ url, configured: true });
}

async function postPlatformLinkEmail(request: Request, env: Env, externalUserId: string): Promise<Response> {
  const body = await readJson<LinkEmailBody>(request);
  if (!body?.email || !isValidEmail(body.email)) return err("valid email required", 400);
  const email = body.email.trim().toLowerCase();
  await setPendingEmail(env.DB, externalUserId, email);
  const code = generateRecoveryCode();
  await storeRecoveryCode(env.DB, email, await hashSecret(code), "link");
  await sendCodeEmail(env, email, "link", code);
  return ok({ ok: true, configured: true });
}

async function postPlatformRecoverRequest(request: Request, env: Env): Promise<Response> {
  const body = await readJson<RecoverRequestBody>(request);
  if (!body?.email || !isValidEmail(body.email)) return err("valid email required", 400);
  const email = body.email.trim().toLowerCase();
  const account = await getAccountByEmail(env.DB, email);
  if (account) {
    const code = generateRecoveryCode();
    await storeRecoveryCode(env.DB, email, await hashSecret(code), "recover");
    await sendCodeEmail(env, email, "recover", code);
  }
  return ok({ status: "sent" }); // generic — no account enumeration
}

async function postPlatformRecoverConfirm(request: Request, env: Env): Promise<Response> {
  const body = await readJson<RecoverConfirmBody>(request);
  if (!body?.email || !body.code) return err("email and code required", 400);
  const email = body.email.trim().toLowerCase();
  const { ok: valid, account } = await consumeRecoveryCode(env.DB, email, body.code, "recover");
  if (!valid || !account) return err("Invalid or expired code", 400);
  const secret = cryptoRandomHex(32);
  await rotateApiKey(env.DB, account.external_user_id, await sha256Hex(secret));
  return ok({ hasApiKey: true, configured: true, externalUserId: account.external_user_id, apiKey: secret });
}

// ---------------------------------------------------------------------------
// Admin
// ---------------------------------------------------------------------------

async function adminPayments(request: Request, env: Env): Promise<Response> {
  const limit = Math.min(Number(new URL(request.url).searchParams.get("limit") || "50") || 50, 500);
  return ok({ payments: await listPayments(env.DB, limit) });
}

async function adminUsers(request: Request, env: Env): Promise<Response> {
  const limit = Math.min(Number(new URL(request.url).searchParams.get("limit") || "100") || 100, 1000);
  return ok({ users: await listAccounts(env.DB, limit) });
}

async function adminBalance(request: Request, env: Env): Promise<Response> {
  const externalUserId = new URL(request.url).searchParams.get("externalUserId") || "";
  if (!externalUserId) return err("externalUserId required", 400);
  return ok({ externalUserId, balance: await new PymtHouseClient(env).getBalance(externalUserId) });
}

async function adminApiKeys(env: Env): Promise<Response> {
  return ok({ apiKeys: await listApiKeys(env.DB) });
}

async function adminRevokeKey(request: Request, env: Env): Promise<Response> {
  const body = await readJson<{ externalUserId: string }>(request);
  if (!body?.externalUserId) return err("externalUserId required", 400);
  await revokeApiKey(env.DB, body.externalUserId);
  return ok({ ok: true, externalUserId: body.externalUserId, note: "key revoked; user must re-provision via /recover/confirm" });
}

type AdminGrantBody = { externalUserId: string; amountUsdMicros: string; reason?: string };
async function adminGrant(request: Request, env: Env): Promise<Response> {
  const body = await readJson<AdminGrantBody>(request);
  if (!body?.externalUserId) return err("externalUserId required", 400);
  const micros = BigInt(body.amountUsdMicros ?? "");
  if (micros <= 0n) return err("amountUsdMicros must be a positive integer", 400);

  const client = new PymtHouseClient(env);
  await client.upsertUser(body.externalUserId);
  await client.grantAllowance(body.externalUserId, String(micros));
  await logPayment(env.DB, { kind: "admin_grant", externalUserId: body.externalUserId, amountUsdMicros: String(micros), reason: body.reason });
  return ok({ ok: true, externalUserId: body.externalUserId, amountUsdMicros: String(micros) });
}

// ---------------------------------------------------------------------------
// Stripe webhook
// ---------------------------------------------------------------------------

async function handleStripeWebhook(raw: string, signature: string | null, env: Env): Promise<Response> {
  const event = await verifyWebhook(raw, signature, env);

  if (event.type === "checkout.session.completed") {
    const s = event.data.object as any;
    const externalUserId: string | undefined = s.client_reference_id;
    const creditMicros: string | undefined = s.metadata?.credit_usd_micros;

    if (!externalUserId || !creditMicros) return json({ received: true, skipped: true });
    if (await alreadyApplied(env.DB, event.id)) return json({ received: true, duplicate: true });

    const client = new PymtHouseClient(env);
    await client.upsertUser(externalUserId);
    await client.grantAllowance(externalUserId, creditMicros); // CREDITS only, never the fee
    await markApplied(env.DB, event.id);
    await logPayment(env.DB, {
      kind: "topup",
      externalUserId,
      amountUsdMicros: creditMicros,
      stripeEventId: event.id,
      stripeSessionId: s.id,
      tierCreditsCents: s.metadata?.tier_credits_cents ? Number(s.metadata.tier_credits_cents) : undefined,
    });
    return json({ received: true, credited: { externalUserId, credit_amount_usd_micros: creditMicros } });
  }

  return json({ received: true });
}

// ---------------------------------------------------------------------------
// CORS
// ---------------------------------------------------------------------------

function handleOptions(env: Env): Response {
  const allow = env.ALLOWED_ORIGIN || "*";
  return new Response(null, {
    status: 204,
    headers: {
      "access-control-allow-origin": allow,
      "access-control-allow-methods": "GET, POST, OPTIONS",
      "access-control-allow-headers": "authorization, content-type",
      "access-control-max-age": "86400",
    },
  });
}
