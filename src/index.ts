import { findTier, publicTiers } from "./config";
import { PymtHouseClient } from "./pymthouse";
import {
  consumeRecoveryCode,
  getAccountByEmail,
  getAccountByExternalUser,
  upsertAccount,
  verifyAccountEmail,
  alreadyApplied,
  markApplied,
  storeRecoveryCode,
} from "./ledger";
import { sendCodeEmail } from "./recovery";
import { createCheckoutSession, verifyWebhook } from "./stripe";
import type { Env } from "./types";
import { err, generateRecoveryCode, hashSecret, isValidEmail, json, ok, readJson, requireApiKey } from "./utils";

export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    const url = new URL(request.url);
    const path = url.pathname;
    const method = request.method;

    // CORS preflight
    if (method === "OPTIONS") return handleOptions(env);

    // Public health
    if (method === "GET" && path === "/health") return ok({ ok: true, ts: Date.now() });

    // Stripe webhook (verified by signature, no API key needed)
    if (method === "POST" && path === "/webhook/stripe") {
      const raw = await request.text();
      try {
        return await handleStripeWebhook(raw, request.headers.get("stripe-signature"), env);
      } catch (e) {
        return err(`Webhook rejected: ${(e as Error).message}`, 400);
      }
    }

    // Phase B: go-livepeer identity webhook (DMZ). Carved out; implement in Phase B.
    if (method === "POST" && path === "/authorize") {
      return json({ ok: false, error: "Phase B DMZ /authorize not yet implemented" }, 501);
    }

    // ---- All routes below require the platform API key ----
    const authFail = requireApiKey(request, env);
    if (authFail) return authFail;

    try {
      if (method === "POST" && path === "/checkout") return await postCheckout(request, env);
      if (method === "GET" && path === "/tiers") return ok({ tiers: publicTiers(env) });
      if (method === "GET" && path === "/balance") return await getBalance(request, env);

      if (method === "POST" && path === "/link-email") return await postLinkEmail(request, env);
      if (method === "POST" && path === "/link-email/verify") return await verifyLinkEmail(request, env);
      if (method === "POST" && path === "/recover/request") return await postRecoverRequest(request, env);
      if (method === "POST" && path === "/recover/confirm") return await postRecoverConfirm(request, env);

      if (method === "GET" && path === "/usage") return await getUsage(request, env);

      return err("Not found", 404);
    } catch (e) {
      const msg = (e as Error).message;
      if (/unauthorized|invalid|not found/i.test(msg)) return err(msg, 400);
      return err(msg || "Server error", 500);
    }
  },
};

// ---------------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------------

type CheckoutBody = { externalUserId: string; tier: number; micros?: string; successUrl?: string; cancelUrl?: string };

async function postCheckout(request: Request, env: Env): Promise<Response> {
  const body = await readJson<CheckoutBody>(request);
  if (!body.externalUserId) return err("externalUserId required", 400);

  // Resolve tier by credits-in-cents (body.tier) OR by credits-in-micros (body.micros).
  const tier = body.micros ? findTier(env, Math.round(Number(body.micros) / 10000)) : findTier(env, body.tier);
  if (!tier) return err("Unknown top-up tier", 400);

  await upsertAccount(env.DB, body.externalUserId);
  const { url } = await createCheckoutSession(env, body.externalUserId, tier, {
    successUrl: body.successUrl || `${new URL(request.url).origin}/success`,
    cancelUrl: body.cancelUrl || `${new URL(request.url).origin}/cancel`,
  });
  return ok({ url });
}

async function getBalance(request: Request, env: Env): Promise<Response> {
  const externalUserId = new URL(request.url).searchParams.get("externalUserId") || "";
  if (!externalUserId) return err("externalUserId required", 400);
  const client = new PymtHouseClient(env);
  const balance = await client.getBalance(externalUserId);
  return ok(balance);
}

type LinkEmailBody = { externalUserId: string; email: string };
async function postLinkEmail(request: Request, env: Env): Promise<Response> {
  const body = await readJson<LinkEmailBody>(request);
  if (!body.externalUserId || !isValidEmail(body.email || "")) return err("externalUserId and valid email required", 400);
  const email = body.email.trim().toLowerCase();

  await upsertAccount(env.DB, body.externalUserId, email);
  const code = generateRecoveryCode();
  // Always store + "send" a code; response is generic to prevent enumeration.
  await storeRecoveryCode(env.DB, email, await hashSecret(code), "link");
  await sendCodeEmail(env, email, "link", code);
  return ok({ ok: true }); // generic
}

type VerifyLinkBody = { externalUserId: string; email: string; code: string };
async function verifyLinkEmail(request: Request, env: Env): Promise<Response> {
  const body = await readJson<VerifyLinkBody>(request);
  if (!body.externalUserId || !body.email || !body.code) return err("externalUserId, email and code required", 400);
  const email = body.email.trim().toLowerCase();
  const { ok: valid } = await consumeRecoveryCode(env.DB, email, body.code, "link");
  if (!valid) return err("Invalid or expired code", 400);
  await verifyAccountEmail(env.DB, body.externalUserId, email);
  return ok({ ok: true, email });
}

type RecoverRequestBody = { email: string };
async function postRecoverRequest(request: Request, env: Env): Promise<Response> {
  const body = await readJson<RecoverRequestBody>(request);
  if (!body.email || !isValidEmail(body.email)) return err("valid email required", 400);
  const email = body.email.trim().toLowerCase();

  const account = await getAccountByEmail(env.DB, email);
  if (account) {
    const code = generateRecoveryCode();
    await storeRecoveryCode(env.DB, email, await hashSecret(code), "recover");
    await sendCodeEmail(env, email, "recover", code);
  }
  // Generic response regardless: prevents account enumeration.
  return ok({ ok: true });
}

type RecoverConfirmBody = { email: string; code: string };
async function postRecoverConfirm(request: Request, env: Env): Promise<Response> {
  const body = await readJson<RecoverConfirmBody>(request);
  if (!body.email || !body.code) return err("email and code required", 400);
  const email = body.email.trim().toLowerCase();

  const { ok: valid, account } = await consumeRecoveryCode(env.DB, email, body.code, "recover");
  if (!valid || !account) return err("Invalid or expired code", 400);
  return ok({ externalUserId: account.external_user_id });
}

async function getUsage(request: Request, env: Env): Promise<Response> {
  const externalUserId = new URL(request.url).searchParams.get("externalUserId") || "";
  if (!externalUserId) return err("externalUserId required", 400);
  const client = new PymtHouseClient(env);
  const balance = await client.getBalance(externalUserId);
  return ok({ externalUserId, balance });
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

    if (!externalUserId || !creditMicros) {
      // Nothing to credit — acknowledge so Stripe stops retrying.
      return json({ received: true, skipped: true });
    }
    // Idempotency: never grant twice for the same event.
    if (await alreadyApplied(env.DB, event.id)) {
      return json({ received: true, duplicate: true });
    }
    const client = new PymtHouseClient(env);
    await client.upsertUser(externalUserId);
    await client.grantAllowance(externalUserId, creditMicros); // grants CREDITS only, never the fee
    await markApplied(env.DB, event.id);
    return json({ received: true, credited: { externalUserId, credit_amount_usd_micros: creditMicros } });
  }

  // checkout.session.expired, payment_intent.*, etc. are acknowledged as no-ops.
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
