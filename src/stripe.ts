import type { Env, Tier } from "./types";
import { toHex } from "./utils";

const STRIPE_API = "https://api.stripe.com/v1";

/** Create a hosted Checkout Session for a top-up tier. Returns the checkout URL. */
export async function createCheckoutSession(
  env: Env,
  externalUserId: string,
  tier: Tier,
  metadata: Record<string, string>,
): Promise<{ url: string; id: string }> {
  const params = new URLSearchParams();
  params.set("mode", "payment");
  params.set("client_reference_id", externalUserId);
  params.set("success_url", metadata.successUrl || "https://example.com/success");
  params.set("cancel_url", metadata.cancelUrl || "https://example.com/cancel");
  params.append("line_items[0][quantity]", "1");
  params.append("line_items[0][price_data][currency]", "usd");
  params.append("line_items[0][price_data][product_data][name]", `Video Creator Credits — $${(tier.creditsCents / 100).toFixed(2)}`);
  params.append("line_items[0][price_data][unit_amount]", String(tier.creditsCents + tier.feeCents));
  // Carry the credit amount through so the webhook grants exactly the credits (never the fee).
  for (const [k, v] of Object.entries(metadata)) params.append(`metadata[${k}]`, v);
  params.append("metadata[credit_usd_micros]", String(tier.creditsCents * 10000));
  params.append("metadata[tier_credits_cents]", String(tier.creditsCents));

  const res = await fetch(`${STRIPE_API}/checkout/sessions`, {
    method: "POST",
    headers: {
      authorization: `Bearer ${env.STRIPE_SECRET_KEY}`,
      "content-type": "application/x-www-form-urlencoded",
    },
    body: params.toString(),
  });
  const data = (await res.json()) as { url?: string; id?: string; error?: { message: string } };
  if (!res.ok || !data.url) {
    throw new Error(`Stripe checkout failed: ${data.error?.message ?? res.status}`);
  }
  return { url: data.url, id: data.id! };
}

/**
 * Verify a Stripe webhook signature (scheme: `t=...,v1=...`), HMAC-SHA256 of
 * "<timestamp>.<rawBody>" with the webhook secret. Returns the parsed event.
 */
export async function verifyWebhook(rawBody: string, signature: string | null, env: Env): Promise<any> {
  if (!signature) throw new Error("Missing Stripe-Signature header");
  const t = signature.split(",").find((p) => p.startsWith("t="))?.slice(2);
  const v1 = signature.split(",").find((p) => p.startsWith("v1="))?.slice(3);
  if (!t || !v1) throw new Error("Malformed Stripe-Signature");

  const timestamp = Number(t);
  if (!Number.isFinite(timestamp)) throw new Error("Bad timestamp");
  if (Math.abs(Date.now() / 1000 - timestamp) > 300) throw new Error("Timestamp outside tolerance");

  const signedPayload = `${t}.${rawBody}`;
  const key = await crypto.subtle.importKey(
    "raw",
    new TextEncoder().encode(env.STRIPE_WEBHOOK_SECRET),
    { name: "HMAC", hash: "SHA-256" },
    false,
    ["sign"],
  );
  const mac = await crypto.subtle.sign("HMAC", key, new TextEncoder().encode(signedPayload));
  const expected = toHex(new Uint8Array(mac));
  if (!constantTimeEq(expected, v1)) throw new Error("Signature mismatch");

  return JSON.parse(rawBody);
}

function constantTimeEq(a: string, b: string): boolean {
  if (a.length !== b.length) return false;
  let diff = 0;
  for (let i = 0; i < a.length; i++) diff |= a.charCodeAt(i) ^ b.charCodeAt(i);
  return diff === 0;
}

export type StripeLineItem = any; // passthrough for events we don't model strictly
