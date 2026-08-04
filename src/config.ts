import type { Env, Tier } from "./types";

const DEFAULT_TIERS: Tier[] = [
  { creditsCents: 1000, feeCents: 100 }, //  $10 credits + $1 fee
  { creditsCents: 2500, feeCents: 150 }, //  $25 credits + $1.50 fee
  { creditsCents: 5000, feeCents: 300 }, //  $50 credits + $3 fee
  { creditsCents: 10000, feeCents: 500 }, // $100 credits + $5 fee
];

/** Parse STRIPE_TIERS env JSON, falling back to the default four tiers. */
export function getTiers(env: Env): Tier[] {
  if (!env.STRIPE_TIERS) return DEFAULT_TIERS;
  try {
    const parsed = JSON.parse(env.STRIPE_TIERS) as Tier[];
    if (Array.isArray(parsed) && parsed.every((t) => Number.isFinite(t?.creditsCents) && Number.isFinite(t?.feeCents))) {
      return parsed;
    }
  } catch {
    // fall through to default
  }
  return DEFAULT_TIERS;
}

/** Public tier list for the desktop (credit + fee + total), price shown in dollars. */
export function publicTiers(env: Env) {
  return getTiers(env).map((t) => ({
    creditsUsd: t.creditsCents / 100,
    feeUsd: t.feeCents / 100,
    totalUsd: (t.creditsCents + t.feeCents) / 100,
    creditsUsdMicros: String(t.creditsCents * 10000), // cents -> micros
  }));
}

/** Look up a tier by its credit value in cents. */
export function findTier(env: Env, creditsCents: number): Tier | undefined {
  return getTiers(env).find((t) => t.creditsCents === creditsCents);
}
