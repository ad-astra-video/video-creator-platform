/** Environment bindings + shared types. */

export interface Env {
  DB: D1Database;

  // Stripe
  STRIPE_SECRET_KEY: string;
  STRIPE_WEBHOOK_SECRET: string;
  /** JSON array of tiers: [{"creditsCents":1000,"feeCents":100}, ...] */
  STRIPE_TIERS: string;

  // PymtHouse (Builder API)
  /** e.g. https://<app>.pymthouse.example  (no trailing slash) */
  PYMTHOUSE_BASE_URL: string;
  PYMTHOUSE_PUBLIC_CLIENT_ID: string;
  PYMTHOUSE_M2M_CLIENT_ID: string;
  PYMTHOUSE_M2M_CLIENT_SECRET: string;

  // Operator/admin auth (separate, more privileged key; NOT shipped to the desktop).
  // User routes authenticate with PER-USER keys provisioned via POST /provision.
  ADMIN_API_KEY: string;

  // Email (recovery codes)
  RESEND_API_KEY: string;
  EMAIL_FROM: string;

  // Phase B (DMZ signing) — optional
  WEBHOOK_SECRET?: string;

  ALLOWED_ORIGIN?: string;
}

/** A single top-up tier definition. */
export interface Tier {
  /** Credit value in cents (e.g. 1000 = $10 of credits). */
  creditsCents: number;
  /** Platform fee in cents (e.g. 100 = $1). */
  feeCents: number;
}

export interface AccountRow {
  id: number;
  external_user_id: string;
  email: string | null;
  email_verified: number;
  created_at: string;
  last_seen_at: string | null;
}

export interface RecoveryCodeRow {
  id: number;
  email: string;
  code_hash: string;
  purpose: string;
  expires_at: string;
  used: number;
}

export interface PaymentRow {
  id: number;
  stripe_event_id: string | null;
  stripe_session_id: string | null;
  external_user_id: string;
  tier_credits_cents: number | null;
  amount_usd_micros: number;
  kind: string;
  reason: string | null;
  created_at: string;
}

export interface Balance {
  hasAccess: boolean;
  balanceUsdMicros: string;
  remainingUsdMicros: string;
  consumedUsdMicros: string;
  lifetimeGrantedUsdMicros: string;
}
