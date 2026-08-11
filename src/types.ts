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
  /** OIDC issuer URL, e.g. https://<app>.pymthouse.example/api/v1/oidc (matches JWT `iss`). */
  PYMTHOUSE_ISSUER_URL: string;
  PYMTHOUSE_PUBLIC_CLIENT_ID: string;
  PYMTHOUSE_M2M_CLIENT_ID: string;
  PYMTHOUSE_M2M_CLIENT_SECRET: string;

  // Remote signer DMZ (Phase B)
  /** Direct signer DMZ base URL, e.g. https://dmz.example.com (from PymtHouse /signer/routing). */
  REMOTE_SIGNER_URL: string;

  // Operator/admin auth (separate, more privileged key; NOT shipped to the desktop).
  // User routes authenticate with PER-USER keys provisioned via POST /provision.
  ADMIN_API_KEY: string;

  // Email (recovery codes)
  RESEND_API_KEY: string;
  EMAIL_FROM: string;

  // Phase B (go-livepeer DMZ identity webhook) — shared with -remoteSignerWebhookSecret
  WEBHOOK_SECRET: string;
  /** Expected `aud` on end-user signer JWTs (optional; skip check if unset). */
  JWT_AUDIENCE?: string;

  ALLOWED_ORIGIN?: string;

  // ---- Video Creator API + dispatch layer (Phase 2) ----
  /** Livepeer orchestrator base URL (e.g. https://orch.example.com). Used by the
   *  orchestrator client for runner discovery + job submission. */
  ORCHESTRATOR_BASE_URL?: string;
  /** Cost charged (USD micros) per dispatched generation job, debited from the
   *  ledger before dispatch. Set as a secret; defaults to 100000 ($0.10). */
  JOB_COST_USD_MICROS?: string;

  // Hugging Face OAuth (Worker-served flow)
  HF_CLIENT_ID?: string;
  HF_CLIENT_SECRET?: string;
  HF_REDIRECT_URI?: string;
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

/** A signed payment-authorization scope for one generation job (audit). */
export interface AuthzSessionRow {
  id: string;
  external_user_id: string;
  job_id: string;
  orchestrator_id: string;
  max_face_value_usd_micros: number;
  expires_at: string;
  status: string;
  created_at: string;
}

/** A signed ticket (audit). */
export interface TicketRow {
  ticket_hash: string;
  session_id: string;
  face_value_usd_micros: number;
  signed_at: string;
  status: string;
  redeemed_at: string | null;
}
