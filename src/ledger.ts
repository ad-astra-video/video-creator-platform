import type { AccountRow, Env, PaymentRow, RecoveryCodeRow } from "./types";
import { addMinutesIso, hashSecret, isExpired, nowIso, verifyHash } from "./utils";

/** Upsert an instance account keyed by external_user_id. Returns the row. */
export async function upsertAccount(
  db: D1Database,
  externalUserId: string,
  email?: string | null,
): Promise<AccountRow> {
  await db
    .prepare(
      `INSERT INTO accounts (external_user_id, email, email_verified, last_seen_at)
       VALUES (?1, ?2, 0, ?3)
       ON CONFLICT(external_user_id) DO UPDATE SET
         email = COALESCE(?2, email),
         last_seen_at = ?3`,
    )
    .bind(externalUserId, email ?? null, nowIso())
    .run();
  return (await getAccountByExternalUser(db, externalUserId))!;
}

export async function getAccountByExternalUser(db: D1Database, externalUserId: string): Promise<AccountRow | null> {
  const res = await db
    .prepare("SELECT * FROM accounts WHERE external_user_id = ?1")
    .bind(externalUserId)
    .first<AccountRow>();
  return res ?? null;
}

export async function getAccountByEmail(db: D1Database, email: string): Promise<AccountRow | null> {
  const res = await db.prepare("SELECT * FROM accounts WHERE email = ?1").bind(email.toLowerCase().trim()).first<AccountRow>();
  return res ?? null;
}

/** Bind the account's email after code verification. */
export async function verifyAccountEmail(db: D1Database, externalUserId: string, email: string): Promise<void> {
  await db
    .prepare("UPDATE accounts SET email = ?1, email_verified = 1 WHERE external_user_id = ?2")
    .bind(email.toLowerCase().trim(), externalUserId)
    .run();
}

// ---------------------------------------------------------------------------
// Recovery codes
// ---------------------------------------------------------------------------

export async function storeRecoveryCode(db: D1Database, email: string, codeHash: string, purpose: "link" | "recover"): Promise<void> {
  await db
    .prepare(
      `INSERT INTO recovery_codes (email, code_hash, purpose, expires_at)
       VALUES (?1, ?2, ?3, ?4)`,
    )
    .bind(email.toLowerCase().trim(), codeHash, purpose, addMinutesIso(15))
    .run();
}

/** Find the newest unused, unexpired code for an email+purpose. */
async function findValidCode(db: D1Database, email: string, purpose: string): Promise<RecoveryCodeRow | null> {
  const res = await db
    .prepare(
      "SELECT * FROM recovery_codes WHERE email = ?1 AND purpose = ?2 AND used = 0 ORDER BY id DESC LIMIT 1",
    )
    .bind(email.toLowerCase().trim(), purpose)
    .first<RecoveryCodeRow>();
  if (!res) return null;
  if (isExpired(res.expires_at)) return null;
  return res;
}

/** Consume (mark used) a code; returns true if it was valid and now used. */
export async function consumeRecoveryCode(
  db: D1Database,
  email: string,
  plainCode: string,
  purpose: "link" | "recover",
): Promise<{ ok: boolean; account?: AccountRow }> {
  const row = await findValidCode(db, email, purpose);
  if (!row) return { ok: false };
  if (!(await verifyHash(row.code_hash, plainCode))) return { ok: false };
  await db.prepare("UPDATE recovery_codes SET used = 1 WHERE id = ?1").bind(row.id).run();
  const account = await getAccountByEmail(db, email);
  return { ok: true, account: account ?? undefined };
}

// ---------------------------------------------------------------------------
// Idempotency (Stripe webhook)
// ---------------------------------------------------------------------------

export async function alreadyApplied(db: D1Database, key: string): Promise<boolean> {
  const row = await db.prepare("SELECT key FROM idempotency WHERE key = ?1").bind(key).first<{ key: string }>();
  return !!row;
}

export async function markApplied(db: D1Database, key: string): Promise<void> {
  await db.prepare("INSERT OR IGNORE INTO idempotency (key, applied_at) VALUES (?1, ?2)").bind(key, nowIso()).run();
}

// ---------------------------------------------------------------------------
// Payment / credit audit log (monitoring + admin)
// ---------------------------------------------------------------------------

export type PaymentKind = "topup" | "admin_grant" | "refund";

export interface PaymentLogInput {
  kind: PaymentKind;
  externalUserId: string;
  /** Credits granted in USD micros (never includes the platform fee). */
  amountUsdMicros: string;
  stripeEventId?: string;
  stripeSessionId?: string;
  tierCreditsCents?: number;
  reason?: string;
}

/** Insert one audit row. Called from the Stripe webhook AND admin grants. */
export async function logPayment(db: D1Database, p: PaymentLogInput): Promise<void> {
  await db
    .prepare(
      `INSERT INTO payments (stripe_event_id, stripe_session_id, external_user_id, tier_credits_cents, amount_usd_micros, kind, reason)
       VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)`,
    )
    .bind(
      p.stripeEventId ?? null,
      p.stripeSessionId ?? null,
      p.externalUserId,
      p.tierCreditsCents ?? null,
      p.amountUsdMicros,
      p.kind,
      p.reason ?? null,
    )
    .run();
}

/** Newest credits first. `sinceIso` filters by created_at (optional). */
export async function listPayments(db: D1Database, limit = 50): Promise<PaymentRow[]> {
  const res = await db.prepare("SELECT * FROM payments ORDER BY id DESC LIMIT ?1").bind(limit).all<PaymentRow>();
  return res.results as PaymentRow[];
}

export async function listAccounts(db: D1Database, limit = 100): Promise<AccountRow[]> {
  const res = await db.prepare("SELECT * FROM accounts ORDER BY id DESC LIMIT ?1").bind(limit).all<AccountRow>();
  return res.results as AccountRow[];
}

// ---------------------------------------------------------------------------
// Per-user API keys
// ---------------------------------------------------------------------------

/** Insert a key row. Returns true only if it was newly created (provision-once). */
export async function createApiKey(db: D1Database, externalUserId: string, keyHash: string): Promise<boolean> {
  const r = await db
    .prepare("INSERT OR IGNORE INTO api_keys (external_user_id, key_hash) VALUES (?1, ?2)")
    .bind(externalUserId, keyHash)
    .run();
  return (r.meta.changes ?? 0) > 0;
}

/** Resolve a user from an (unsalted) SHA-256 of their presented bearer key. */
export async function getExternalUserByKeyHash(db: D1Database, keyHash: string): Promise<string | null> {
  const row = await db
    .prepare("SELECT external_user_id FROM api_keys WHERE key_hash = ?1 AND revoked = 0")
    .bind(keyHash)
    .first<{ external_user_id: string }>();
  return row?.external_user_id ?? null;
}

/** Upsert a fresh key (used by email-verified rotation). */
export async function rotateApiKey(db: D1Database, externalUserId: string, keyHash: string): Promise<void> {
  await db
    .prepare(
      `INSERT INTO api_keys (external_user_id, key_hash, revoked) VALUES (?1, ?2, 0)
       ON CONFLICT(external_user_id) DO UPDATE SET key_hash = excluded.key_hash, revoked = 0, created_at = datetime('now')`,
    )
    .bind(externalUserId, keyHash)
    .run();
}

export async function revokeApiKey(db: D1Database, externalUserId: string): Promise<void> {
  await db.prepare("UPDATE api_keys SET revoked = 1 WHERE external_user_id = ?1").bind(externalUserId).run();
}

export async function listApiKeys(db: D1Database): Promise<{ external_user_id: string; created_at: string; last_used_at: string | null; revoked: number }[]> {
  const res = await db.prepare("SELECT external_user_id, created_at, last_used_at, revoked FROM api_keys ORDER BY created_at DESC").all();
  return res.results as any;
}

// ---------------------------------------------------------------------------
// Pending email (link-email two-step)
// ---------------------------------------------------------------------------

export async function setPendingEmail(db: D1Database, externalUserId: string, email: string): Promise<void> {
  await db.prepare("UPDATE accounts SET pending_email = ?1 WHERE external_user_id = ?2").bind(email.toLowerCase().trim(), externalUserId).run();
}

export async function getPendingEmail(db: D1Database, externalUserId: string): Promise<string | null> {
  const row = await db.prepare("SELECT pending_email FROM accounts WHERE external_user_id = ?1").bind(externalUserId).first<{ pending_email: string | null }>();
  return row?.pending_email ?? null;
}

export async function clearPendingEmail(db: D1Database, externalUserId: string): Promise<void> {
  await db.prepare("UPDATE accounts SET pending_email = NULL WHERE external_user_id = ?1").bind(externalUserId).run();
}

// ---------------------------------------------------------------------------
// Phase B: signed-payment audit (authz sessions + tickets)
// ---------------------------------------------------------------------------

export async function createAuthzSession(
  db: D1Database,
  row: { id: string; external_user_id: string; job_id: string; orchestrator_id: string; max_face_value_usd_micros: number; expires_at: string },
): Promise<void> {
  await db
    .prepare(
      `INSERT INTO authz_sessions (id, external_user_id, job_id, orchestrator_id, max_face_value_usd_micros, expires_at)
       VALUES (?1, ?2, ?3, ?4, ?5, ?6)`,
    )
    .bind(row.id, row.external_user_id, row.job_id, row.orchestrator_id, row.max_face_value_usd_micros, row.expires_at)
    .run();
}

/** Record one signed ticket. `INSERT OR IGNORE` makes it idempotent by ticketHash. */
export async function recordSignedTicket(
  db: D1Database,
  row: { ticket_hash: string; session_id: string; face_value_usd_micros: number },
): Promise<void> {
  await db
    .prepare(
      `INSERT OR IGNORE INTO tickets (ticket_hash, session_id, face_value_usd_micros)
       VALUES (?1, ?2, ?3)`,
    )
    .bind(row.ticket_hash, row.session_id, row.face_value_usd_micros)
    .run();
}

// Re-exported for index.ts convenience
export { hashSecret, generateRecoveryCode as _gen } from "./utils";
