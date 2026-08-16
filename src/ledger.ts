import type { AccountRow, Env, PaymentRow, RecoveryCodeRow } from "./types";
import { addMinutesIso, hashSecret, isExpired, nowIso, verifyHash, sha256Hex } from "./utils";

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
// Backup recovery code (no-email recovery; optional alternative to email codes)
// ---------------------------------------------------------------------------

/** Store the (salted-hash) backup code for an account. */
export async function setBackupCode(db: D1Database, externalUserId: string, codeHash: string): Promise<void> {
  await db
    .prepare("UPDATE accounts SET backup_code_hash = ?1 WHERE external_user_id = ?2")
    .bind(codeHash, externalUserId)
    .run();
}

/** Find an account whose stored backup-code hash equals the presented (hashed) code. */
export async function getAccountByBackupHash(db: D1Database, codeHash: string): Promise<AccountRow | null> {
  const res = await db
    .prepare("SELECT * FROM accounts WHERE backup_code_hash = ?1")
    .bind(codeHash)
    .first<AccountRow>();
  return res ?? null;
}

/**
 * Recover by backup code (no email needed): verify the hash, rotate the API key, and
 * clear the used backup code. Returns whether it matched a real account.
 */
export async function useBackupCode(
  db: D1Database,
  plainCode: string,
  keyHash: string,
): Promise<{ ok: boolean; account?: AccountRow }> {
  const plainHash = await sha256Hex(plainCode.toUpperCase());
  const account = await getAccountByBackupHash(db, plainHash);
  if (!account) return { ok: false };
  // Rotate to the freshly-minted key, then clear the used backup code (one-time use).
  await rotateApiKey(db, account.external_user_id, keyHash);
  await setBackupCode(db, account.external_user_id, "");
  return { ok: true, account };
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

export type PaymentKind = "topup" | "admin_grant" | "refund" | "job_debit" | "job_refund";

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

// ---------------------------------------------------------------------------
// Optimistic debit mirror (optimistic balance reduction on ticket send)
//
// NON-AUTHORITATIVE display counter. Each signed Livepeer ticket is mirrored here
// (deduped by request_id) so `balance` reads can subtract the outstanding total
// from PymtHouse's authoritative balance immediately. Rows are reconciled away as
// PymtHouse's metering consumes the allowance (see balance.ts) or pruned by TTL.
// Money itself is NEVER moved here — only PymtHouse debits.
// ---------------------------------------------------------------------------

export interface OptimisticDebitRow {
  id: number;
  external_user_id: string;
  request_id: string;
  /** Ticket expected value in USD micros (ticketEV) — what PymtHouse charges, fixed at ticket-send time. */
  expected_value_usd_micros: number;
  created_at: string;
}

/** True if a (user, requestId) debit is already recorded (dedupe). */
export async function optimisticDebitExists(db: D1Database, externalUserId: string, requestId: string): Promise<boolean> {
  const row = await db
    .prepare("SELECT id FROM optimistic_debits WHERE external_user_id = ?1 AND request_id = ?2")
    .bind(externalUserId, requestId)
    .first<{ id: number }>();
  return !!row;
}

/** Record one optimistic debit (idempotent by (user, requestId)). Returns true only if newly inserted. */
export async function recordOptimisticDebit(
  db: D1Database,
  externalUserId: string,
  requestId: string,
  expectedValueUsdMicros: number,
): Promise<boolean> {
  const inserted = await db
    .prepare(
      `INSERT OR IGNORE INTO optimistic_debits (external_user_id, request_id, expected_value_usd_micros)
       VALUES (?1, ?2, ?3)`,
    )
    .bind(externalUserId, requestId, Math.max(0, Math.floor(expectedValueUsdMicros)))
    .run();
  return (inserted.meta.changes ?? 0) > 0;
}

/** All outstanding optimistic debits for a user, oldest first. */
export async function listOptimisticDebits(db: D1Database, externalUserId: string): Promise<OptimisticDebitRow[]> {
  const res = await db
    .prepare("SELECT * FROM optimistic_debits WHERE external_user_id = ?1 ORDER BY id ASC")
    .bind(externalUserId)
    .all<OptimisticDebitRow>();
  return res.results as OptimisticDebitRow[];
}

export async function deleteOptimisticDebit(db: D1Database, id: number): Promise<void> {
  await db.prepare("DELETE FROM optimistic_debits WHERE id = ?1").bind(id).run();
}

/** Delete mirrored debits older than cutoffIso "YYYY-MM-DD HH:MM:SS" (UTC). */
export async function pruneOptimisticDebits(db: D1Database, externalUserId: string, cutoffIso: string): Promise<void> {
  await db.prepare("DELETE FROM optimistic_debits WHERE external_user_id = ?1 AND created_at < ?2").bind(externalUserId, cutoffIso).run();
}

/**
 * Prune stale rows then SUM the remaining outstanding face value for a user.
 * Returns a bigint of total optimistic debits not yet absorbed by PymtHouse.
 */
export async function sumOptimisticDebits(db: D1Database, externalUserId: string, ttlMs = 24 * 60 * 60 * 1000): Promise<bigint> {
  const cutoff = new Date(Date.now() - ttlMs).toISOString().replace("T", " ").slice(0, 19);
  await pruneOptimisticDebits(db, externalUserId, cutoff);
  const rows = await listOptimisticDebits(db, externalUserId);
  let total = 0n;
  for (const r of rows) total += BigInt(r.expected_value_usd_micros || 0);
  return total;
}

export interface BalanceSyncRow {
  external_user_id: string;
  last_consumed_usd_micros: number;
  updated_at: string;
}

export async function getBalanceSync(db: D1Database, externalUserId: string): Promise<BalanceSyncRow | null> {
  const row = await db
    .prepare("SELECT * FROM balance_sync WHERE external_user_id = ?1")
    .bind(externalUserId)
    .first<BalanceSyncRow>();
  return row ?? null;
}

/** Seed/update the per-user consumed baseline (source of the absorption delta). */
export async function setBalanceSync(db: D1Database, externalUserId: string, lastConsumedUsdMicros: bigint | number): Promise<void> {
  await db
    .prepare(
      `INSERT INTO balance_sync (external_user_id, last_consumed_usd_micros, updated_at)
       VALUES (?1, ?2, ?3)
       ON CONFLICT(external_user_id) DO UPDATE SET
         last_consumed_usd_micros = excluded.last_consumed_usd_micros,
         updated_at = excluded.updated_at`,
    )
    .bind(externalUserId, Number(lastConsumedUsdMicros), nowIso())
    .run();
}

