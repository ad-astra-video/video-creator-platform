import type { AccountRow, Env, RecoveryCodeRow } from "./types";
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

// Re-exported for index.ts convenience
export { hashSecret, generateRecoveryCode as _gen } from "./utils";
