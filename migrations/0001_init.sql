-- Initial D1 schema for the LTX credits worker.
-- Stores per-instance accounts (email recovery), one-time recovery codes, and Stripe webhook idempotency.

CREATE TABLE IF NOT EXISTS accounts (
  id                INTEGER PRIMARY KEY AUTOINCREMENT,
  external_user_id  TEXT UNIQUE NOT NULL,   -- instance UUID (credit owner)
  email             TEXT UNIQUE,            -- nullable until attached & verified
  email_verified    INTEGER DEFAULT 0,
  created_at        TEXT DEFAULT (datetime('now')),
  last_seen_at      TEXT
);

CREATE TABLE IF NOT EXISTS recovery_codes (
  id          INTEGER PRIMARY KEY AUTOINCREMENT,
  email       TEXT NOT NULL,
  code_hash   TEXT NOT NULL,                -- "salt:sha256hex" -- never store plaintext
  purpose     TEXT NOT NULL,                -- 'link' | 'recover'
  expires_at  TEXT NOT NULL,
  used        INTEGER DEFAULT 0,
  created_at  TEXT DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_recovery_codes_email ON recovery_codes (email);

CREATE TABLE IF NOT EXISTS idempotency (
  key         TEXT PRIMARY KEY,             -- Stripe event.id
  applied_at  TEXT DEFAULT (datetime('now'))
);
