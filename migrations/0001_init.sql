-- Full D1 schema for the LTX credits worker (single init migration;
-- the database has not been created yet, so there is no need to split).

-- Per-instance accounts (credit owners) with recovery email.
CREATE TABLE IF NOT EXISTS accounts (
  id                INTEGER PRIMARY KEY AUTOINCREMENT,
  external_user_id  TEXT UNIQUE NOT NULL,   -- install UUID (credit owner)
  email             TEXT UNIQUE,            -- nullable until attached & verified
  pending_email     TEXT,                   -- email awaiting /link-email/verify
  email_verified    INTEGER DEFAULT 0,
  created_at        TEXT DEFAULT (datetime('now')),
  last_seen_at      TEXT
);

-- One-time recovery / link codes (never store plaintext).
CREATE TABLE IF NOT EXISTS recovery_codes (
  id          INTEGER PRIMARY KEY AUTOINCREMENT,
  email       TEXT NOT NULL,
  code_hash   TEXT NOT NULL,                -- "salt:sha256hex"
  purpose     TEXT NOT NULL,                -- 'link' | 'recover'
  expires_at  TEXT NOT NULL,
  used        INTEGER DEFAULT 0,
  created_at  TEXT DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_recovery_codes_email ON recovery_codes (email);

-- Stripe webhook idempotency (never double-credit).
CREATE TABLE IF NOT EXISTS idempotency (
  key         TEXT PRIMARY KEY,             -- Stripe event.id
  applied_at  TEXT DEFAULT (datetime('now'))
);

-- Audit ledger of every credit pushed to PymtHouse (top-ups + admin grants).
CREATE TABLE IF NOT EXISTS payments (
  id                   INTEGER PRIMARY KEY AUTOINCREMENT,
  stripe_event_id      TEXT,                -- null for manual/admin grants
  stripe_session_id    TEXT,                -- null for manual/admin grants
  external_user_id     TEXT NOT NULL,
  tier_credits_cents   INTEGER,             -- null for manual/admin grants
  amount_usd_micros    INTEGER NOT NULL,    -- credits granted (never the platform fee)
  kind                 TEXT NOT NULL DEFAULT 'topup',  -- 'topup' | 'admin_grant' | 'refund'
  reason               TEXT,
  created_at           TEXT DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_payments_user ON payments (external_user_id);
CREATE INDEX IF NOT EXISTS idx_payments_event ON payments (stripe_event_id);

-- Per-user API keys (replaces the single shared platform key).
-- Provisioned once (/provision); rotated after email proof (/recover/confirm);
-- revocable by an admin. Only the SHA-256 of the key is stored.
CREATE TABLE IF NOT EXISTS api_keys (
  external_user_id TEXT PRIMARY KEY,
  key_hash         TEXT NOT NULL,
  created_at       TEXT DEFAULT (datetime('now')),
  last_used_at     TEXT,
  revoked          INTEGER DEFAULT 0
);

-- Phase B: signed-payment authorization scopes (one per generation job) for audit/reconciliation.
CREATE TABLE IF NOT EXISTS authz_sessions (
  id                       TEXT PRIMARY KEY,                    -- authz_…
  external_user_id         TEXT NOT NULL,
  job_id                   TEXT NOT NULL,
  orchestrator_id          TEXT NOT NULL,
  max_face_value_usd_micros INTEGER NOT NULL,
  expires_at               TEXT NOT NULL,
  status                   TEXT NOT NULL DEFAULT 'active',      -- active|consumed|expired|refunded
  created_at               TEXT DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_authz_sessions_user ON authz_sessions (external_user_id);

-- Phase B: signed payment tickets (debit/reversal reconciled by PymtHouse metering; audit here).
CREATE TABLE IF NOT EXISTS tickets (
  ticket_hash            TEXT PRIMARY KEY,
  session_id             TEXT NOT NULL,
  face_value_usd_micros  INTEGER NOT NULL,
  signed_at              TEXT DEFAULT (datetime('now')),
  status                 TEXT NOT NULL DEFAULT 'signed',        -- signed|redeemed|refunded
  redeemed_at            TEXT
);
CREATE INDEX IF NOT EXISTS idx_tickets_session ON tickets (session_id);
