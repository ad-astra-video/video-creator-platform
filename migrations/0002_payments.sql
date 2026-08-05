-- Audit ledger of every credit pushed to PymtHouse.
-- Rows come from Stripe top-up webhooks (kind='topup') OR operator admin grants (kind='admin_grant').
CREATE TABLE IF NOT EXISTS payments (
  id                   INTEGER PRIMARY KEY AUTOINCREMENT,
  stripe_event_id      TEXT,              -- null for manual/admin grants
  stripe_session_id    TEXT,              -- null for manual/admin grants
  external_user_id     TEXT NOT NULL,
  tier_credits_cents   INTEGER,           -- null for manual/admin grants
  amount_usd_micros    INTEGER NOT NULL,  -- credits granted (never includes the platform fee)
  kind                 TEXT NOT NULL DEFAULT 'topup',  -- 'topup' | 'admin_grant' | 'refund'
  reason               TEXT,
  created_at           TEXT DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_payments_user ON payments (external_user_id);
CREATE INDEX IF NOT EXISTS idx_payments_event ON payments (stripe_event_id);
