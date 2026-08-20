-- ---------------------------------------------------------------------------
-- Durable spend ledger + per-project attribution (2026-08).
--
-- One row per signed Livepeer payment ticket (deduped by request_id), recording
-- the EXACT USD the user will be charged (the ticket's expected value / ticketEV,
-- = usage.computed_fee_usd_micros, fixed at ticket-send time) plus an OPTIONAL
-- project_id attribution for per-project spend breakdowns.
--
-- This is the DURABLE spend HISTORY the webapp surfaces ("spending history" +
-- per-project totals). It is DISTINCT from the transient `optimistic_debits`
-- mirror (migrations/0003), which only exists to make the displayed balance drop
-- the instant a ticket is signed and is reconciled/pruned away. Unlike that
-- mirror, spend_entries is never pruned: it is the authoritative local record of
-- what this app signed and charged the user, grouped by project.
--
-- NOTE: PymtHouse itself has NO project dimension — its metering subject is the
-- end-user (externalUserId). Per-project spend therefore cannot come from
-- PymtHouse; we capture it here at the only point the app knows both the ticket
-- charge AND which project triggered it (the /sign-ticket rail). project_id is
-- null when the client didn't send one (older builds / non-project tasks) and
-- still counts toward the global total.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS spend_entries (
  id                    INTEGER PRIMARY KEY AUTOINCREMENT,
  external_user_id      TEXT NOT NULL,
  project_id            TEXT,                          -- null => unattributed (counts toward global only)
  request_id            TEXT NOT NULL,                 -- PymtHouse usage.request_id (dedupe key)
  expected_value_usd_micros INTEGER NOT NULL,          -- ticketEV: exact USD micros PymtHouse charges, fixed at ticket-send time
  created_at            TEXT DEFAULT (datetime('now'))
);

-- Idempotent: a retried /sign-ticket must not double-count a spend entry.
CREATE UNIQUE INDEX IF NOT EXISTS idx_spend_entries_req ON spend_entries (external_user_id, request_id);
CREATE INDEX IF NOT EXISTS idx_spend_entries_user ON spend_entries (external_user_id, created_at);
CREATE INDEX IF NOT EXISTS idx_spend_entries_project ON spend_entries (external_user_id, project_id);
