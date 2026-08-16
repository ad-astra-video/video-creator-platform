-- ---------------------------------------------------------------------------
-- Optimistic balance debit mirror + reconciliation baseline (2026-08).
--
-- PymtHouse debits the user's allowance ASYNCHRONOUSLY via its metering
-- pipeline after the orchestrator redeems a signed ticket, so the displayed
-- balance read from /users/{id}/allowances lags ticket-send time. To make the
-- UI balance drop the instant a ticket is signed, we record a local mirror of
-- each signed ticket's USD face value ("optimistic debit") and subtract the
-- OUTSTANDING (not-yet-absorbed) total from the authoritative PymtHouse balance
-- when displaying it.
--
-- This is a DISPLAY / reconciliation counter, NOT an authoritative balance:
-- money STILL moves only through PymtHouse. The Worker never writes a balance
-- it owns; it only remembers what it signed so the display can reflect spend
-- early, and it reconciles (drops mirrored rows) as PymtHouse's metering
-- catches up (consumedUsdMicros advances) or via a TTL safety net for
-- never-redeemed tickets.
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS optimistic_debits (
  id                    INTEGER PRIMARY KEY AUTOINCREMENT,
  external_user_id      TEXT NOT NULL,
  request_id            TEXT NOT NULL,               -- PymtHouse usage.request_id (dedupe key)
  expected_value_usd_micros INTEGER NOT NULL,        -- ticketEV: expected value PymtHouse charges (usage.computed_fee_usd_micros), fixed at ticket-send time
  created_at            TEXT DEFAULT (datetime('now'))
);

-- One optimistic debit per (user, requestId) — a retried /sign-ticket must not double-count.
CREATE UNIQUE INDEX IF NOT EXISTS idx_optimistic_debits_req ON optimistic_debits (external_user_id, request_id);
CREATE INDEX IF NOT EXISTS idx_optimistic_debits_user ON optimistic_debits (external_user_id);

-- Per-user balance-sync baseline: the last observed PymtHouse consumedUsdMicros.
-- Used to absorb (delete) optimistic_debits once PymtHouse metering actually
-- consumes the allowance, so the mirror never double-counts.
CREATE TABLE IF NOT EXISTS balance_sync (
  external_user_id         TEXT PRIMARY KEY,
  last_consumed_usd_micros INTEGER NOT NULL DEFAULT 0,
  updated_at               TEXT DEFAULT (datetime('now'))
);
