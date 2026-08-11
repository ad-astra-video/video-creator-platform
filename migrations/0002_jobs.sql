-- Phase 2 (architecture.md): generation job records, project metadata,
-- per-user settings, provider choice, and HF auth tokens. Additive to 0001_init.sql.

-- Jobs: one row per dispatched generation task (t2v/restyle/image/extend/...).
CREATE TABLE IF NOT EXISTS jobs (
  id            TEXT PRIMARY KEY,              -- uuid (client-agnostic; shared with orchestrator as jobId)
  user_id       TEXT NOT NULL,
  type          TEXT NOT NULL,                 -- generate | generate-image | enhance-prompt | extend | retake | restyle | ic-lora | ...
  status        TEXT NOT NULL DEFAULT 'queued',-- queued | running | completed | failed | cancelled
  runner        TEXT,                          -- chosen runner id (from orchestrator discovery)
  request_json  TEXT,                          -- forwarded dispatch payload (metadata only; no blobs)
  created_at    TEXT DEFAULT (datetime('now')),
  updated_at    TEXT DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_jobs_user ON jobs (user_id, created_at DESC);

-- Projects: metadata only (name + asset-path manifest). Media blobs live client-side.
CREATE TABLE IF NOT EXISTS projects (
  id                    TEXT PRIMARY KEY,
  user_id               TEXT NOT NULL,
  name                  TEXT NOT NULL,
  asset_manifest_json   TEXT,                  -- JSON: relative asset paths + types
  created_at            TEXT DEFAULT (datetime('now')),
  updated_at            TEXT DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_projects_user ON projects (user_id);

-- Per-user app settings (JSON blob).
CREATE TABLE IF NOT EXISTS settings (
  user_id        TEXT PRIMARY KEY,
  settings_json  TEXT NOT NULL DEFAULT '{}',
  updated_at     TEXT DEFAULT (datetime('now'))
);

-- Per-user chosen provider (runner id + metadata from orchestrator discovery).
CREATE TABLE IF NOT EXISTS providers (
  user_id        TEXT PRIMARY KEY,
  provider_json  TEXT NOT NULL DEFAULT '{}',
  updated_at     TEXT DEFAULT (datetime('now'))
);

-- Hugging Face OAuth tokens, scoped per user. token_enc is base64 (base64 is
-- NOT encryption — upgrade to keyed AES once a key is provisioned). The raw
-- token is never returned over the API; /status reports a boolean only.
CREATE TABLE IF NOT EXISTS hf_tokens (
  user_id      TEXT PRIMARY KEY,
  token_enc    TEXT NOT NULL,
  user_name    TEXT,
  created_at   TEXT DEFAULT (datetime('now')),
  updated_at   TEXT DEFAULT (datetime('now'))
);
