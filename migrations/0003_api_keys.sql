-- Per-user API keys replace the single shared PLATFORM_API_KEY.
-- One key per external_user_id. Provisioned once (/provision); rotated after email
-- proof (/recover/confirm); revocable by an admin. Only the SHA-256 of the key is stored.
CREATE TABLE IF NOT EXISTS api_keys (
  external_user_id TEXT PRIMARY KEY,
  key_hash         TEXT NOT NULL,
  created_at       TEXT DEFAULT (datetime('now')),
  last_used_at     TEXT,
  revoked          INTEGER DEFAULT 0
);

-- Track the email waiting to be confirmed by /link-email/verify.
ALTER TABLE accounts ADD COLUMN pending_email TEXT;
