-- Optional no-email recovery: a one-time backup code (like a TOTP backup / seed phrase).
-- The user saves this at sign-up; it is NEVER emailed or sent again. Only its SHA-256 hash
-- is stored. `/provision` returns the plaintext once; `/recover/backup` validates it,
-- rotates the API key, and returns a FRESH backup code (shown once) for future use.
ALTER TABLE accounts ADD COLUMN backup_code_hash TEXT;
