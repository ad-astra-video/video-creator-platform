/**
 * D1 persistence for the dispatch layer: `jobs` (per-generation task records)
 * and `projects` (metadata only — media blobs stay client-side). Also exposes
 * per-user settings, provider choice and HF auth-token storage. See
 * migrations/0002_jobs.sql for the schema.
 */

export const JOB_STATUS = {
  QUEUED: "queued",
  RUNNING: "running",
  COMPLETED: "completed",
  FAILED: "failed",
  CANCELLED: "cancelled",
} as const;
export type JobStatus = (typeof JOB_STATUS)[keyof typeof JOB_STATUS];

export interface JobRow {
  id: string;
  user_id: string;
  type: string;
  status: JobStatus;
  runner: string | null;
  request_json: string | null;
  created_at: string;
  updated_at: string;
}

export interface JobUpdate {
  status?: JobStatus;
  runner?: string | null;
  request_json?: string | null;
}

export interface CreateJobInput {
  id: string;
  user_id: string;
  type: string;
  status?: JobStatus;
  runner?: string | null;
  request_json?: unknown;
  created_at?: string;
}

export interface ProjectRow {
  id: string;
  user_id: string;
  name: string;
  asset_manifest_json: string | null;
  created_at: string;
  updated_at: string;
}

export interface CreateProjectInput {
  id: string;
  user_id: string;
  name: string;
  asset_manifest?: unknown;
}

export interface HfTokenRow {
  user_id: string;
  token_enc: string; // the raw token (base64) — scoped to this user's D1 row
  user_name: string | null;
  created_at: string;
  updated_at: string;
}

function nowIso(): string {
  return new Date().toISOString().replace("T", " ").slice(0, 19);
}

// --- UTF-8 safe base64 (Web-API only; no Buffer, works in Workers + node) ---
const B64_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

export function utf8ToB64(input: string): string {
  const bytes = new TextEncoder().encode(input);
  let out = "";
  for (let i = 0; i < bytes.length; i += 3) {
    const b0 = bytes[i];
    const b1 = i + 1 < bytes.length ? bytes[i + 1] : 0;
    const b2 = i + 2 < bytes.length ? bytes[i + 2] : 0;
    out += B64_ALPHABET[b0 >> 2];
    out += B64_ALPHABET[((b0 & 3) << 4) | (b1 >> 4)];
    out += i + 1 < bytes.length ? B64_ALPHABET[((b1 & 15) << 2) | (b2 >> 6)] : "=";
    out += i + 2 < bytes.length ? B64_ALPHABET[b2 & 63] : "=";
  }
  return out;
}

export function b64ToUtf8(input: string): string {
  const clean = input.replace(/=+$/, "");
  const bytes: number[] = [];
  for (let i = 0; i < clean.length; i += 4) {
    const c0 = B64_ALPHABET.indexOf(clean[i]);
    const c1 = B64_ALPHABET.indexOf(clean[i + 1]);
    const c2 = clean[i + 2] ? B64_ALPHABET.indexOf(clean[i + 2]) : 0;
    const c3 = clean[i + 3] ? B64_ALPHABET.indexOf(clean[i + 3]) : 0;
    bytes.push((c0 << 2) | (c1 >> 4));
    if (clean[i + 2]) bytes.push(((c1 & 15) << 4) | (c2 >> 2));
    if (clean[i + 3]) bytes.push(((c2 & 3) << 6) | c3);
  }
  return new TextDecoder().decode(new Uint8Array(bytes));
}

// ---------------------------------------------------------------------------
// Jobs
// ---------------------------------------------------------------------------

/** Insert a job row. */
export async function createJob(db: D1Database, input: CreateJobInput): Promise<void> {
  const ts = input.created_at ?? nowIso();
  await db
    .prepare(
      `INSERT INTO jobs (id, user_id, type, status, runner, request_json, created_at, updated_at)
       VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?7)`,
    )
    .bind(
      input.id,
      input.user_id,
      input.type,
      input.status ?? JOB_STATUS.QUEUED,
      input.runner ?? null,
      input.request_json === undefined ? null : JSON.stringify(input.request_json),
      ts,
    )
    .run();
}

/** Update a job row's mutable fields; bumps updated_at. */
export async function updateJob(db: D1Database, jobId: string, update: JobUpdate): Promise<void> {
  const sets: string[] = [];
  const binds: unknown[] = [];
  if (update.status !== undefined) {
    sets.push("status = ?" + (binds.length + 1));
    binds.push(update.status);
  }
  if (update.runner !== undefined) {
    sets.push("runner = ?" + (binds.length + 1));
    binds.push(update.runner);
  }
  if (update.request_json !== undefined) {
    sets.push("request_json = ?" + (binds.length + 1));
    binds.push(JSON.stringify(update.request_json));
  }
  sets.push("updated_at = ?" + (binds.length + 1));
  binds.push(nowIso());
  binds.push(jobId);
  await db.prepare(`UPDATE jobs SET ${sets.join(", ")} WHERE id = ?${binds.length}`).bind(...binds).run();
}

/** Fetch a single job row (or null). */
export async function getJob(db: D1Database, jobId: string): Promise<JobRow | null> {
  const row = await db.prepare("SELECT * FROM jobs WHERE id = ?1").bind(jobId).first<JobRow>();
  return row ?? null;
}

/** A job only if it belongs to this user (ownership check). */
export async function getOwnedJob(db: D1Database, jobId: string, userId: string): Promise<JobRow | null> {
  const row = await db.prepare("SELECT * FROM jobs WHERE id = ?1 AND user_id = ?2").bind(jobId, userId).first<JobRow>();
  return row ?? null;
}

/** Recent jobs for a user, newest first. */
export async function listJobs(db: D1Database, userId: string, limit = 50): Promise<JobRow[]> {
  const res = await db
    .prepare("SELECT * FROM jobs WHERE user_id = ?1 ORDER BY created_at DESC LIMIT ?2")
    .bind(userId, limit)
    .all<JobRow>();
  return res.results as JobRow[];
}

// ---------------------------------------------------------------------------
// Projects (metadata only)
// ---------------------------------------------------------------------------

/** Upsert a project (metadata only). */
export async function upsertProject(db: D1Database, input: CreateProjectInput): Promise<void> {
  await db
    .prepare(
      `INSERT INTO projects (id, user_id, name, asset_manifest_json, created_at, updated_at)
       VALUES (?1, ?2, ?3, ?4, ?5, ?5)
       ON CONFLICT(id) DO UPDATE SET
         name = excluded.name,
         asset_manifest_json = excluded.asset_manifest_json,
         updated_at = ?5`,
    )
    .bind(
      input.id,
      input.user_id,
      input.name,
      input.asset_manifest === undefined ? null : JSON.stringify(input.asset_manifest),
      nowIso(),
    )
    .run();
}

/** A project only if it belongs to this user. */
export async function getOwnedProject(db: D1Database, projectId: string, userId: string): Promise<ProjectRow | null> {
  const row = await db.prepare("SELECT * FROM projects WHERE id = ?1 AND user_id = ?2").bind(projectId, userId).first<ProjectRow>();
  return row ?? null;
}

// ---------------------------------------------------------------------------
// Per-user settings (metadata, stored as JSON in D1)
// ---------------------------------------------------------------------------

export async function getSettings(db: D1Database, userId: string): Promise<Record<string, unknown>> {
  const row = await db.prepare("SELECT settings_json FROM settings WHERE user_id = ?1").bind(userId).first<{ settings_json: string | null }>();
  if (!row?.settings_json) return {};
  try {
    return JSON.parse(row.settings_json) as Record<string, unknown>;
  } catch {
    return {};
  }
}

export async function setSettings(db: D1Database, userId: string, settings: Record<string, unknown>): Promise<void> {
  await db
    .prepare(
      `INSERT INTO settings (user_id, settings_json, updated_at)
       VALUES (?1, ?2, ?3)
       ON CONFLICT(user_id) DO UPDATE SET settings_json = ?2, updated_at = ?3`,
    )
    .bind(userId, JSON.stringify(settings), nowIso())
    .run();
}

// ---------------------------------------------------------------------------
// Per-user provider choice (backed by orchestrator discovery)
// ---------------------------------------------------------------------------

export async function getProvider(db: D1Database, userId: string): Promise<Record<string, unknown> | null> {
  const row = await db.prepare("SELECT provider_json FROM providers WHERE user_id = ?1").bind(userId).first<{ provider_json: string | null }>();
  if (!row?.provider_json) return null;
  try {
    return JSON.parse(row.provider_json) as Record<string, unknown>;
  } catch {
    return null;
  }
}

export async function setProvider(db: D1Database, userId: string, provider: Record<string, unknown>): Promise<void> {
  await db
    .prepare(
      `INSERT INTO providers (user_id, provider_json, updated_at)
       VALUES (?1, ?2, ?3)
       ON CONFLICT(user_id) DO UPDATE SET provider_json = ?2, updated_at = ?3`,
    )
    .bind(userId, JSON.stringify(provider), nowIso())
    .run();
}

export async function deleteProvider(db: D1Database, userId: string): Promise<void> {
  await db.prepare("DELETE FROM providers WHERE user_id = ?1").bind(userId).run();
}

// ---------------------------------------------------------------------------
// Hugging Face auth tokens (per user)
// ---------------------------------------------------------------------------

/** Store a user's HF token. Base64 is not encryption; the token is scoped to the
 *  user's D1 row and never returned over the API (status reports a boolean). */
export async function storeHfToken(db: D1Database, userId: string, token: string, userName?: string): Promise<void> {
  await db
    .prepare(
      `INSERT INTO hf_tokens (user_id, token_enc, user_name, created_at, updated_at)
       VALUES (?1, ?2, ?3, ?4, ?4)
       ON CONFLICT(user_id) DO UPDATE SET token_enc = ?2, user_name = COALESCE(?3, user_name), updated_at = ?4`,
    )
    .bind(userId, utf8ToB64(token), userName ?? null, nowIso())
    .run();
}

export async function getHfTokenRow(db: D1Database, userId: string): Promise<HfTokenRow | null> {
  const row = await db.prepare("SELECT * FROM hf_tokens WHERE user_id = ?1").bind(userId).first<HfTokenRow>();
  return row ?? null;
}

export async function clearHfToken(db: D1Database, userId: string): Promise<void> {
  await db.prepare("DELETE FROM hf_tokens WHERE user_id = ?1").bind(userId).run();
}
