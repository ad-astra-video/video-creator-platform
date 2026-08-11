/**
 * Shared helpers for the dispatch/catalog/provider routes: zod body parsing,
 * orchestrator construction (fetch seam), and the `dispatchJob` pipeline —
 * credit check + decrement (ledger) → D1 job row → runner dispatch → {jobId},
 * with refund-on-failure.
 */

import { err, ok, readJson } from "../utils";
import { getExternalUserByKeyHash } from "../ledger";
import { logPayment } from "../ledger";
import { PymtHouseClient } from "../pymthouse";
import { OrchestratorClient, type RunnerInfo } from "../orchestrator";
import { createJob, updateJob, getSettings } from "../jobs";
import { sha256Hex, cryptoRandomHex } from "../utils";
import type { Env } from "../types";

/** Default per-job cost (USD micros) if JOB_COST_USD_MICROS is unset. */
export const DEFAULT_JOB_COST_USD_MICROS = "100000"; // $0.10

/** Default orchestrator base URL when ORCHESTRATOR_BASE_URL is unset (local dev). */
export const DEFAULT_ORCHESTRATOR_URL = "http://127.0.0.1:8000";

/** Build an orchestrator client from env; the fetch seam defaults to real fetch. */
export function makeOrchestrator(env: Env): OrchestratorClient {
  const baseUrl = env.ORCHESTRATOR_BASE_URL || DEFAULT_ORCHESTRATOR_URL;
  return new OrchestratorClient({ baseUrl });
}

export function jobCostMicros(env: Env): bigint {
  const raw = env.JOB_COST_USD_MICROS || DEFAULT_JOB_COST_USD_MICROS;
  const n = BigInt(raw);
  return n > 0n ? n : BigInt(DEFAULT_JOB_COST_USD_MICROS);
}

/**
 * Task -> runner API endpoint. The webapp posts a job directly to the runner at
 * `runner.url + endpoint` (there is NO job API / intermediary between runners).
 */
const TASK_ENDPOINTS: Record<string, string> = {
  generate: "/video-creator/v1/t2v",
  "generate-image": "/video-creator/v1/image",
  "enhance-prompt": "/video-creator/v1/prompt-enhance",
  extend: "/video-creator/v1/extend",
  retake: "/video-creator/v1/retake",
  restyle: "/video-creator/v1/restyle",
  "restyle:extract-first-frame": "/video-creator/v1/extract-first-frame",
  "restyle:segment-subject": "/video-creator/v1/sam3",
  "restyle:style-frame": "/video-creator/v1/style",
  "ic-lora": "/video-creator/v1/ic-lora-generate",
  "ic-lora:extract-conditioning": "/video-creator/v1/extract-conditioning",
  edit: "/video-creator/v1/edit",
};

export function endpointForTask(type: string): string {
  return TASK_ENDPOINTS[type] || `/video-creator/v1/${type}`;
}

/**
 * True when a runner advertises a non-zero price. PymtHouse should be called
 * ONLY when this is true — a free runner is dispatched with no ledger touch.
 */
export function runnerChargesPrice(runner: RunnerInfo): boolean {
  if (typeof runner.priceUsdMicrosPerSec === "number" && runner.priceUsdMicrosPerSec > 0) return true;
  const ppu = runner.priceInfo?.pricePerUnit;
  if (typeof ppu === "number" && ppu > 0) return true;
  return false;
}

/** Parse + zod-validate a JSON body. Returns { ok:false } response on failure. */
export async function parseBody<T>(request: Request, schema: { safeParse: (v: unknown) => { success: true; data: T } | { success: false; error: unknown } }): Promise<{ ok: true; data: T } | { ok: false; response: Response }> {
  let raw: unknown;
  try {
    raw = await readJson<unknown>(request);
  } catch {
    return { ok: false, response: err("invalid JSON body", 400) };
  }
  const parsed = schema.safeParse(raw);
  if (!parsed.success) {
    const issues = (parsed.error as { issues?: { path: (string | number)[]; message: string }[] }).issues ?? [];
    const detail = issues.map((i) => `${i.path.join(".") || "(root)"}: ${i.message}`).join("; ");
    return { ok: false, response: err(`validation failed: ${detail || "invalid body"}`, 400) };
  }
  return { ok: true, data: parsed.data };
}

/** Resolve the user id from a per-user bearer key, or a 401 response. */
export async function resolveUserFromRequest(request: Request, env: Env): Promise<{ ok: true; userId: string } | { ok: false; response: Response }> {
  const auth = request.headers.get("authorization") || "";
  const secret = auth.startsWith("Bearer ") ? auth.slice(7) : "";
  if (!secret || !env.DB) return { ok: false, response: err("Unauthorized", 401) };
  const hash = await sha256Hex(secret);
  const userId = await getExternalUserByKeyHash(env.DB, hash);
  if (!userId) return { ok: false, response: err("Unauthorized", 401) };
  return { ok: true, userId };
}

export interface DispatchResult {
  ok: boolean;
  response: Response;
  jobId?: string;
}

/**
 * Full dispatch pipeline used by every worker-dispatch route.
 *
 *   1. credit-check the ledger (PymtHouse balance)
 *   2. decrement (consume) + record a `job_debit` audit row        [BEFORE dispatch]
 *   3. create a `jobs` D1 row (queued)
 *   4. discover + select a capable runner, submit the job
 *   5. mark running + return { jobId }
 * On any dispatch failure: refund the debit (`job_refund`), mark the job failed.
 */
export async function dispatchJob(
  env: Env,
  externalUserId: string,
  type: string,
  requestBody: unknown,
  requiredCaps: string[] = [],
): Promise<DispatchResult> {
  if (!env.DB) return { ok: false, response: err("Server error: DB unavailable", 500) };

  // Resolve the orchestrator base from THIS user's configured Discovery URL
  // (fall back to env, then the local default), then discover + select a
  // capable runner FIRST — we need its advertised price to decide on the ledger.
  const settings = (await getSettings(env.DB, externalUserId).catch(() => null)) as
    { livepeerDiscoveryUrl?: unknown } | null;
  const toTrim = settings?.livepeerDiscoveryUrl ?? "";
  const discoveryBase = (typeof toTrim === "string" ? toTrim.trim() : "") || env.ORCHESTRATOR_BASE_URL || DEFAULT_ORCHESTRATOR_URL;
  const orch = new OrchestratorClient({ baseUrl: discoveryBase });

  let runner: RunnerInfo;
  try {
    const runners = await orch.discoverRunners(requiredCaps);
    const picked = orch.selectRunner(runners, requiredCaps);
    if (!picked) throw new Error("no ready runner available for the requested capabilities");
    runner = picked;
  } catch (e) {
    return { ok: false, response: err(`dispatch failed: ${(e as Error).message}`, 502) };
  }

  // ONLY charge through PymtHouse if this runner advertises a non-zero price.
  // A free runner dispatches with no ledger call at all.
  const client = new PymtHouseClient(env);
  const charges = runnerChargesPrice(runner);
  const cost = charges ? jobCostMicros(env) : 0n;
  if (charges) {
    try {
      const balance = await client.getBalance(externalUserId);
      const remaining = BigInt(balance?.remainingUsdMicros ?? "0");
      if (remaining < cost) return { ok: false, response: err("insufficient credits — please top up", 402) };
      await client.consumeCredits(externalUserId, cost.toString());
    } catch (e) {
      return { ok: false, response: err(`debit failed: ${(e as Error).message}`, 502) };
    }
    await logPayment(env.DB, { kind: "job_debit", externalUserId, amountUsdMicros: cost.toString(), reason: type });
  }

  // Persist the job (our local id is canonical for progress in D1).
  const jobId = cryptoRandomHex(12);
  await createJob(env.DB, { id: jobId, user_id: externalUserId, type, request_json: requestBody });

  // POST the job DIRECTLY to the runner (no orchestrator job intermediary).
  try {
    const endpoint = endpointForTask(type);
    const res = await orch.postToRunner(runner, endpoint, { jobId, ...(requestBody as Record<string, unknown>) });
    if (res.status < 200 || res.status >= 300) {
      throw new Error(`runner ${res.status}: ${res.data !== null ? JSON.stringify(res.data) : ""}`);
    }
    await updateJob(env.DB, jobId, { status: "running", runner: runner.id });
    return { ok: true, jobId, response: ok({ jobId }) };
  } catch (e) {
    // Failure: refund (only if we charged) + mark failed.
    if (charges) {
      try { await client.refundCredits(externalUserId, cost.toString()); } catch { /* best-effort */ }
      await logPayment(env.DB, { kind: "job_refund", externalUserId, amountUsdMicros: cost.toString(), reason: `${type} dispatch failure` });
    }
    await updateJob(env.DB, jobId, { status: "failed" });
    return { ok: false, response: err(`dispatch failed: ${(e as Error).message}`, 502) };
  }
}

/** Cancel helper: mark an owned, non-terminal job cancelled (best-effort). */
export function runnerUrl(runner: RunnerInfo): string {
  return runner.url;
}
