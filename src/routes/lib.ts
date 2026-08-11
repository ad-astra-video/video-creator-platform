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
import { createJob, updateJob } from "../jobs";
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
  const client = new PymtHouseClient(env);
  const cost = jobCostMicros(env);

  // 1 + 2: check then decrement the ledger BEFORE dispatch.
  let balance;
  try {
    balance = await client.getBalance(externalUserId);
  } catch (e) {
    return { ok: false, response: err(`balance check failed: ${(e as Error).message}`, 502) };
  }
  const remaining = BigInt(balance?.remainingUsdMicros ?? "0");
  if (remaining < cost) {
    return { ok: false, response: err("insufficient credits — please top up", 402) };
  }
  try {
    await client.consumeCredits(externalUserId, cost.toString());
  } catch (e) {
    return { ok: false, response: err(`debit failed: ${(e as Error).message}`, 402) };
  }
  await logPayment(env.DB, { kind: "job_debit", externalUserId, amountUsdMicros: cost.toString(), reason: type });

  // 3: persist the job.
  const jobId = cryptoRandomHex(12);
  await createJob(env.DB, { id: jobId, user_id: externalUserId, type, request_json: requestBody });

  // 4: dispatch via the orchestrator.
  const orch = makeOrchestrator(env);
  try {
    const runners = await orch.discoverRunners(requiredCaps);
    const runner = orch.selectRunner(runners, requiredCaps);
    if (!runner) throw new Error("no ready runner available for the requested capabilities");
    const submitted = await orch.submitJob(runner, { type, jobId, request: requestBody });
    await updateJob(env.DB, jobId, { status: "running", runner: runner.id });
    // The local jobId is canonical (shared with the orchestrator as jobId, per the jobs schema).
    return { ok: true, jobId, response: ok({ jobId }) };
  } catch (e) {
    // 5 (failure): refund + mark failed.
    try {
      await client.refundCredits(externalUserId, cost.toString());
    } catch { /* best-effort refund */ }
    await logPayment(env.DB, { kind: "job_refund", externalUserId, amountUsdMicros: cost.toString(), reason: `${type} dispatch failure` });
    await updateJob(env.DB, jobId, { status: "failed" });
    return { ok: false, response: err(`dispatch failed: ${(e as Error).message}`, 502) };
  }
}

/** Cancel helper: mark an owned, non-terminal job cancelled (best-effort). */
export function runnerUrl(runner: RunnerInfo): string {
  return runner.url;
}
