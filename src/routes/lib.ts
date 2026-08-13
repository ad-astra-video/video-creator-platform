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
 * Repurposed under the direct-transport design (plans/20260811_direct_transport.md,
 * PHASE_B_BROWSER_TICKET_FLOW.md): the D1 Worker is NOT in the media/inference path. The
 * browser posts generations DIRECTLY to the runner (Livepeer) or to FAL; this Worker's only
 * involvement is minting the PymtHouse payment ticket (`/sign-ticket`) + auth/balance.
 *
 * These shared worker-dispatch entry points (`/api/generate`, `/api/generate-image`, ...) are
 * therefore GONE as media carriers. They authenticate (so a stale/degraded client still gets a
 * clean, distinguishable response) and then return 410 Gone directing the caller to the direct
 * transport, instead of silently doing the old debit-and-proxy that would bypass the Phase B
 * ticket rail. No D1 job row, no PymtHouse decrement, no runner proxy happens here.
 */
export async function dispatchJob(
  _env: Env,
  _externalUserId: string,
  type: string,
  _requestBody: unknown,
  _requiredCaps: string[] = [],
): Promise<DispatchResult> {
  return {
    ok: false,
    response: err(
      `Generation endpoint /api/${type} is served by the DIRECT transport (browser -> runner / FAL). ` +
        `This worker no longer carries the media path; use the frontend's direct transport (lib/direct-transport.ts).`,
      410,
    ),
  };
}

/** Cancel helper: mark an owned, non-terminal job cancelled (best-effort). */
export function runnerUrl(runner: RunnerInfo): string {
  return runner.url;
}
