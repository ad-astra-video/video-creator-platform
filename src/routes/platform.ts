/**
 * Platform-credits surface for the web CreditsPanel (/api/platform/* contract).
 *
 * These are the shapes the frontend CreditsPanel already expects (the desktop app
 * used to get them from a Python backend that proxied the Worker). In the direct
 * web build there is no desktop backend, so the Worker serves the same contract
 * directly:
 *
 *   GET  /api/platform/status         -> { userId, configured, hasApiKey, baseUrl }
 *   GET  /api/platform/balance        -> Balance + { configured }
 *   POST /api/platform/checkout       -> { url, configured }        (body { tier: cents })
 *   POST /api/platform/link-email     -> { configured }             (body { email })
 *   POST /api/platform/recover/request  -> { status }               (public, body { email })
 *   POST /api/platform/recover/confirm  -> { hasApiKey, configured }(public, body { email, code })
 *
 * All of them stay graceful when PymtHouse / Stripe / email are not configured in the
 * local environment: they report `configured: false` (or a valid empty result) instead
 * of 502/404, so the panel renders a clean "not configured" state.
 */

import { err, ok } from "../utils";
import { PymtHouseClient } from "../pymthouse";
import { listSpendEntries, sumSpendByProject } from "../ledger";
import { resolveUserFromRequest } from "./lib";
import type { Env } from "../types";

export async function getPlatformStatus(request: Request, env: Env): Promise<Response> {
  const u = await resolveUserFromRequest(request, env);
  if (!u.ok) return u.response;

  let balance = null;
  try {
    balance = await new PymtHouseClient(env).getBalance(u.userId);
  } catch {
    // PymtHouse not configured / unreachable — report "not configured" instead of 502.
  }

  return ok({
    ok: true,
    userId: u.userId,
    configured: balance !== null,
    hasApiKey: true,
    baseUrl: new URL(request.url).origin,
  });
}

/**
 * GET /api/platform/history — a user's spending history.
 *
 * Merges two sources:
 *   - `history` / `perProject`: the Worker's durable local spend ledger
 *     (every /sign-ticket records one row with the exact ticketEV PymtHouse
 *     charges + optional project_id). This is instant, per-transaction, and
 *     supports the per-project breakdown — PymtHouse itself has NO project
 *     dimension (it meters by end-user only), so per-project spend can only
 *     come from here.
 *   - `invoices`: PymtHouse's authoritative per-user invoices (settled/billed
 *     charges). Best-effort: if PymtHouse is unconfigured/unreachable we still
 *     return the local ledger.
 *
 * Never renders hashes — it returns USD micros + timestamps + project ids only.
 */
export async function getPlatformHistory(request: Request, env: Env, externalUserId: string): Promise<Response> {
  if (!env.DB) return ok({ configured: true, history: [], perProject: [], invoices: [], ok: true });

  const [history, perProject] = await Promise.all([
    listSpendEntries(env.DB, externalUserId, 200),
    sumSpendByProject(env.DB, externalUserId).catch(() => []),
  ]);

  let invoices: unknown[] = [];
  const client = new PymtHouseClient(env);
  try {
    invoices = await client.getInvoices(externalUserId);
  } catch {
    // PymtHouse unconfigured / unreachable — local ledger is still valid history.
  }

  return ok({
    ok: true,
    configured: true,
    history: history.map((h) => ({
      requestId: h.request_id,
      projectId: h.project_id,
      amountUsdMicros: String(h.expected_value_usd_micros),
      createdAt: h.created_at,
    })),
    perProject: perProject.map((p) => ({
      projectId: p.project_id,
      totalUsdMicros: String(p.total_usd_micros),
      count: p.count,
    })),
    invoices,
  });
}
