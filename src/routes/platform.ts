/**
 * Platform status (keep (existing Worker handler) per api-contract.md). The
 * existing /api/platform/balance|checkout|link-email|recover/* handlers stay in
 * src/index.ts untouched; this adds GET /api/platform/status, which reports the
 * service + credit state (balance via PymtHouse) for the authenticated user.
 */

import { err, ok } from "../utils";
import { PymtHouseClient } from "../pymthouse";
import { resolveUserFromRequest } from "./lib";
import type { Env } from "../types";

export async function getPlatformStatus(request: Request, env: Env): Promise<Response> {
  const u = await resolveUserFromRequest(request, env);
  if (!u.ok) return u.response;
  const client = new PymtHouseClient(env);
  let balance = null;
  try {
    balance = await client.getBalance(u.userId);
  } catch (e) {
    return err(`balance unavailable: ${(e as Error).message}`, 502);
  }
  return ok({
    ok: true,
    service: "video-creator-platform",
    version: "2.0",
    credits: balance,
    orchestratorConfigured: !!env.ORCHESTRATOR_BASE_URL,
  });
}
