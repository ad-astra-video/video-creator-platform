/**
 * Shared helpers for routes.
 *
 * This module used to host the worker-dispatch machinery (zod body parsing, orchestrator
 * construction, the `dispatchJob` pipeline, task->runner endpoint map). That surface was
 * removed: under the direct-transport design the Worker is control-plane ONLY and does not
 * carry the media/inference path (the browser posts generations directly to the runner/FAL,
 * and the Worker's only involvement is minting the PymtHouse payment ticket via /sign-ticket
 * plus auth/balance/settings). Only `resolveUserFromRequest` is still shared.
 */

import { err, sha256Hex } from "../utils";
import { getExternalUserByKeyHash } from "../ledger";
import type { Env } from "../types";

/** Resolve the user id from a per-user bearer key, or a 401 response. */
export async function resolveUserFromRequest(
  request: Request,
  env: Env,
): Promise<{ ok: true; userId: string } | { ok: false; response: Response }> {
  const auth = request.headers.get("authorization") || "";
  const secret = auth.startsWith("Bearer ") ? auth.slice(7) : "";
  if (!secret || !env.DB) return { ok: false, response: err("Unauthorized", 401) };
  const hash = await sha256Hex(secret);
  const userId = await getExternalUserByKeyHash(env.DB, hash);
  if (!userId) return { ok: false, response: err("Unauthorized", 401) };
  return { ok: true, userId };
}
