/**
 * GET /health (keep-remote per api-contract.md): real liveness. Reports the
 * service identity without requiring auth.
 */

import { ok } from "../utils";

export async function getHealth(): Promise<Response> {
  return ok({ ok: true, service: "video-creator-platform", ts: Date.now() });
}
