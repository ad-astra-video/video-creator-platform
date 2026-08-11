/**
 * Settings (keep (Worker/D1, per-user)): per-user app settings stored as JSON
 * in D1. GET returns the current settings; POST merges the provided fields.
 */

import { z } from "zod";
import { err, ok } from "../utils";
import { getSettings, setSettings } from "../jobs";
import { parseBody, resolveUserFromRequest } from "./lib";
import type { Env } from "../types";

const settingsSchema = z.object({
  // Arbitrary per-user settings; we validate it's a JSON object.
  settings: z.record(z.string(), z.unknown()),
});

export async function getSettingsRoute(request: Request, env: Env): Promise<Response> {
  const u = await resolveUserFromRequest(request, env);
  if (!u.ok) return u.response;
  if (!env.DB) return err("Server error", 500);
  const stored = await getSettings(env.DB, u.userId);
  return ok({
    settings: {
      ...stored,
      // Serverless web app: inference is ALWAYS remote (Worker -> orchestrator -> runners),
      // so report remote mode so the desktop's local-install / API-key first-run gates are skipped.
      remoteInferenceEnabled: true,
      hasLivepeerDiscoveryUrl: true,
      hasLtxApiKey: false,
      hasFalApiKey: false,
    },
  });
}

export async function postSettingsRoute(request: Request, env: Env): Promise<Response> {
  const u = await resolveUserFromRequest(request, env);
  if (!u.ok) return u.response;
  const body = await parseBody(request, settingsSchema);
  if (!body.ok) return body.response;
  if (!env.DB) return err("Server error", 500);
  await setSettings(env.DB, u.userId, body.data.settings);
  return ok({ ok: true, settings: body.data.settings });
}
