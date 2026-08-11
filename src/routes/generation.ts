/**
 * worker-dispatch routes (api-contract.md): validated task submission that runs
 * the shared dispatchJob pipeline (credit check+decrement -> D1 job row ->
 * orchestrator submit -> { jobId }), plus cancel + REST progress fallbacks.
 */

import { z } from "zod";
import { err, ok, readJson } from "../utils";
import { dispatchJob, makeOrchestrator, parseBody, resolveUserFromRequest } from "./lib";
import { getOwnedJob, updateJob, type JobStatus } from "../jobs";
import type { Env } from "../types";

type Z<T> = { safeParse: (v: unknown) => { success: true; data: T } | { success: false; error: unknown } };

const promptSchema = z.object({ prompt: z.string().min(1).max(20000) });
const generateSchema = z.object({
  prompt: z.string().min(1).max(20000),
  numFrames: z.number().int().min(1).max(600).optional(),
  width: z.number().int().min(128).max(2048).optional(),
  height: z.number().int().min(128).max(2048).optional(),
  seed: z.number().int().optional(),
  fps: z.number().min(1).max(60).optional(),
});
const imageSchema = z.object({
  prompt: z.string().min(1).max(20000),
  width: z.number().int().min(128).max(2048).optional(),
  height: z.number().int().min(128).max(2048).optional(),
  numImages: z.number().int().min(1).max(8).optional(),
});
const extendSchema = z.object({
  projectId: z.string().min(1),
  timeline: z.unknown().optional(),
  durationSec: z.number().int().min(1).max(600).optional(),
});
const retakeSchema = z.object({
  projectId: z.string().min(1),
  clipIndex: z.number().int().min(0).optional(),
  prompt: z.string().min(1).max(20000).optional(),
});
const restyleSchema = z.object({
  projectId: z.string().min(1),
  prompt: z.string().min(1).max(20000),
  styleFrameIndex: z.number().int().min(0).optional(),
  subjectMask: z.unknown().optional(),
});
const frameExtractSchema = z.object({
  projectId: z.string().min(1).optional(),
  video_path: z.string().min(1).optional(),
  timestampSec: z.number().min(0).optional(),
});
const segmentSchema = z.object({
  projectId: z.string().min(1).optional(),
  image_path: z.string().min(1),
});
const styleFrameSchema = z.object({
  projectId: z.string().min(1).optional(),
  image_path: z.string().min(1),
});
const cancelSchema = z.object({ jobId: z.string().min(1) });

/** Generic handler that validates with a zod schema then runs dispatchJob. */
function makeDispatch(type: string, schema: Z<any>, caps: string[] = []) {
  return async (request: Request, env: Env): Promise<Response> => {
    const u = await resolveUserFromRequest(request, env);
    if (!u.ok) return u.response;
    const body = await parseBody(request, schema);
    if (!body.ok) return body.response;
    const result = await dispatchJob(env, u.userId, type, body.data, caps);
    return result.response;
  };
}

export async function postGenerate(request: Request, env: Env): Promise<Response> {
  return makeDispatch("generate", generateSchema, ["t2v"])(request, env);
}
export async function postGenerateImage(request: Request, env: Env): Promise<Response> {
  return makeDispatch("generate-image", imageSchema, ["image"])(request, env);
}
export async function postEnhancePrompt(request: Request, env: Env): Promise<Response> {
  return makeDispatch("enhance-prompt", promptSchema, ["prompt"])(request, env);
}
export async function postExtend(request: Request, env: Env): Promise<Response> {
  return makeDispatch("extend", extendSchema, ["extend"])(request, env);
}
export async function postRetake(request: Request, env: Env): Promise<Response> {
  return makeDispatch("retake", retakeSchema, ["t2v"])(request, env);
}
export async function postRestyle(request: Request, env: Env): Promise<Response> {
  return makeDispatch("restyle", restyleSchema, ["restyle"])(request, env);
}
export async function postRestyleExtractFirstFrame(request: Request, env: Env): Promise<Response> {
  return makeDispatch("restyle:extract-first-frame", frameExtractSchema, ["restyle"])(request, env);
}
export async function postRestyleSegmentSubject(request: Request, env: Env): Promise<Response> {
  const u = await resolveUserFromRequest(request, env);
  if (!u.ok) return u.response;
  const body = await parseBody(request, segmentSchema);
  if (!body.ok) return body.response;
  // In the serverless/web path the image exists only as a browser-local web:// blob
  // key, which a remote runner cannot fetch — automatic subject segmentation (SAM3)
  // can't run. Skip it gracefully instead of returning a 400 from the runner.
  if (body.data.image_path.startsWith("web://")) {
    return ok({ ok: true, skipped: true, note: "auto subject segmentation unavailable in browser" });
  }
  const result = await dispatchJob(env, u.userId, "restyle:segment-subject", body.data, ["restyle", "sam3"]);
  return result.response;
}
export async function postRestyleStyleFrame(request: Request, env: Env): Promise<Response> {
  const u = await resolveUserFromRequest(request, env);
  if (!u.ok) return u.response;
  const body = await parseBody(request, styleFrameSchema);
  if (!body.ok) return body.response;
  // A browser-local web:// blob can't be fetched by a remote runner yet (asset
  // handoff pending) — signal it gracefully instead of a 400/502 from the runner.
  if (body.data.image_path.startsWith("web://")) {
    return ok({ ok: true, skipped: true, note: "style-frame unavailable in browser without asset upload" });
  }
  const result = await dispatchJob(env, u.userId, "restyle:style-frame", body.data, ["restyle"]);
  return result.response;
}
export async function postIcLoraGenerate(request: Request, env: Env): Promise<Response> {
  return makeDispatch("ic-lora", imageSchema, ["ic-lora"])(request, env);
}
export async function postIcLoraExtractConditioning(request: Request, env: Env): Promise<Response> {
  return makeDispatch("ic-lora:extract-conditioning", imageSchema, ["ic-lora"])(request, env);
}

/** POST /api/generate/cancel — mark an owned in-flight job cancelled (best-effort). */
export async function postCancel(request: Request, env: Env): Promise<Response> {
  const u = await resolveUserFromRequest(request, env);
  if (!u.ok) return u.response;
  const body = await parseBody(request, cancelSchema);
  if (!body.ok) return body.response;
  if (!env.DB) return err("Server error", 500);

  const job = await getOwnedJob(env.DB, body.data.jobId, u.userId);
  if (!job) return err("job not found", 404);
  const terminal: JobStatus[] = ["completed", "failed", "cancelled"];
  if (terminal.includes(job.status)) return ok({ ok: true, jobId: job.id, status: job.status, note: "already terminal" });

  // Best-effort orchestrator cancel, then mark cancelled locally.
  try {
    const orch = makeOrchestrator(env);
    if (job.runner) await orch.cancelJob(job.id).catch(() => {});
  } catch { /* local cancellation still applies */ }
  await updateJob(env.DB, job.id, { status: "cancelled" });
  return ok({ ok: true, jobId: job.id, status: "cancelled" });
}

/**
 * GET /api/generation/progress — REST fallback (the WS channel supersedes it).
 * Reads the jobs row for the current status.
 */
export async function getGenerationProgress(request: Request, env: Env): Promise<Response> {
  const u = await resolveUserFromRequest(request, env);
  if (!u.ok) return u.response;
  if (!env.DB) return err("Server error", 500);
  const jobId = new URL(request.url).searchParams.get("jobId") || "";
  if (!jobId) {
    // The frontend calls once without a jobId to capture the baseline generation
    // id before starting a new generation (see GenSpace.writeRecoveryContext).
    return ok({ id: null, status: "idle", phase: "idle", runner: null, updatedAt: null });
  }
  const job = await getOwnedJob(env.DB, jobId, u.userId);
  if (!job) return err("job not found", 404);
  return ok({ id: job.id, jobId: job.id, status: job.status, phase: progressPhase(job.status), runner: job.runner, updatedAt: job.updated_at });
}

/** DERIVED from the jobs row (persisted status drives the WS + REST fallback). */
export async function getDownloadProgress(request: Request, env: Env): Promise<Response> {
  return getGenerationProgress(request, env);
}

function progressPhase(status: string): string {
  switch (status) {
    case "queued": return "queued";
    case "running": return "running";
    case "completed": return "completed";
    case "failed": return "failed";
    case "cancelled": return "cancelled";
    default: return "unknown";
  }
}

// Re-export readJson for any route needing raw bodies (kept for API parity).
export { readJson };
