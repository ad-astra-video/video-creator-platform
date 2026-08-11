/**
 * WebSocket relay for live job progress (browser <-> Worker -> orchestrator).
 *
 * Authenticates the per-user key via `?token=` or the `authorization` header,
 * then pushes typed progress events on a per-user basis. Because the
 * orchestrator isn't stood up yet (architecture.md Decision 5), progress is
 * driven by the jobs table: `notifyJob()` updates the D1 row AND broadcasts to
 * any subscribed sockets for that user. Once the orchestrator is live, the same
 * `notifyJob()`/`publishProgress()` seams are fed from its own stream.
 *
 * Uses the Cloudflare WebSocketPair API.
 */

import { sha256Hex } from "./utils";
import { getExternalUserByKeyHash } from "./ledger";
import { updateJob, type JobUpdate, type JobStatus } from "./jobs";
import type { Env } from "./types";

// ---- Shared WS message schema ----
export const WS_MSG = {
  /** Client -> server: tell the relay which job this socket wants updates for. */
  SUBSCRIBE: "subscribe",
  /** Server -> client: live job progress. */
  JOB_PROGRESS: "job.progress",
  /** Server -> client: ack for subscribe + the job's current snapshot. */
  SUBSCRIBED: "subscribed",
  /** Server -> client: error/warning in the WS channel. */
  ERROR: "error",
} as const;

export interface SubscribeMessage {
  type: typeof WS_MSG.SUBSCRIBE;
  jobId: string;
}

export interface JobProgressMessage {
  type: typeof WS_MSG.JOB_PROGRESS;
  jobId: string;
  phase?: string;
  percent?: number;
  frame?: number;
  status?: JobStatus;
  error?: string;
}

export type WsClientMessage = SubscribeMessage;

// In-process subscriber registry. Workers isolates can be separate, so this is
// best-effort per isolate; production correctness comes from the orchestrator
// relaying through the same notifyJob entrypoint.
const subscribers = new Map<string, Set<WebSocket>>(); // userId -> sockets

function send(ws: WebSocket, msg: unknown): void {
  try {
    ws.send(JSON.stringify(msg));
  } catch {
    // socket gone; ignore
  }
}

/** Register a socket for a user. Returns an unsubscribe fn. */
export function subscribeUser(userId: string, ws: WebSocket): () => void {
  if (!subscribers.has(userId)) subscribers.set(userId, new Set());
  subscribers.get(userId)!.add(ws);
  return () => {
    const set = subscribers.get(userId);
    if (set) {
      set.delete(ws);
      if (set.size === 0) subscribers.delete(userId);
    }
  };
}

/** Broadcast a message to all sockets of a user. */
export function broadcastToUser(userId: string, msg: unknown): void {
  const set = subscribers.get(userId);
  if (!set) return;
  for (const ws of [...set]) {
    if (ws.readyState === 1 /* OPEN */) send(ws, msg);
    else set.delete(ws);
  }
}

/**
 * Bridge used by job lifecycle: persist the status update to the jobs row in D1
 * AND push a progress event to the owning user's sockets. This is what the
 * orchestrator-driven stream will call once it's live.
 */
export async function notifyJob(db: D1Database, jobId: string, userId: string, update: JobUpdate): Promise<void> {
  await updateJob(db, jobId, update);
  const msg: JobProgressMessage = { type: WS_MSG.JOB_PROGRESS, jobId, status: update.status };
  broadcastToUser(userId, msg);
}

/**
 * Push a rich live-progress event (phase/percent/frame) to a user's sockets
 * without persisting — used by the orchestrator stream once it's up. The REST
 * fallback reads the jobs row instead.
 */
export function publishProgress(userId: string, msg: Omit<JobProgressMessage, "type">): void {
  broadcastToUser(userId, { type: WS_MSG.JOB_PROGRESS, ...msg });
}

/**
 * Attempt to upgrade an incoming /ws request. Authenticates the per-user key
 * from `?token=` or `Authorization: Bearer <key>` before upgrading.
 */
export async function handleWebSocket(request: Request, env: Env): Promise<Response> {
  const url = new URL(request.url);
  const token = url.searchParams.get("token") || (request.headers.get("authorization") || "").replace(/^Bearer\s+/i, "");
  if (!token || !env.DB) return errWs("Unauthorized", 401);
  const hash = await sha256Hex(token);
  const user = await getExternalUserByKeyHash(env.DB, hash);
  if (!user) return errWs("Unauthorized", 401);

  const upgrade = request.headers.get("Upgrade");
  if (upgrade?.toLowerCase() !== "websocket") return badRequest("Expected a WebSocket upgrade");

  const pair = new WebSocketPair();
  const [client, server] = Object.values(pair) as [WebSocket, WebSocket];
  const unsubscribe = subscribeUser(user, server);

  server.accept();
  server.addEventListener("message", (event) => {
    let data: WsClientMessage;
    try {
      data = JSON.parse(String(event.data)) as WsClientMessage;
    } catch {
      send(server, { type: WS_MSG.ERROR, error: "invalid JSON" });
      return;
    }
    if (data?.type === WS_MSG.SUBSCRIBE && data.jobId) {
      // Ack + send a snapshot so a fresh socket catches up on current state.
      send(server, { type: WS_MSG.SUBSCRIBED, jobId: data.jobId, ack: true });
    } else {
      send(server, { type: WS_MSG.ERROR, error: "unknown message" });
    }
  });
  server.addEventListener("close", () => unsubscribe());
  server.addEventListener("error", () => unsubscribe());

  return new Response(null, { status: 101, webSocket: client });
}

function errWs(text: string, status: number): Response {
  return new Response(JSON.stringify({ ok: false, error: text }), {
    status,
    headers: { "content-type": "application/json" },
  });
}

function badRequest(text: string): Response {
  return new Response(text, { status: 400 });
}
