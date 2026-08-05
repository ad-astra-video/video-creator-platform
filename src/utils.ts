import type { Env } from "./types";

export function json(data: unknown, status = 200): Response {
  return new Response(JSON.stringify(data), {
    status,
    headers: { "content-type": "application/json" },
  });
}

export function ok(data: unknown): Response {
  return json(data, 200);
}

export function err(message: string, status: number): Response {
  return json({ ok: false, error: message }, status);
}

/**
 * Guard for admin routes (`/admin/*`). Operator must send
 * `Authorization: Bearer <ADMIN_API_KEY>`; never shipped to the desktop.
 */
export function requireAdminKey(req: Request, env: Env): Response | null {
  const auth = req.headers.get("authorization") || "";
  const token = auth.startsWith("Bearer ") ? auth.slice(7) : "";
  if (!env.ADMIN_API_KEY || token !== env.ADMIN_API_KEY) {
    return err("Unauthorized", 401);
  }
  return null;
}

export async function readJson<T>(req: Request): Promise<T> {
  return (await req.json()) as T;
}

/** SHA-256 hash with a random salt; returns "salt:hex". */
export async function hashSecret(value: string, salt?: string): Promise<string> {
  const s = salt ?? cryptoRandomHex(16);
  const data = new TextEncoder().encode(`${s}:${value}`);
  const digest = await crypto.subtle.digest("SHA-256", data);
  return `${s}:${toHex(new Uint8Array(digest))}`;
}

export async function verifyHash(stored: string, value: string): Promise<boolean> {
  const [salt, _hex] = stored.split(":");
  if (!salt) return false;
  const recomputed = await hashSecret(value, salt);
  return constantTimeEq(recomputed, stored);
}

function constantTimeEq(a: string, b: string): boolean {
  if (a.length !== b.length) return false;
  let diff = 0;
  for (let i = 0; i < a.length; i++) diff |= a.charCodeAt(i) ^ b.charCodeAt(i);
  return diff === 0;
}

/**
 * Unsalted SHA-256 hex of a high-entropy secret (API keys). Storage is plain-SHA
 * only because the key is 256-bit random (no rainbow-table concern) and it lets
 * us look a token up directly by hash — the salted `hashSecret` would break lookup.
 */
export async function sha256Hex(value: string): Promise<string> {
  const digest = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(value));
  return toHex(new Uint8Array(digest));
}

export function toHex(bytes: Uint8Array): string {
  let hex = "";
  for (const b of bytes) hex += b.toString(16).padStart(2, "0");
  return hex;
}

export function cryptoRandomHex(bytes: number): string {
  const buf = new Uint8Array(bytes);
  crypto.getRandomValues(buf);
  return toHex(buf);
}

/** Human-friendly 8-char recovery code (no ambiguous chars). */
export function generateRecoveryCode(): string {
  const alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789";
  let code = "";
  const buf = new Uint8Array(8);
  crypto.getRandomValues(buf);
  for (const b of buf) code += alphabet[b % alphabet.length];
  return code;
}

/** Datetime string in the same format D1 uses (UTC). */
export function nowIso(): string {
  return new Date().toISOString().replace("T", " ").slice(0, 19);
}

export function addMinutesIso(minutes: number): string {
  const d = new Date(Date.now() + minutes * 60_000);
  return d.toISOString().replace("T", " ").slice(0, 19);
}

export function isExpired(iso: string): boolean {
  const s = new Date(iso.replace(" ", "T") + "Z").getTime();
  return Date.now() > s;
}

/** True when the given email looks like a valid address (basic check). */
export function isValidEmail(email: string): boolean {
  return /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email.trim());
}
