import type { Env } from "./types";
import { addMinutesIso, err } from "./utils";

/**
 * Email the user a one-time code. Uses Resend (`POST /emails`) by default.
 * Swap this module for Mailgun/SES by replacing `sendEmail` if needed.
 */
export async function sendCodeEmail(env: Env, to: string, purpose: "link" | "recover", code: string): Promise<Response> {
  const subject = purpose === "link" ? "Confirm your Video Creator credits recovery email" : "Your Video Creator credits recovery code";
  const html = `
    <div style="font-family:sans-serif;max-width:480px;margin:auto">
      <h2>${purpose === "link" ? "Confirm your email" : "Recover your Video Creator credits"}</h2>
      <p>Your one-time code is:</p>
      <p style="font-size:28px;letter-spacing:4px;font-weight:bold">${code}</p>
      <p>It expires in 15 minutes. If you didn't request this, you can ignore this email.</p>
    </div>`;
  return sendEmail(env, to, subject, html);
}

export async function sendEmail(env: Env, to: string, subject: string, html: string): Promise<Response> {
  if (!env.RESEND_API_KEY) {
    // No email provider configured: in dev this just no-ops with a marker.
    return err("Email provider not configured (RESEND_API_KEY missing)", 500);
  }
  const res = await fetch("https://api.resend.com/emails", {
    method: "POST",
    headers: {
      authorization: `Bearer ${env.RESEND_API_KEY}`,
      "content-type": "application/json",
    },
    body: JSON.stringify({ from: env.EMAIL_FROM, to, subject, html }),
  });
  return res;
}

/** For local dev without a provider: compute expiry so tests can consume codes. */
export { addMinutesIso };
