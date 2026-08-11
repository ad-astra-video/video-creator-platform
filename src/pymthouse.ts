import type { Balance, Env } from "./types";

/**
 * Minimal PymtHouse Builder API client (no external SDK needed).
 * All calls use M2M HTTP Basic auth against the app tenant scope.
 * Base path: {PYMTHOUSE_BASE_URL}/api/v1/apps/{PUBLIC_CLIENT_ID}/...
 */
export class PymtHouseClient {
  private app: string;

  constructor(private env: Env) {
    this.app = env.PYMTHOUSE_PUBLIC_CLIENT_ID;
  }

  private base(): string {
    return `${this.env.PYMTHOUSE_BASE_URL}/api/v1/apps/${this.app}`;
  }

  private authHeaders(): Record<string, string> {
    const raw = `${this.env.PYMTHOUSE_M2M_CLIENT_ID}:${this.env.PYMTHOUSE_M2M_CLIENT_SECRET}`;
    const b64 = btoa(raw);
    return { authorization: `Basic ${b64}`, "content-type": "application/json" };
  }

  /** Upsert a user (idempotent). New users auto-subscribe to the Starter plan. */
  async upsertUser(externalUserId: string, email?: string): Promise<void> {
    await this.request("POST", `${this.base()}/users`, { externalUserId, email, status: "active" });
  }

  /** Grant a manual allowance top-up (semi: USD micros). Additive to any subscription allowance. */
  async grantAllowance(externalUserId: string, amountUsdMicros: string): Promise<void> {
    await this.request("POST", `${this.base()}/users/${encodeURIComponent(externalUserId)}/allowances`, {
      amountUsdMicros,
      source: "manual",
      featureKey: null,
    });
  }

  /** Real-time entitlement check. */
  async getBalance(externalUserId: string): Promise<Balance> {
    const data = await this.request("GET", `${this.base()}/usage/balance?externalUserId=${encodeURIComponent(externalUserId)}`);
    return data as Balance;
  }

  /**
   * Debit/consume credits for a generation job (the "decrement before dispatch"
   * step of the worker-dispatch routes). Calls the measure/consume metering
   * endpoint so the provider records the spend against the user's allowance.
   * Throws if the debit is rejected (bad balance / provider error).
   */
  async consumeCredits(externalUserId: string, amountUsdMicros: string): Promise<void> {
    await this.request("POST", `${this.base()}/users/${encodeURIComponent(externalUserId)}/usage/consume`, {
      amountUsdMicros,
      source: "job",
    });
  }

  /** Reversal/settlement after a job that failed and was refunded. */
  async refundCredits(externalUserId: string, amountUsdMicros: string): Promise<void> {
    await this.request("POST", `${this.base()}/users/${encodeURIComponent(externalUserId)}/usage/refund`, {
      amountUsdMicros,
      source: "job",
    });
  }

  /** DMZ / identity-webhook URLs (Phase B). */
  async getSignerRouting(): Promise<{ dmzUrl: string; webhookUrl: string; jwksUrl: string; meteringMode: string }> {
    const data = await this.request("GET", `${this.base()}/signer/routing`);
    return data as any;
  }

  private async request(method: string, url: string, body?: unknown): Promise<unknown> {
    const res = await fetch(url, {
      method,
      headers: this.authHeaders(),
      body: body === undefined ? undefined : JSON.stringify(body),
    });
    const text = await res.text();
    let data: unknown = null;
    if (text) {
      try {
        data = JSON.parse(text);
      } catch {
        data = text;
      }
    }
    if (!res.ok) {
      throw new Error(`PymtHouse ${method} ${url} -> ${res.status}: ${text}`);
    }
    return data;
  }
}
