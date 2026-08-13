import type { Balance, Env } from "./types";

/** Shape of the per-user allowances response body (PymtHouse). */
interface AllowanceView {
  balanceUsdMicros?: string;
  consumedUsdMicros?: string;
  lifetimeGrantedUsdMicros?: string;
  hasAccess?: boolean;
}

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

  /**
   * Real-time entitlement check. Reads the canonical per-user allowances endpoint,
   * which INCLUDES manual top-up grants (the /usage/balance read only reflects the
   * subscription/Starter ledger and omits top-ups). allowances.balanceUsdMicros is
   * the current remaining entitlement ("Balance" in the UI).
   */
  async getBalance(externalUserId: string): Promise<Balance> {
    const raw = (await this.request(
      "GET",
      `${this.base()}/users/${encodeURIComponent(externalUserId)}/allowances`,
    )) as { allowances?: AllowanceView } | AllowanceView;
    const a = (raw && (raw as { allowances?: AllowanceView }).allowances) || (raw as AllowanceView);
    const balUsd = String(a?.balanceUsdMicros ?? "0");
    return {
      hasAccess: Boolean(a?.hasAccess),
      balanceUsdMicros: balUsd,
      remainingUsdMicros: balUsd,
      consumedUsdMicros: String(a?.consumedUsdMicros ?? "0"),
      lifetimeGrantedUsdMicros: String(a?.lifetimeGrantedUsdMicros ?? balUsd),
    };
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

  /** DMZ / identity-webhook URLs (Phase B). Resolves the direct signer DMZ from
   *  PymtHouse so `REMOTE_SIGNER_URL` doesn't have to be configured separately —
   *  the base URL covers it. */
  async getSignerRouting(): Promise<{ dmzUrl: string; webhookUrl: string; jwksUrl: string; meteringMode: string }> {
    const data = (await this.request("GET", `${this.base()}/signer/routing`)) as any;
    const routing = (data && typeof data === "object" ? data.routing : data) || {};
    return {
      dmzUrl: String(routing.remoteDmzUrl || routing.signerApiUrl || ""),
      webhookUrl: String(routing.webhookUrl || ""),
      jwksUrl: String(routing.jwksUri || ""),
      meteringMode: String(routing.meteringMode || ""),
    };
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
