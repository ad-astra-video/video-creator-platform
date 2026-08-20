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
   * Real-time per-user entitlement check. Reads the lightweight `/usage/balance`
   * gate endpoint which returns the per-user remaining entitlement
   * (`balanceUsdMicros`, `consumedUsdMicros`, `lifetimeGrantedUsdMicros`,
   * `hasAccess`). In merchant billing mode this endpoint is scoped per
   * `externalUserId` (verified live); the legacy `/users/{id}/allowances` read
   * returns an incompatible shape and must not be used.
   */
  async getBalance(externalUserId: string): Promise<Balance> {
    const raw = (await this.request(
      "GET",
      `${this.base()}/usage/balance?externalUserId=${encodeURIComponent(externalUserId)}`,
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

  /**
   * Create a per-user prepaid top-up via PymtHouse's hosted Stripe Checkout
   * (merchant/Connect rail). In merchant billing mode the `externalUserId` is
   * required in the body and the checkout is scoped to that end-user; funds land
   * in the user's wallet when Stripe fires `checkout.session.completed`
   * (PymtHouse handles the Connect webhook — the worker does not). Returns the
   * Stripe-hosted checkout URL the end-user completes.
   */
  async createWalletTopUp(
    externalUserId: string,
    amountUsd: number,
    urls?: { successUrl?: string; cancelUrl?: string },
  ): Promise<{ checkoutUrl: string }> {
    const data = (await this.request("POST", `${this.base()}/billing/wallet/top-up`, {
      externalUserId,
      amountUsd,
      successUrl: urls?.successUrl,
      cancelUrl: urls?.cancelUrl,
    })) as { checkoutUrl?: string };
    return { checkoutUrl: String(data?.checkoutUrl || "") };
  }

  /** PymtHouse-issued invoices / billed charges for a user (authoritative settled spend). */
  async getInvoices(externalUserId: string): Promise<unknown[]> {
    const data = (await this.request(
      "GET",
      `${this.base()}/users/${encodeURIComponent(externalUserId)}/invoices`,
    )) as { items?: unknown[] };
    return Array.isArray(data?.items) ? data.items : [];
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
