# LTX Credits Platform (Serverless Backend)

The **serverless** part of the LTX-Desktop deposit-to-inference stack. A single Cloudflare
**Worker + D1** that owns all secrets and wires four services together:

```
LTX-Desktop (Electron + Python) ──> this Worker ──> Stripe   (collects top-up + platform fee)
                                              └──> PymtHouse (allowance ledger + remote signer DMZ -> orchestrators)
```

It does **not** sit in the discovery or inference hot path — it is only hit on deposit/checkout,
email recovery, and job dispatch (minting a signer session). See `../ONBOARDING_AND_EXECUTION_PLAN.md`
for the full architecture and economics.

---

## What it does

| Route | Auth | Purpose |
|---|---|---|
| `GET /health` | public | liveness |
| `POST /checkout { externalUserId, tier }` | API key | create a Stripe hosted Checkout Session (credits + platform fee) |
| `GET /tiers` | API key | the 4 top-up tiers (for the desktop picker) |
| `GET /balance?externalUserId=` | API key | current PymtHouse allowance (USD micros) |
| `POST /link-email` / `POST /link-email/verify` | API key | attach & verify a recovery email |
| `POST /recover/request` / `POST /recover/confirm` | API key | recover a lost install's UUID via emailed one-time code |
| `GET /usage?externalUserId=` | API key | balance-backed usage view |
| `POST /webhook/stripe` | signature | grants `credits` only (never the fee) — idempotent |
| `POST /authorize` | — | (Phase B) go-livepeer DMZ identity webhook — stubbed |

The 4 tiers (only these are offered):

| Credits | Platform fee | User pays | Stripe fee | Net |
|---|---|---|---|---|
| $10 | $1.00 | $11.00 | ~$0.62 | $0.38 |
| $25 | $1.50 | $26.50 | ~$1.07 | $0.43 |
| $50 | $3.00 | $53.00 | ~$1.84 | $1.16 |
| $100 | $5.00 | $105.00 | ~$3.35 | $1.65 |

Inference is **pass-through** (no markup); all margin is the platform fee, and the webhook
grants exactly the **credit** amount (never the fee).

---

## Prerequisites

- Node.js ≥ 20 (dev on 24), pnpm, and a **Cloudflare account**
- A **Stripe** account (test mode to start)
- A **PymtHouse** app (Builder) with its client pair + scopes
- A **Resend** account (or swap `src/recovery.ts` for Mailgun/SES)

---

## Directory layout

```
platform/
  wrangler.toml          # Worker + D1 configuration (edit database_id)
  package.json           # scripts: dev / deploy / d1 migrate / typecheck
  migrations/0001_init.sql
  src/
    index.ts             # router + handlers
    config.ts            # STRIPE_TIERS + tier helpers
    stripe.ts            # createCheckoutSession + webhook HMAC verify
    pymthouse.ts         # PymtHouse Builder API client (M2M Basic auth)
    ledger.ts            # D1: accounts, recovery_codes, idempotency
    recovery.ts          # email delivery of one-time codes
    utils.ts             # hashing, codes, validation
    types.ts             # Env + shared types
```

---

## Stand-up (step by step)

### 1. Install
```bash
cd platform
pnpm install                 # first run: `pnpm approve-builds` if it prompts about build scripts
```

### 2. Create the D1 database
```bash
pnpm d1:list                 # or: wrangler d1 create ltx-credits
```
Copy the returned `database_id` into `wrangler.toml` under `[[d1_databases]]`.

### 3. Set secrets
```bash
pnpm wrangler secret put STRIPE_SECRET_KEY          # sk_test_...
pnpm wrangler secret put STRIPE_WEBHOOK_SECRET      # whsec_...  (create in Stripe first, step 6)
pnpm wrangler secret put STRIPE_TIERS               # JSON of tiers
pnpm wrangler secret put PYMTHOUSE_BASE_URL         # https://<app>.pymthouse.example
pnpm wrangler secret put PYMTHOUSE_PUBLIC_CLIENT_ID # app_...
pnpm wrangler secret put PYMTHOUSE_M2M_CLIENT_ID    # m2m_...
pnpm wrangler secret put PYMTHOUSE_M2M_CLIENT_SECRET# pmth_cs_...
pnpm wrangler secret put PLATFORM_API_KEY           # shared key the desktop uses
pnpm wrangler secret put RESEND_API_KEY             # re_...
pnpm wrangler secret put EMAIL_FROM                 # credits@yourdomain.com
# (Phase B, optional) pnpm wrangler secret put WEBHOOK_SECRET
```
Default tiers (used if `STRIPE_TIERS` is unset):
```json
[{"creditsCents":1000,"feeCents":100},{"creditsCents":2500,"feeCents":150},{"creditsCents":5000,"feeCents":300},{"creditsCents":10000,"feeCents":500}]
```

### 4. PymtHouse app setup (once)
- Register a **Builder** app. You get a public `app_…` client (no secret) and a confidential
  `m2m_…` client (+ `pmth_cs_…` secret). Keep the M2M secret **only** in the Worker.
- Scopes: public client `sign:job` (+ `users:token` for per-user billing); M2M `users:read`,
  `users:write`, `users:token`.
- Set the **Starter plan allowance to $0** so no free credit leaks:
  ```
  PUT /api/v1/apps/{clientId}/starter-plan  {"includedUsdMicros":"0"}
  ```
  (dashboard session auth).

### 5. Stripe webhook
In the Stripe dashboard → Developers → Webhooks, add an endpoint:
`https://<your-worker>.workers.dev/webhook/stripe`, select the **`checkout.session.completed`**
event, and copy the signing secret into `STRIPE_WEBHOOK_SECRET`.

### 6. Apply migrations & run
```bash
pnpm d1:migrate:local        # local dev DB
pnpm dev                     # http://localhost:8787

# when ready to deploy:
pnpm d1:migrate              # remote DB
pnpm deploy                  # -> https://<name>.<account>.workers.dev
```

---

## Local verification (no Stripe/PymtHouse needed for the happy-ish path)

```bash
export KEY="<PLATFORM_API_KEY>"
export BASE="http://localhost:8787"

curl -s "$BASE/health"
curl -s -H "Authorization: Bearer $KEY" "$BASE/tiers"
curl -s -H "Authorization: Bearer $KEY" "$BASE/balance?externalUserId=test-1"
```

`POST /webhook/stripe` can be exercised with a **signed** test event by exporting it from the
Stripe dashboard → Webhooks → "Send test webhook" against your dev URL, then confirming the
credit lands in PymtHouse (`/usage` / `/balance`). The handler grants only `credit_usd_micros`
and is idempotent by Stripe `event.id`.

---

## Money flow (recap)

1. Desktop picks a tier → `POST /checkout` → Stripe page charges **credits + fee**.
2. `checkout.session.completed` webhook → Worker grants the **credit** amount to the PymtHouse
   allowance (the fee stays with you as profit).
3. On job dispatch, the Worker mints a `sign:job` signer session; PymtHouse's DMZ signs a
   Livepeer ticket **from your funding wallet** to pay the orchestrator, and the allowance is
   decremented at network cost (pass-through).
4. Stripe payouts (fiat) periodically refill the funding wallet (via exchange/on-ramp) — this
   is an operator step, out of scope of this Worker.

---

## Phase B (not yet implemented)

`POST /authorize` (the go-livepeer identity webhook) and the direct-DMZ signer proxy are stubbed
in `src/index.ts` and `src/pymthouse.ts#getSignerRouting`. Wire them with
`@pymthouse/builder-sdk/signer/server` and `/signer/webhook` when orchestrator payment is live.

---

## Notes / decisions

- **Keep the M2M secret and Stripe secret out of the desktop app** — they live only here.
- **Starter = $0** ensures the platform fee is the only margin and credits only come from paid
  top-ups.
- **Tiers are the only options** (10/25/50/100). To change, update `STRIPE_TIERS` or the
  `DEFAULT_TIERS` in `src/config.ts`.
- Windows/pnpm quirk: set `pnpm config set verify-deps-before-run false` if `pnpm run` re-checks
  deps and errors on ignored build scripts (already reflected in `.npmrc`).
