# LTX Credits Platform (Serverless Backend)

The **serverless** part of the LTX-Desktop deposit-to-inference stack. A single Cloudflare
**Worker + D1** that owns all secrets and wires four services together:

```
LTX-Desktop (Electron + Python) ──> this Worker ──> Stripe   (collects top-up + platform fee)
                                              └──> PymtHouse (allowance ledger + remote signer DMZ -> orchestrators)
```

It does **not** sit in the discovery or inference hot path — it is only hit on deposit/checkout,
email recovery, job dispatch (minting a signer session), and operator admin. See
`../ONBOARDING_AND_EXECUTION_PLAN.md` for the full architecture and economics.

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
| `GET /admin/payments` | **admin** key | monitor payments received (credit-audit ledger) |
| `GET /admin/users` | **admin** key | list registered accounts |
| `GET /admin/balance?externalUserId=` | **admin** key | live PymtHouse allowance (reconciliation) |
| `POST /admin/grant` | **admin** key | manually credit a user's PymtHouse allowance (send funds) |

Two distinct keys:
- `PLATFORM_API_KEY` — shipped in the desktop app; can create checkouts, read balances, do recovery.
- `ADMIN_API_KEY` — operator-only; **never** ships to the desktop; unlocks `/admin/*`.

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

## Prerequisites — gather these first

You need the **Stripe and PymtHouse accounts before any secret can be set**, so set those up
early (steps 3–4) and have them handy:

1. Node.js ≥ 20 (dev on 24), pnpm, and a **Cloudflare account**.
2. A **Stripe** account (test mode to start) — gives you a secret key + webhook secret.
3. A **PymtHouse** app (Builder) with its client pair — gives you the base URL + M2M credentials.
4. A **Resend** account (or swap `src/recovery.ts` for Mailgun/SES).

---

## Directory layout

```
platform/
  wrangler.toml          # Worker + D1 configuration (edit database_id)
  package.json           # scripts: dev / deploy / d1 migrate / typecheck
  migrations/0001_init.sql   # accounts, recovery_codes, idempotency
  migrations/0002_payments.sql # credit-audit ledger (top-ups + admin grants)
  src/
    index.ts             # router + handlers (incl. /admin/*)
    config.ts            # STRIPE_TIERS + tier helpers
    stripe.ts            # createCheckoutSession + webhook HMAC verify
    pymthouse.ts         # PymtHouse Builder API client (M2M Basic auth)
    ledger.ts            # D1: accounts, codes, idempotency, payments log
    recovery.ts          # email delivery of one-time codes
    utils.ts             # hashing, codes, validation
    types.ts             # Env + shared types
```

---

## Stand-up (step by step)

### 1. Install
```bash
cd platform
pnpm install                 # first run: approve build scripts if prompted
```

### 2. Create the D1 database
```bash
pnpm d1:list                 # or: wrangler d1 create ltx-credits
```
Copy the returned `database_id` into `wrangler.toml` under `[[d1_databases]]`.

### 3. Set up PymtHouse (get your client pair)
- Register a **Builder** app. You get a public `app_…` client (no secret) and a confidential
  `m2m_…` client (+ `pmth_cs_…` secret). Keep the M2M secret **only** in the Worker.
- Scopes: public client `sign:job` (+ `users:token` for per-user billing); M2M `users:read`,
  `users:write`, `users:token`.
- Set the **Starter plan allowance to $0** so no free credit leaks:
  ```
  PUT /api/v1/apps/{clientId}/starter-plan  {"includedUsdMicros":"0"}
  ```
- Note `PYMTHOUSE_BASE_URL` (e.g. `https://<app>.pymthouse.example`).

### 4. Set up Stripe (get keys + webhook secret)
- Grab a **secret key** (`sk_test_…` / `sk_live_…`) from Developers → API keys.
- **Create the webhook before setting secrets** — add endpoint
  `https://<your-worker>.workers.dev/webhook/stripe`, select the **`checkout.session.completed`**
  event, and copy the signing secret (`whsec_…`) — you'll need it in step 5.
- The Stripe dashboard's "Send test webhook" is also how you'll exercise the grant locally.

### 5. Set secrets
Now that you have real values from steps 3–4, store them:
```bash
pnpm wrangler secret put STRIPE_SECRET_KEY          # sk_test_...
pnpm wrangler secret put STRIPE_WEBHOOK_SECRET      # whsec_...  (from step 4)
pnpm wrangler secret put STRIPE_TIERS               # JSON of tiers (optional; defaults below)
pnpm wrangler secret put PYMTHOUSE_BASE_URL         # https://<app>.pymthouse.example
pnpm wrangler secret put PYMTHOUSE_PUBLIC_CLIENT_ID # app_...
pnpm wrangler secret put PYMTHOUSE_M2M_CLIENT_ID    # m2m_...
pnpm wrangler secret put PYMTHOUSE_M2M_CLIENT_SECRET# pmth_cs_...
pnpm wrangler secret put PLATFORM_API_KEY           # shared key the desktop uses
pnpm wrangler secret put ADMIN_API_KEY              # operator key for /admin/* (never ship)
pnpm wrangler secret put RESEND_API_KEY             # re_...
pnpm wrangler secret put EMAIL_FROM                 # credits@yourdomain.com
# (Phase B, optional) pnpm wrangler secret put WEBHOOK_SECRET
```
Default tiers (used if `STRIPE_TIERS` is unset):
```json
[{"creditsCents":1000,"feeCents":100},{"creditsCents":2500,"feeCents":150},{"creditsCents":5000,"feeCents":300},{"creditsCents":10000,"feeCents":500}]
```

### 6. Apply migrations & run
```bash
pnpm d1:migrate:local        # local dev DB
pnpm dev                     # http://localhost:8787

# when ready to deploy:
pnpm d1:migrate              # remote DB
pnpm deploy                  # -> https://<name>.<account>.workers.dev
```

---

## Monitoring payments & sending funds (admin)

`/admin/*` requires `Authorization: Bearer <ADMIN_API_KEY>` (never the desktop key).

```bash
export AKEY="<ADMIN_API_KEY>"
export BASE="https://<your-worker>.workers.dev"

# See every credit that landed (top-ups + manual grants), newest first
curl -s -H "Authorization: Bearer $AKEY" "$BASE/admin/payments?limit=50"
#   -> { "payments": [ { "kind":"topup", "external_user_id":"...",
#                       "tier_credits_cents":1000, "amount_usd_micros":"10000000",
#                       "stripe_event_id":"evt_...", "created_at":"..." } ] }

# List accounts
curl -s -H "Authorization: Bearer $AKEY" "$BASE/admin/users"

# Live allowance for one user (reconciliation)
curl -s -H "Authorization: Bearer $AKEY" "$BASE/admin/balance?externalUserId=<uuid>"

# Manually credit a user (bank transfer, compensation, refund, or re-send a missed webhook):
curl -s -X POST -H "Authorization: Bearer $AKEY" -H "content-type: application/json" \
  -d '{"externalUserId":"<uuid>","amountUsdMicros":"5000000","reason":"manual - bank transfer"}' \
  "$BASE/admin/grant"
# Every grant is audited in /admin/payments.
```

The `payments` table is the single source of truth for money received on-platform. The Stripe
webhook writes `kind='topup'` rows automatically; `POST /admin/grant` writes `kind='admin_grant'`.
Platform-fee revenue is simply `sum(platform fee)` derived from `tier_credits_cents`, or from
Stripe's own dashboard — this table tracks **credits granted** (what flows to PymtHouse).

> Note: this credits the **PymtHouse allowance**. Funding the crypto **signing wallet** that
> actually pays orchestrators (buying LPT/ETH from Stripe payouts) is a separate operator step,
> out of scope of this Worker.

---

## Local verification

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
   allowance (the fee stays with you as profit) and logs it in `payments`.
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

- **Keep the M2M secret, Stripe secret, and ADMIN_API_KEY out of the desktop app** — they live
  only here.
- **Starter = $0** ensures the platform fee is the only margin and credits only come from paid
  top-ups or explicit admin grants.
- **Tiers are the only options** (10/25/50/100). To change, update `STRIPE_TIERS` or the
  `DEFAULT_TIERS` in `src/config.ts`.
- Windows/pnpm quirk: set `pnpm config set verify-deps-before-run false` if `pnpm run` re-checks
  deps and errors on ignored build scripts (already reflected in `.npmrc`).
