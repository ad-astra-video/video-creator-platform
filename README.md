# Video Creator Platform — Serverless Backend

The **serverless** backend for the **Video Creator** desktop app (`../video-creator`). A single
Cloudflare **Worker + D1** that owns all secrets and wires services together. **Credits** are the
first productized feature, powering the deposit-to-inference flow:

```
Video Creator desktop (Electron + Python) ──> this Worker ──> Stripe   (collects top-up + platform fee)
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
| `GET /tiers` | public | the 4 top-up tiers (for the desktop picker) |
| `POST /provision { externalUserId }` | public | mint a **per-user** key (first run); `409` if already provisioned |
| `POST /checkout { tier }` | **user key** | Stripe Checkout (credits + platform fee); user resolved from key |
| `GET /balance` | **user key** | current PymtHouse allowance (USD micros) |
| `GET /usage` | **user key** | balance-backed usage view |
| `POST /link-email { email }` / `POST /link-email/verify { code }` | **user key** | attach & verify a recovery email |
| `POST /recover/request { email }` / `POST /recover/confirm { email, code }` | public | lost/compromised key via **email**: one-time code; confirm **rotates** to a fresh key |
| `POST /recover/backup { code }` | public | lost key via **one-time backup code** (no email): rotate to a fresh key + mint a NEW backup code |
| `POST /webhook/stripe` | signature | grants `credits` only (never the fee) — idempotent |
| `POST /sign-ticket { pymt: { orchestrator, type, ManifestID, state? }, projectId? }` | **user key** | PymtHouse signer DMZ proxy. The `pymt` field is relayed verbatim to the DMZ; `projectId` and other sibling fields are local tracking read by the Worker and never forwarded. Returns `{ payment, segCreds, state }` (the `Livepeer-Payment` / `Livepeer-Segment` header values) |
| `GET /signer/address` | **user key** | payer/broadcaster address for the `Livepeer-Payer-Address` header |
| `POST /authorize` | **webhook secret** | go-livepeer remote-signer identity webhook (verifies the end-user signer JWT) |
| `GET /admin/payments` | **admin** key | monitor payments received (credit-audit ledger) |
| `GET /admin/users` | **admin** key | list registered accounts |
| `GET /admin/balance?externalUserId=` | **admin** key | live PymtHouse allowance (reconciliation) |
| `POST /admin/grant` | **admin** key | manually credit a user's PymtHouse allowance (send funds) |
| `GET /admin/api-keys` | **admin** key | list issued per-user keys (hash only) |
| `POST /admin/revoke-key` | **admin** key | revoke a user's key (forces re-provision via recovery) |

Two distinct auth models:
- **Per-user API key** — one per install, minted once by `POST /provision` (only the SHA-256 is
  stored). The desktop sends `Authorization: Bearer <key>`; the user is resolved **from the key**
  server-side, so no client-supplied `externalUserId` can be spoofed. Rotate via email recovery;
  revoke per-user from `/admin/revoke-key`. **This replaces the old single shared key.**
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

## Prerequisites

### 1. Accounts — create these FIRST

There are **four** external accounts, each with a distinct job. **Cloudflare** hosts the
Worker + D1 control plane, **Resend** delivers the recovery emails, **Stripe** collects the
credit top-up, and **PymtHouse** is the credits ledger. (Emails cannot be sent directly from
D1 — D1 is just the database; an email provider like Resend always does the delivery.) Create
them **before doing anything else** — you can't configure the worker until you have the
credentials each one provides:

| # | Account | What it's for | What you'll need to create | Credentials you'll need from it |
|---|---|---|---|---|
| 1 | **Cloudflare** — dash.cloudflare.com | hosts the Worker + D1 database (control plane: accounts, keys, recovery codes, settings) | a Cloudflare account (free); the D1 DB + Worker are created here in the Stand-up steps | your account login (for `wrangler login`); the Worker's `database_id` |
| 2 | **Stripe** — dashboard.stripe.com (test mode to start) | collects the top-up + platform fee (user buys credits) | a Stripe account (free); a Checkout webhook endpoint | **secret key** (`sk_test_…`) and **webhook signing secret** (`whsec_…`) |
| 3 | **PymtHouse** — register a Builder app | allowance ledger + remote signer (where credits live & get spent) | a Builder app in your PymtHouse account | **base URL** (`https://<app>.pymthouse.example`) + **M2M client pair** (`app_…`, `m2m_…`, `pmth_cs_…`) |
| 4 | **Resend** — resend.com | sends one-time recovery / email-link codes | a Resend account (free, 3,000 emails/mo — more than enough for recovery codes); verify a sending domain | API key (`re_…`) + the `EMAIL_FROM` address |

### 2. Technical requirements

Install these on your machine before the "Stand-up" section below:

- **Node.js ≥ 20** (developed on 24)
- **pnpm** package manager
- **git** (you're in a git repo already)

That's it — no other tooling. `wrangler` is installed as a project dev-dependency, not globally.

---

## Directory layout

```
platform/
  wrangler.toml          # Worker + D1 configuration (edit database_id)
  package.json           # scripts: dev / deploy / d1 migrate / typecheck
  migrations/0001_init.sql   # full schema (accounts, codes, idempotency, payments, api_keys)
  smoke.mjs                    # dev: in-memory per-user key flow self-test (pnpm smoke)
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

> Assumes the **four accounts** (Cloudflare, Stripe, PymtHouse, Resend) from "Prerequisites →
> Accounts" already exist, and Node/pnpm are installed. Each step names which account it uses.
> **Resend is required** — recovery/link codes are sent by email and cannot be sent from D1 alone.

### 1. Install
```bash
cd platform
pnpm install                 # first run: approve build scripts if prompted
```

### 2. Create the D1 database (uses your **Cloudflare** account)
```bash
pnpm d1:list                 # or: wrangler d1 create video-creator-platform
```
Copy the returned `database_id` into `wrangler.toml` under `[[d1_databases]]`.

### 3. Configure PymtHouse (uses the account from Prerequisites — you already have the client pair)
- Register a **Builder** app. You get a public `app_…` client (no secret) and a confidential
  `m2m_…` client (+ `pmth_cs_…` secret). Keep the M2M secret **only** in the Worker.
- Scopes: public client `sign:job` (+ `users:token` for per-user billing); M2M `users:read`,
  `users:write`, `users:token`.
- Set the **Starter plan allowance to $0** so no free credit leaks:
  ```
  PUT /api/v1/apps/{clientId}/starter-plan  {"includedUsdMicros":"0"}
  ```
- Note `PYMTHOUSE_BASE_URL` (e.g. `https://<app>.pymthouse.example`).

### 4. Configure Stripe (uses the account from Prerequisites)
- Grab a **secret key** (`sk_test_…` / `sk_live_…`) from Developers → API keys.
- **Create the webhook before setting secrets** — add endpoint
  `https://<your-worker>.workers.dev/webhook/stripe`, select the **`checkout.session.completed`**
  event, and copy the signing secret (`whsec_…`) — you'll need it in step 5.

**Charging the platform fee — nothing else is needed.** The fee is *not* a separate Stripe
mechanism. `/checkout` charges **credits + fee as one amount** (e.g. $11.00 for the $10 tier)
through a plain `payment` mode Checkout Session with an inline `price_data` line item. Since you
are **the merchant** (not a platform routing money to connected Stripe accounts), there is **no
Stripe Connect / application-fee setup** — the fee is just part of the single charge, and the
webhook grants only the credit portion ($10), keeping the $1 fee as your profit.

Optional — none are required for the fee or tiers to work:
- **Payment methods** — cards are enabled by default; add Apple/Google Pay etc. if you want.
- **Stripe Tax** — only if you must collect sales tax on digital services (needs an
  `automatic_tax` param in `src/stripe.ts`); otherwise leave it off.
- **Live payout** — add a bank/payout method in the dashboard so settled balances reach you
  (test mode doesn't need this).
- **Products/Prices** — we use one-off inline `price_data`, so you don't have to pre-create
  products; create them only if you want cleaner reporting names.

### 5. Configure Resend *(optional — no-email recovery is built in)*

> **You don't actually need Resend.** Sign-up issues a **one-time backup recovery code**
> (shown once, never emailed, stored hashed) that can be used on the Login screen to
> recover an account with **no email at all**. Resend is only needed if you also want
> **email**-based recovery. Configure it only if you do:

- Create a **free** Resend account and an **API key** (`re_…`).
- **Verify a sending domain** (add the MX/SPF/DKIM records it provides) — or for a quick test use
  the shared sender `onboarding@resend.dev` with no DNS setup.
- You'll set `RESEND_API_KEY` and `EMAIL_FROM` in the secrets step below.
- With Resend configured, email recovery works: `/recover/request` emails a code, `/recover/confirm`
  verifies + rotates. See `src/recovery.ts`. (No-email backup-code recovery needs no provider.)

### 6. Set secrets
Now that you have real values from steps 3–5, store them:
```bash
pnpm wrangler secret put STRIPE_SECRET_KEY          # sk_test_...
pnpm wrangler secret put STRIPE_WEBHOOK_SECRET      # whsec_...  (from step 4)
pnpm wrangler secret put STRIPE_TIERS               # JSON of tiers (optional; defaults below)
pnpm wrangler secret put PYMTHOUSE_BASE_URL         # https://<app>.pymthouse.example
pnpm wrangler secret put PYMTHOUSE_PUBLIC_CLIENT_ID # app_...
pnpm wrangler secret put PYMTHOUSE_M2M_CLIENT_ID    # m2m_...
pnpm wrangler secret put PYMTHOUSE_M2M_CLIENT_SECRET# pmth_cs_...
pnpm wrangler secret put ADMIN_API_KEY              # operator key for /admin/* (never ship)
pnpm wrangler secret put RESEND_API_KEY             # re_...
pnpm wrangler secret put EMAIL_FROM                 # credits@yourdomain.com
# (Phase B, optional) pnpm wrangler secret put WEBHOOK_SECRET
```
Default tiers (used if `STRIPE_TIERS` is unset):
```json
[{"creditsCents":1000,"feeCents":100},{"creditsCents":2500,"feeCents":150},{"creditsCents":5000,"feeCents":300},{"creditsCents":10000,"feeCents":500}]
```

### 7. Apply migrations & run
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
export BASE="http://localhost:8787"

curl -s "$BASE/health"
curl -s "$BASE/tiers"

# User routes need a per-user key (see "Per-user keys & the desktop flow"):
#   PVKEY="<apiKey returned by POST /provision>"
#   curl -s -H "Authorization: Bearer $PVKEY" "$BASE/balance"
```

`POST /webhook/stripe` can be exercised two ways:
- **Local dev:** `stripe listen --forward-to localhost:8787/webhook/stripe` (Stripe CLI) forwards
  live test events to your running worker, then complete a test checkout (`4242 4242 4242 4242`).
- **Deployed:** Stripe dashboard → Webhooks → "Send test webhook" against your worker URL.

In both cases confirm the credit lands in PymtHouse (`/usage` / `/balance`) and that `/admin/payments`
shows the `topup` row. The handler grants only `credit_usd_micros` and is idempotent by
Stripe `event.id`.

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

## Per-user keys & the desktop flow

What changes for the Video Creator desktop app (vs. a shared key):

1. **First run:** the Python backend generates a UUID `externalUserId`, then calls
   `POST /provision { externalUserId }` (public) and **stores the returned `apiKey` locally**
   (electron-store / secure settings). `/provision` returns a key **once** — re-calling it for
   the same UUID returns `409`, so a stolen UUID can't be used to seize a balance.
2. **Every user-route call** sends `Authorization: Bearer <apiKey>` (`/checkout`, `/balance`,
   `/usage`, `/link-email`…). There is **no `externalUserId` in the body** — the server derives
   the user from the key, so a client can't claim to be someone else.
3. **Key loss or compromise:** `POST /recover/request { email }` emails a one-time code, then
   `POST /recover/confirm { email, code }` proves ownership and **rotates** to a fresh key
   (returned in the response). No new provisioning needed.
4. **No-email recovery:** at sign-up `/provision` returns a **one-time backup code** (shown
   once, never emailed; only its SHA-256 is stored). Present it at `POST /recover/backup` to
   rotate to a fresh key — it is consumed and a **new** backup code is minted. Use this on the
   Login screen's "Backup code" tab if you don't want email recovery.
5. **Operators** can see which keys exist (`GET /admin/api-keys`, hashes only — never plaintext)
   and revoke one (`POST /admin/revoke-key { externalUserId }`); a revoked user must re-prove
   themselves via `/recover/confirm`.

Security notes:
- Only the **SHA-256 of each key is stored**; `/provision` never returns a key twice.
- Keys are 256-bit random (64 hex chars) — un-guessable and unsalted-hash lookup is safe.
- Revoking a key is instant and per-user, which the single shared key never allowed.

Self-test the whole flow locally with `pnpm smoke` (in-memory D1 + mocked PymtHouse/Stripe):
it exercises provision → dup-409 → authenticated balance → wrong-key 401 → checkout → admin
list → revoke-invalidates.

## Phase B (browser → orchestrator, billed by signed tickets)

`POST /sign-ticket` (direct DMZ proxy via `@pymthouse/builder-sdk/signer/server`), `GET /signer/address`,
and `POST /authorize` (the go-livepeer identity webhook) are implemented in `src/signer.ts`.
The browser submits generation to the orchestrator directly, with payment in the
`Livepeer-Payment` / `Livepeer-Segment` / `Livepeer-Payer-Address` headers; the Worker only ever
does control-plane work (auth, balance gate, signer proxy, identity webhook). See
`plans/PHASE_B_BROWSER_TICKET_FLOW.md` for the full flow and contracts. **Invariant: every balance
debit is authorized through PymtHouse** (DMZ signs on `/generate-live-payment`, PymtHouse metering
debits); this Worker never writes to a stored balance.

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
