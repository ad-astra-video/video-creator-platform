// Smoke test for the per-user key auth flow. In-memory D1 + mocked PymtHouse.
import worker from './dist/worker.js';

// --- In-memory D1 (mini) ---
const apiKeys = new Map(); // externalUserId -> { key_hash, revoked }
const accounts = new Map(); // externalUserId -> {}
function fakeDB() {
  const prep = (sql) => {
    const args = [];
    const stmt = {
      bind: (...more) => { args.push(...more); return stmt; },
      async all() { return { results: [] }; },
      async first() {
        if (sql.includes('SELECT external_user_id FROM api_keys')) {
          const hash = args[0];
          for (const [uid, row] of apiKeys) if (row.key_hash === hash && row.revoked === 0) return { external_user_id: uid };
        }
        if (sql.includes('SELECT * FROM api_keys')) return null;
        if (sql.includes('SELECT pending_email FROM accounts')) return { pending_email: null };
        return null;
      },
      async run() {
        if (sql.includes('INSERT OR IGNORE INTO api_keys')) {
          const [uid, hash] = args;
          if (apiKeys.has(uid)) return { meta: { changes: 0 } };
          apiKeys.set(uid, { key_hash: hash, revoked: 0 }); return { meta: { changes: 1 } };
        }
        if (sql.includes('UPDATE api_keys SET revoked')) { const [uid] = args; if (apiKeys.has(uid)) apiKeys.get(uid).revoked = 1; return { meta: { changes: 1 } }; }
        if (sql.includes('INSERT INTO accounts')) { accounts.set(args[0], {}); return { meta: { changes: 1 } }; }
        return { meta: { changes: 1 } };
      },
    };
    return stmt;
  };
  return { prepare: prep };
}

// --- Mock outbound: PymtHouse + Stripe ---
const realFetch = globalThis.fetch;
globalThis.fetch = async (url, opts = {}) => {
  const u = String(url);
  if (u.includes('pymthouse.example') && u.includes('/usage/balance')) {
    return new Response(JSON.stringify({ hasAccess: true, balanceUsdMicros: '10000000', remainingUsdMicros: '9000000', consumedUsdMicros: '1000000', lifetimeGrantedUsdMicros: '10000000' }), { status: 200 });
  }
  if (u.includes('pymthouse.example')) return new Response(JSON.stringify({}), { status: 200 });
  if (u.includes('api.stripe.com')) return new Response(JSON.stringify({ url: 'https://pay.example/checkout', id: 'cs_1' }), { status: 200 });
  return realFetch(url, opts);
};

const env = {
  DB: fakeDB(), STRIPE_TIERS: undefined,
  STRIPE_SECRET_KEY: 'sk_test', STRIPE_WEBHOOK_SECRET: 'whsec_x',
  PYMTHOUSE_BASE_URL: 'https://app.pymthouse.example',
  PYMTHOUSE_PUBLIC_CLIENT_ID: 'app_x', PYMTHOUSE_M2M_CLIENT_ID: 'm2m_x',
  PYMTHOUSE_M2M_CLIENT_SECRET: 's',
  ADMIN_API_KEY: 'adminkey', RESEND_API_KEY: '', EMAIL_FROM: 'a@b.c',
  WEBHOOK_SECRET: '', ALLOWED_ORIGIN: '*',
};

const j = async (x) => { const r = await worker.fetch(x, env); return { s: r.status, b: JSON.parse(await r.text()) }; };
const bearer = (k) => ({ headers: { authorization: 'Bearer ' + k, 'content-type': 'application/json' } });

const A = 'uuid-A';
// 1. provision user A -> fresh key
let r = await j(new Request('https://x/provision', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ externalUserId: A }) }));
const keyA = r.b.apiKey;
console.log('provision A          ->', r.s, 'got key:', !!keyA, 'len', (keyA||'').length);
// 2. re-provision same UUID -> 409 (prevents key theft)
r = await j(new Request('https://x/provision', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ externalUserId: A }) }));
console.log('re-provision A (dup) ->', r.s, r.b.error);
// 3. /balance with keyA -> 200, user resolved from key
r = await j(new Request('https://x/balance', bearer(keyA)));
console.log('balance keyA         ->', r.s, 'hasAccess:', r.b.hasAccess);
// 4. wrong key -> 401
r = await j(new Request('https://x/balance', bearer('deadbeef')));
console.log('balance wrong key    ->', r.s);
// 5. no key -> 401
r = await j(new Request('https://x/balance', {}));
console.log('balance no key       ->', r.s);
// 6. checkout (user from key, no externalUserId in body) -> 200 url
r = await j(new Request('https://x/checkout', { method: 'POST', ...bearer(keyA), body: JSON.stringify({ tier: 1000 }) }));
console.log('checkout keyA        ->', r.s, 'url ok:', !!r.b.url);
// 7. admin lists keys -> sees A
r = await j(new Request('https://x/admin/api-keys', { headers: { authorization: 'Bearer adminkey' } }));
console.log('admin api-keys       ->', r.s, 'count:', (r.b.apiKeys||[]).length);
// 8. admin revokes A -> then keyA invalid
await j(new Request('https://x/admin/revoke-key', { method: 'POST', headers: { authorization: 'Bearer adminkey', 'content-type': 'application/json' }, body: JSON.stringify({ externalUserId: A }) }));
r = await j(new Request('https://x/balance', bearer(keyA)));
console.log('balance after revoke ->', r.s);
