// Smoke test for the per-user key auth flow + Phase 2 dispatch/catalog/provider routes.
// In-memory D1 + mocked PymtHouse/Stripe/orchestrator.
import worker from './dist/worker.js';

// --- In-memory D1 (mini) ---
const apiKeys = new Map(); // externalUserId -> { key_hash, revoked }
const accounts = new Map(); // externalUserId -> {}
const jobs = new Map();     // jobId -> { id, user_id, status, runner, ... }

function firstByJobs(sql, args) {
  if (sql.includes('FROM jobs') && sql.includes(' WHERE id = ?1')) {
    const job = jobs.get(args[0]);
    if (job && (!sql.includes('user_id = ?2') || job.user_id === args[1])) return job;
  }
  return null;
}

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
        const j = firstByJobs(sql, args);
        if (j) return j;
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
        if (sql.includes('INSERT INTO jobs')) {
          const [id, uid] = args;
          jobs.set(id, { id, user_id: uid, status: 'queued', runner: null, created_at: new Date().toISOString(), updated_at: new Date().toISOString() });
          return { meta: { changes: 1 } };
        }
        if (sql.includes('UPDATE jobs SET')) {
          const last = args[args.length - 1];
          const job = jobs.get(last);
          if (job) {
            // name=val bindings precede the id (the id is the last bound arg)
            for (let i = 0; i < args.length - 1; i++) {
              const key = ['status','runner','request_json','updated_at'][i];
              if (key && sql.includes(key + ' =')) job[key] = typeof args[i] === 'string' && key !== 'runner' ? args[i] : args[i];
            }
          }
          return { meta: { changes: 1 } };
        }
        return { meta: { changes: 1 } };
      },
    };
    return stmt;
  };
  return { prepare: prep };
}

// --- Mock outbound: PymtHouse + Stripe + orchestrator + runner ---
const realFetch = globalThis.fetch;
const ORCH = 'http://orchestrator.test';
globalThis.fetch = async (url, opts = {}) => {
  const u = String(url);
  if (u.includes('pymthouse.example') && u.includes('/usage/balance')) {
    return new Response(JSON.stringify({ hasAccess: true, balanceUsdMicros: '10000000', remainingUsdMicros: '9000000', consumedUsdMicros: '1000000', lifetimeGrantedUsdMicros: '10000000' }), { status: 200 });
  }
  if (u.includes('pymthouse.example')) return new Response(JSON.stringify({ ok: true, remainingUsdMicros: '9000000' }), { status: 200 });
  if (u.includes('api.stripe.com')) return new Response(JSON.stringify({ url: 'https://pay.example/checkout', id: 'cs_1' }), { status: 200 });
  if (u.startsWith(ORCH + '/api/discovery')) {
    return new Response(JSON.stringify({ runners: [
      { id: 'r1', name: 'runner-1', url: 'http://runner-1:8000', status: 'ready',
        capabilities: { tasks: ['t2v','image','prompt','extend','retake','restyle','ic-lora','sam3'] },
        priceUsdMicrosPerSec: 1200 },
    ] }), { status: 200 });
  }
  if (u.startsWith(ORCH + '/api/jobs')) {
    if ((opts.method || 'GET') === 'POST') return new Response(JSON.stringify({ jobId: 'job-1', status: 'queued' }), { status: 201 });
    return new Response(JSON.stringify({ id: 'job-1', status: 'running' }), { status: 200 });
  }
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
  ORCHESTRATOR_BASE_URL: ORCH, JOB_COST_USD_MICROS: '100000',
};

const j = async (x) => { const r = await worker.fetch(x, env); return { s: r.status, b: JSON.parse(await r.text()) }; };
const bearer = (k) => ({ headers: { authorization: 'Bearer ' + k, 'content-type': 'application/json' } });

const A = 'uuid-A';
let r = await j(new Request('https://x/provision', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ externalUserId: A }) }));
const keyA = r.b.apiKey;
console.log('provision A          ->', r.s, 'got key:', !!keyA, 'len', (keyA||'').length);
r = await j(new Request('https://x/provision', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ externalUserId: A }) }));
console.log('re-provision A (dup) ->', r.s, r.b.error);
r = await j(new Request('https://x/balance', bearer(keyA)));
console.log('balance keyA         ->', r.s, 'hasAccess:', r.b.hasAccess);
r = await j(new Request('https://x/balance', bearer('deadbeef')));
console.log('balance wrong key    ->', r.s);
r = await j(new Request('https://x/balance', {}));
console.log('balance no key       ->', r.s);
r = await j(new Request('https://x/checkout', { method: 'POST', ...bearer(keyA), body: JSON.stringify({ tier: 1000 }) }));
console.log('checkout keyA        ->', r.s, 'url ok:', !!r.b.url);
r = await j(new Request('https://x/admin/api-keys', { headers: { authorization: 'Bearer adminkey' } }));
console.log('admin api-keys       ->', r.s, 'count:', (r.b.apiKeys||[]).length);

// ---- Phase 2: no-key public health (no auth) ----
r = await j(new Request('https://x/health'));
console.log('health no key        ->', r.s, 'status:', r.b.status);

// ---- Phase 2: auth-gated catalog / providers / settings ----
r = await j(new Request('https://x/api/models', bearer(keyA)));
console.log('api/models           ->', r.s, 'has models:', Array.isArray(r.b.models) || Array.isArray(r.b));
r = await j(new Request('https://x/api/models', {}));
console.log('api/models no key    ->', r.s);
r = await j(new Request('https://x/api/catalog', bearer(keyA)));
console.log('api/catalog          ->', r.s, 'keys:', Object.keys(r.b));
r = await j(new Request('https://x/api/providers', bearer(keyA)));
console.log('api/providers        ->', r.s);

// ---- Phase 2: dispatch generate -> jobId, then progress ----
r = await j(new Request('https://x/api/generate', { method: 'POST', ...bearer(keyA), body: JSON.stringify({ prompt: 'a cat on a skateboard', numFrames: 81, width: 480, height: 720 }) }));
console.log('generate dispatch    ->', r.s, 'jobId:', r.b.jobId, r.b.error || '');
const jobId = r.b.jobId;
if (r.s === 200 && jobId) {
  r = await j(new Request('https://x/api/generation/progress?jobId=' + jobId, bearer(keyA)));
  console.log('generation progress ->', r.s, 'status:', r.b.status, 'jobId match:', r.b.jobId === jobId);
} else {
  console.log('generate dispatch FAILED (see above); skipping progress');
}

// ---- Phase 2: no key on dispatch = 401 ----
r = await j(new Request('https://x/api/generate', { method: 'POST', body: JSON.stringify({ prompt: 'x' }) }));
console.log('generate no key       ->', r.s);

// ---- Existing: revoke invalidates key ----
await j(new Request('https://x/admin/revoke-key', { method: 'POST', headers: { authorization: 'Bearer adminkey', 'content-type': 'application/json' }, body: JSON.stringify({ externalUserId: A }) }));
r = await j(new Request('https://x/balance', bearer(keyA)));
console.log('balance after revoke ->', r.s);
