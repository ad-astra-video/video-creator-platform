/**
 * Free ETH -> USD price lookup with source failover and a 30-minute client cache.
 *
 * The Livepeer discovery feed quotes prices in wei (1 ETH = 1e18 wei), so to show
 * a USD price we need the current ETH/USD rate. Sources are tried in order and the
 * first success wins:
 *   1. Coinbase (https://api.coinbase.com/v2/prices/ETH-USD/spot)
 *   2. Kraken   (https://api.kraken.com/0/public/Ticker?pair=ETHUSD)
 *   3. Binance  (https://api.binance.com/api/v3/ticker/price?symbol=ETHUSDT)
 * Binance is last because it is geo-blocked in some regions.
 *
 * The result is cached for 30 minutes both in memory and in localStorage, so the
 * rate survives page reloads and we don't hammer free endpoints. If every source
 * fails and no stale cache exists, getEthUsd() resolves to null (callers fall back).
 */

const CACHE_KEY = 'vc-eth-usd'
const CACHE_TTL_MS = 30 * 60 * 1000 // 30 minutes
const FETCH_TIMEOUT_MS = 8000
const WEI_PER_ETH = 1e18

interface CacheEntry {
  usd: number
  at: number
  source: string
}

let memCache: CacheEntry | null = null

interface Source {
  name: string
  fetch: () => Promise<number>
}

function toUsd(value: unknown): number {
  const v = typeof value === 'string' ? parseFloat(value) : typeof value === 'number' ? value : NaN
  return isFinite(v) && v > 0 ? v : NaN
}

const sources: Source[] = [
  {
    name: 'coinbase',
    fetch: async () => {
      const r = await fetch('https://api.coinbase.com/v2/prices/ETH-USD/spot', {
        signal: AbortSignal.timeout(FETCH_TIMEOUT_MS),
      })
      if (!r.ok) throw new Error(`coinbase HTTP ${r.status}`)
      const j: unknown = await r.json()
      const usd = toUsd((j as { data?: { amount?: unknown } })?.data?.amount)
      if (isNaN(usd)) throw new Error('coinbase bad payload')
      return usd
    },
  },
  {
    name: 'kraken',
    fetch: async () => {
      const r = await fetch('https://api.kraken.com/0/public/Ticker?pair=ETHUSD', {
        signal: AbortSignal.timeout(FETCH_TIMEOUT_MS),
      })
      if (!r.ok) throw new Error(`kraken HTTP ${r.status}`)
      const j: unknown = await r.json()
      const usd = toUsd((j as { result?: { XETHZUSD?: { c?: unknown[] } } })?.result?.XETHZUSD?.c?.[0])
      if (isNaN(usd)) throw new Error('kraken bad payload')
      return usd
    },
  },
  {
    name: 'binance',
    fetch: async () => {
      const r = await fetch('https://api.binance.com/api/v3/ticker/price?symbol=ETHUSDT', {
        signal: AbortSignal.timeout(FETCH_TIMEOUT_MS),
      })
      if (!r.ok) throw new Error(`binance HTTP ${r.status}`)
      const j: unknown = await r.json()
      const usd = toUsd((j as { price?: unknown })?.price)
      if (isNaN(usd)) throw new Error('binance bad payload')
      return usd
    },
  },
]

function loadFromStorage(): CacheEntry | null {
  try {
    const raw = localStorage.getItem(CACHE_KEY)
    if (!raw) return null
    const parsed = JSON.parse(raw) as Partial<CacheEntry>
    if (typeof parsed.usd === 'number' && typeof parsed.at === 'number' && parsed.usd > 0) {
      return { usd: parsed.usd, at: parsed.at, source: parsed.source ?? 'cache' }
    }
    return null
  } catch {
    return null
  }
}

function saveToStorage(entry: CacheEntry) {
  try {
    localStorage.setItem(CACHE_KEY, JSON.stringify(entry))
  } catch {
    /* non-fatal */
  }
}

/** Current ETH/USD rate, or null if every source failed (and no usable cache). Cached 30 min. */
export async function getEthUsd(): Promise<number | null> {
  const now = Date.now()

  // 1) In-memory cache, still fresh.
  if (memCache && now - memCache.at < CACHE_TTL_MS) return memCache.usd

  // 2) localStorage cache, still fresh (hydrates mem on reload).
  const stored = loadFromStorage()
  if (stored && now - stored.at < CACHE_TTL_MS) {
    memCache = stored
    return stored.usd
  }

  // 3) Fetch, failover across sources.
  for (const source of sources) {
    try {
      const usd = await source.fetch()
      const entry: CacheEntry = { usd, at: Date.now(), source: source.name }
      memCache = entry
      saveToStorage(entry)
      return usd
    } catch {
      /* try the next source */
    }
  }

  // 4) Every source failed — fall back to stale localStorage rather than return nothing.
  if (stored) {
    memCache = stored
    return stored.usd
  }
  return null
}

/** Convert a Livepeer wei price to a USD figure using ETH/USD. */
export function weiToUsd(wei: number, ethUsd: number): number {
  return (wei / WEI_PER_ETH) * ethUsd
}
