import { useCallback, useEffect, useState } from 'react'
import { AlertTriangle, DollarSign, Info, RefreshCw, Server } from 'lucide-react'
import { ApiClient } from '../../lib/api-client'
import { useAppSettings } from '../../contexts/AppSettingsContext'
import { isWebPlatform, discoverRunners } from '../../lib/livepeer-discovery'
import { getEthUsd, weiToUsd } from '../../lib/ethPrice'
import { runnerCapacity } from '../../lib/runner-availability'
import { Button } from '../ui/button'

interface RunnerCap {
  id: string
  label: string
}

interface ProviderDto {
  runner_id: string
  url: string
  status: string
  gpu?: { name?: string; vram_mb?: number } | null
  capacity?: number
  capacityUsed?: number
  capacityAvailable?: number
  price_info?: { price?: number; currency?: string; unit?: string; usdPerSec?: number; pricePerUnit?: number; pixelsPerUnit?: number } | null
  selected: boolean
  excluded: boolean
  demo?: boolean
  capabilities?: RunnerCap[]
  models?: string[]
}

// Browser-local list of runner addresses the user has chosen not to use ("excluded"),
// persisted so the choice is remembered for the same orchestrator/runner across sessions.
const STORAGE_KEY = 'vc-excluded-providers'

function loadExcluded(): string[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    const parsed = raw ? JSON.parse(raw) : []
    return Array.isArray(parsed) ? parsed.filter((x): x is string => typeof x === 'string') : []
  } catch {
    return []
  }
}

function saveExcluded(list: string[]) {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(list))
  } catch {
    /* non-fatal */
  }
}

function fmtPrice(usd: number, unit?: string): string {
  if (usd <= 0) return 'Free'
  // "fixed" = once per request/session. Anything else is a live (metered) billing
  // unit, shown by the unit it is advertised in (hour, seconds, 720p, ...).
  const u = (unit || '').toLowerCase()
  let suffix: string
  if (u === 'fixed') {
    suffix = ' per request'
  } else if (u === 'hour') {
    suffix = '/hr'
  } else if (u === 'second' || u === 'seconds') {
    suffix = '/sec'
  } else if (u) {
    suffix = `/${u}`
  } else {
    suffix = '/sec'
  }
  const s = usd < 0.0001 ? usd.toFixed(6) : usd < 1 ? usd.toFixed(4) : usd.toFixed(2)
  return `$${s}${suffix}`
}

function formatPrice(
  pi: NonNullable<ProviderDto['price_info']>,
  ethUsd: number | null,
): string | null {
  if (typeof pi.usdPerSec === 'number') return fmtPrice(pi.usdPerSec, pi.unit)
  if (typeof pi.price === 'number') {
    // Livepeer discovery quotes wei (1 ETH = 1e18 wei); convert to USD using the rate.
    if (pi.currency === 'wei') {
      if (ethUsd == null) return null // rate not available yet -> 'Pricing not advertised'
      return fmtPrice(weiToUsd(pi.price, ethUsd), pi.unit)
    }
    return fmtPrice(pi.price, pi.unit)
  }
  if (typeof pi.pricePerUnit === 'number') {
    return pi.pricePerUnit === 0 ? 'Free' : `${pi.pricePerUnit} wei / ${pi.pixelsPerUnit ?? 1} px`
  }
  return null
}

export function RunnersSection() {
  const { settings: appSettings } = useAppSettings()
  const [providers, setProviders] = useState<ProviderDto[]>([])
  const [excluded, setExcluded] = useState<string[]>(() => loadExcluded())
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [discoveryUrl, setDiscoveryUrl] = useState('')
  const [ethUsd, setEthUsd] = useState<number | null>(null)

  // ETH/USD for converting wei discovery prices to USD (failover feeds, 30-min cache).
  useEffect(() => {
    let alive = true
    void getEthUsd().then(v => {
      if (alive) setEthUsd(v)
    })
    return () => {
      alive = false
    }
  }, [])

  const isExcluded = useCallback((p: ProviderDto) => p.excluded || excluded.includes(p.url), [excluded])

  const toggleExclude = (p: ProviderDto) => {
    setExcluded(prev => {
      const next = prev.includes(p.url) ? prev.filter(u => u !== p.url) : [...prev, p.url]
      saveExcluded(next)
      return next
    })
  }

  const refresh = useCallback(async () => {
    setLoading(true)
    setError(null)
    let providers: ProviderDto[] = []
    let url = ''
    if (isWebPlatform()) {
      // Web build: discover straight against the configured Discovery URL (from app settings),
      // not the Worker's /api/providers.
      url = appSettings.livepeerDiscoveryUrl ?? ''
      if (url) {
        try {
          providers = (await discoverRunners(url)) as unknown as ProviderDto[]
        } catch (e) {
          setError(e instanceof Error ? e.message : 'Could not reach Discovery URL')
          providers = []
        }
      }
    } else {
      const [pr, st] = await Promise.all([ApiClient.getProviders(), ApiClient.getSettings()])
      if (!pr.ok) {
        setError(pr.error.message || 'Failed to load runners.')
        providers = []
      } else {
        providers = (pr.data?.providers ?? []) as ProviderDto[]
      }
      if (st.ok) url = st.data.livepeerDiscoveryUrl ?? ''
    }
    setProviders(providers)
    setDiscoveryUrl(url)
    setLoading(false)
  }, [appSettings.livepeerDiscoveryUrl])

  useEffect(() => {
    void refresh()
  }, [refresh])

  // Keep capacity/availability live: re-poll the discovery every 30s while mounted.
  useEffect(() => {
    const iv = setInterval(() => { void refresh() }, 30000)
    return () => clearInterval(iv)
  }, [refresh])

  const demo = providers.length > 0 && providers.every(p => p.demo)
  const excludedCount = providers.filter(isExcluded).length

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Server className="h-4 w-4 text-blue-400" />
          <h3 className="text-sm font-semibold text-white">Available Runners</h3>
        </div>
        <Button variant="outline" size="sm" className="border-zinc-700" onClick={() => void refresh()} disabled={loading}>
          <RefreshCw className={`h-3.5 w-3.5 ${loading ? 'animate-spin' : ''}`} /> Refresh
        </Button>
      </div>

      <p className="text-xs text-zinc-500 leading-relaxed">
        Runners advertise the models they can execute and the price to run them. You can exclude a runner to
        stop it being used.
        {demo ? ' No live runner is connected yet — showing reference runners from the platform catalog.' : ''}
      </p>

      {discoveryUrl ? (
        <div className="flex items-center gap-1.5 text-[11px] text-zinc-500">
          <Info className="h-3 w-3 flex-shrink-0" /> Discovery:{' '}
          <span className="text-zinc-400 truncate">{discoveryUrl}</span>
        </div>
      ) : null}

      {excludedCount > 0 ? (
        <div className="text-[11px] text-amber-400/90">
          {excludedCount} runner{excludedCount === 1 ? '' : 's'} excluded — they won't be used.
        </div>
      ) : null}

      {error && <div className="text-xs text-red-400">{error}</div>}

      {loading ? (
        <div className="text-xs text-zinc-600">Loading runners…</div>
      ) : providers.length === 0 ? (
        <div className="rounded-lg border border-dashed border-zinc-800 p-4 text-xs text-zinc-600">
          No runners currently available. Configure a Livepeer Discovery URL in the API Keys tab, or wait for
          orchestrator runners to come online.
        </div>
      ) : (
        <div className="space-y-2">
          {demo && (
            <div className="flex items-start gap-2 rounded-lg border border-amber-500/40 bg-amber-500/10 px-3 py-2 text-xs text-amber-200 leading-relaxed">
              <AlertTriangle className="h-4 w-4 mt-0.5 flex-shrink-0" />
              <span>
                <span className="font-semibold">Demo placeholders.</span> The runners below are sample
                entries from the platform catalog and <span className="font-semibold">cannot perform real work</span>.
                Configure a Livepeer Discovery URL in Settings to connect real runners.
              </span>
            </div>
          )}
          {providers.map(p => {
            const ex = isExcluded(p)
            const vramMb = p.gpu?.vram_mb
            const hasAdvertisedCapacity = typeof p.capacity === 'number' || typeof p.capacityAvailable === 'number'
            const cap = runnerCapacity({ capacity: p.capacity, capacityUsed: p.capacityUsed, capacityAvailable: p.capacityAvailable, vramMb, modelIds: p.models ?? [] })
            const capacityInUse = hasAdvertisedCapacity && typeof p.capacityUsed === 'number' ? p.capacityUsed : null
            return (
              <div key={p.runner_id} className={`rounded-lg bg-zinc-800/50 p-3 space-y-2 ${ex ? 'opacity-55' : ''}`}>
                <div className="flex items-center justify-between gap-2">
                  <div className="flex items-center gap-2 min-w-0">
                    <span className="text-sm text-white truncate">{p.gpu?.name || p.runner_id}</span>
                    {p.demo && (
                      <span className="text-[10px] px-1.5 py-0.5 rounded bg-amber-500/10 text-amber-400 flex-shrink-0">
                        demo
                      </span>
                    )}
                    {ex && (
                      <span className="text-[10px] px-1.5 py-0.5 rounded bg-red-500/10 text-red-400 flex-shrink-0">
                        excluded
                      </span>
                    )}
                  </div>
                  <div className="flex items-center gap-1.5 flex-shrink-0">
                    <span
                      className={`text-[10px] px-1.5 py-0.5 rounded ${
                        p.status === 'ready' ? 'bg-green-500/10 text-green-400' : 'bg-zinc-700 text-zinc-300'
                      }`}
                    >
                      {p.status}
                    </span>
                    <label className="flex items-center gap-1.5 cursor-pointer select-none" title={ex ? 'Allow this runner' : 'Exclude this runner'}>
                      <span className={`text-[11px] ${ex ? 'text-red-400' : 'text-zinc-300'}`}>Exclude</span>
                      <button
                        type="button"
                        role="switch"
                        aria-checked={ex}
                        aria-label="Exclude runner"
                        onClick={() => toggleExclude(p)}
                        className={`relative inline-flex h-4 w-7 items-center rounded-full transition-colors focus:outline-none focus:ring-2 focus:ring-blue-500/50 ${ex ? 'bg-red-500' : 'bg-zinc-700'}`}
                      >
                        <span
                          className={`inline-block h-3.5 w-3.5 transform rounded-full bg-white transition-transform ${ex ? 'translate-x-3' : 'translate-x-0.5'}`}
                        />
                      </button>
                    </label>
                  </div>
                </div>

                <div className="flex items-center gap-2 text-xs text-zinc-300">
                  <DollarSign className="h-3.5 w-3.5 text-green-400 flex-shrink-0" />
                  {p.price_info ? formatPrice(p.price_info, ethUsd) ?? 'Pricing not advertised' : 'Pricing not advertised'}
                </div>

                <div className="flex items-center gap-2 text-xs text-zinc-400">
                  <span className="font-medium text-zinc-300">Capacity:</span>
                  {vramMb ? (
                    <span className="text-zinc-300">{Math.round(vramMb / 1024)} GB VRAM</span>
                  ) : (
                    <span>VRAM unknown</span>
                  )}
                  {cap != null && (
                    <span className={hasAdvertisedCapacity ? 'text-emerald-400' : 'text-emerald-400/90'}>
                      {hasAdvertisedCapacity ? `· ${cap} concurrent` : `· ~${cap} concurrent (est.)`}
                      {capacityInUse != null && cap > 0 ? ` · ${capacityInUse} in use` : ''}
                    </span>
                  )}
                </div>

                {(p.capabilities?.length ?? 0) > 0 && (
                  <div className="flex flex-wrap gap-1.5">
                    {p.capabilities!.map(c => (
                      <span key={c.id} className="text-[10px] px-1.5 py-0.5 rounded bg-blue-500/10 text-blue-400">
                        {c.label}
                      </span>
                    ))}
                  </div>
                )}
                {p.models && p.models.length > 0 && (
                  <div className="flex flex-wrap items-center gap-1.5 text-[10px] text-zinc-500">
                    <span>Models:</span>
                    {p.models.map(m => (
                      <span key={m} className="px-1.5 py-0.5 rounded bg-purple-500/10 text-purple-400">
                        {m}
                      </span>
                    ))}
                  </div>
                )}
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}
