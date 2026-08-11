import { useCallback, useEffect, useState } from 'react'
import { DollarSign, Info, RefreshCw, Server } from 'lucide-react'
import { ApiClient } from '../../lib/api-client'
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
  price_info?: { price?: number; currency?: string; unit?: string; usdPerSec?: number; pricePerUnit?: number; pixelsPerUnit?: number } | null
  selected: boolean
  excluded: boolean
  demo?: boolean
  capabilities?: RunnerCap[]
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

function fmtPrice(usd: number): string {
  if (usd <= 0) return 'Free'
  if (usd < 0.0001) return `$${usd.toFixed(6)}/sec`
  return `$${usd.toFixed(4)}/sec`
}

function formatPrice(pi: NonNullable<ProviderDto['price_info']>): string | null {
  if (typeof pi.usdPerSec === 'number') return fmtPrice(pi.usdPerSec)
  if (typeof pi.price === 'number') return fmtPrice(pi.price)
  if (typeof pi.pricePerUnit === 'number') {
    return pi.pricePerUnit === 0 ? 'Free' : `${pi.pricePerUnit} wei / ${pi.pixelsPerUnit ?? 1} px`
  }
  return null
}

export function RunnersSection() {
  const [providers, setProviders] = useState<ProviderDto[]>([])
  const [excluded, setExcluded] = useState<string[]>(() => loadExcluded())
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [discoveryUrl, setDiscoveryUrl] = useState('')

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
    const [pr, st] = await Promise.all([ApiClient.getProviders(), ApiClient.getSettings()])
    if (!pr.ok) {
      setError(pr.error.message || 'Failed to load runners.')
      setProviders([])
    } else {
      setProviders((pr.data?.providers ?? []) as ProviderDto[])
    }
    if (st.ok) setDiscoveryUrl(st.data.livepeerDiscoveryUrl ?? '')
    setLoading(false)
  }, [])

  useEffect(() => {
    void refresh()
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
          {providers.map(p => {
            const ex = isExcluded(p)
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
                  {p.price_info ? formatPrice(p.price_info) ?? 'Pricing not advertised' : 'Pricing not advertised'}
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
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}
