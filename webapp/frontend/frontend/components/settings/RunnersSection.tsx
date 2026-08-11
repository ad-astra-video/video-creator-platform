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
  price_info?: { price: number; currency: string; unit: string } | null
  selected: boolean
  excluded: boolean
  demo?: boolean
  capabilities?: RunnerCap[]
}

function fmtPrice(priceUsdPerSec: number): string {
  if (priceUsdPerSec >= 0.1) return `$${priceUsdPerSec.toFixed(2)}/sec`
  if (priceUsdPerSec >= 0.001) return `$${priceUsdPerSec.toFixed(4)}/sec`
  return `$${priceUsdPerSec.toFixed(6)}/sec`
}

/**
 * "Available Runners" — shows the runners the orchestrator discovered, which models
 * (capabilities/tasks) each can run, and the price to run them. Backed by the real
 * GET /api/providers (orchestrator discovery; demo set in local dev until runners exist).
 */
export function RunnersSection() {
  const [providers, setProviders] = useState<ProviderDto[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [discoveryUrl, setDiscoveryUrl] = useState('')

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

  const demo = providers.length > 0 && providers.every((p) => p.demo)

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Server className="h-4 w-4 text-blue-400" />
          <h3 className="text-sm font-semibold text-white">Available Runners</h3>
        </div>
        <Button
          variant="outline"
          size="sm"
          className="border-zinc-700 flex-shrink-0"
          onClick={() => void refresh()}
          disabled={loading}
        >
          <RefreshCw className={`h-3.5 w-3.5 ${loading ? 'animate-spin' : ''}`} /> Refresh
        </Button>
      </div>

      <p className="text-xs text-zinc-500 leading-relaxed">
        Runners advertise the models they can execute and the price to run them.
        {demo
          ? ' No live runner is connected yet — showing reference models and pricing from the platform catalog.'
          : ''}
      </p>

      {discoveryUrl ? (
        <div className="flex items-center gap-1.5 text-[11px] text-zinc-500">
          <Info className="h-3 w-3 flex-shrink-0" /> Discovery: <span className="text-zinc-400 truncate">{discoveryUrl}</span>
        </div>
      ) : null}

      {error && <div className="text-xs text-red-400">{error}</div>}

      {loading ? (
        <div className="text-xs text-zinc-600">Loading runners…</div>
      ) : providers.length === 0 ? (
        <div className="rounded-lg border border-dashed border-zinc-800 p-4 text-xs text-zinc-600">
          No runners currently available. Configure a Livepeer Discovery URL in the API Keys tab, or
          wait for orchestrator runners to come online.
        </div>
      ) : (
        <div className="space-y-2">
          {providers.map((p) => (
            <div key={p.runner_id} className="rounded-lg bg-zinc-800/50 p-3 space-y-2">
              <div className="flex items-center justify-between gap-2">
                <div className="flex items-center gap-2 min-w-0">
                  <span className="text-sm text-white truncate">{p.gpu?.name || p.runner_id}</span>
                  {p.demo && (
                    <span className="text-[10px] px-1.5 py-0.5 rounded bg-amber-500/10 text-amber-400 flex-shrink-0">
                      demo
                    </span>
                  )}
                </div>
                <span
                  className={`text-[10px] px-1.5 py-0.5 rounded flex-shrink-0 ${
                    p.status === 'ready' ? 'bg-green-500/10 text-green-400' : 'bg-zinc-700 text-zinc-300'
                  }`}
                >
                  {p.status}
                </span>
              </div>

              <div className="flex items-center gap-2 text-xs text-zinc-300">
                <DollarSign className="h-3.5 w-3.5 text-green-400 flex-shrink-0" />
                {p.price_info ? fmtPrice(p.price_info.price) : 'Pricing unavailable'}
              </div>

              {(p.capabilities?.length ?? 0) > 0 && (
                <div className="flex flex-wrap gap-1.5">
                  {p.capabilities!.map((c) => (
                    <span key={c.id} className="text-[10px] px-1.5 py-0.5 rounded bg-blue-500/10 text-blue-400">
                      {c.label}
                    </span>
                  ))}
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
