import { useCallback, useEffect, useState } from 'react'
import { AlertCircle, CreditCard, Receipt, RefreshCw, Wallet } from 'lucide-react'
import { ApiClient, type PlatformBalance, type PlatformHistory, type PlatformStatus } from '../lib/api-client'

const TIERS: Array<{ label: string; cents: number; charge: string }> = [
  { label: '$10', cents: 1000, charge: 'pays $11' },
  { label: '$25', cents: 2500, charge: 'pays $26.50' },
  { label: '$50', cents: 5000, charge: 'pays $53' },
  { label: '$100', cents: 10000, charge: 'pays $105' },
]

function formatMicros(micros: number): string {
  return `$${(micros / 1_000_000).toFixed(2)}`
}

/**
 * Account summary for the Video-Creator platform: credit balance and other
 * account facts that are safe to display (no hashes, no keys). Rendered as a
 * compact card under "New Project" in the sidebar and as the full Account tab
 * in the settings modal.
 */
export function AccountPanel({ compact = false }: { compact?: boolean }) {
  const [status, setStatus] = useState<PlatformStatus | null>(null)
  const [balance, setBalance] = useState<PlatformBalance | null>(null)
  const [history, setHistory] = useState<PlatformHistory | null>(null)
  const [loading, setLoading] = useState(false)
  const [busyTier, setBusyTier] = useState<number | null>(null)
  const [message, setMessage] = useState<{ kind: 'ok' | 'err'; text: string } | null>(null)

  // "Configured" = the platform actually answers (PymtHouse reachable on the Worker),
  // NOT the desktop-only `settings.hasPlatformBaseUrl` flag (which is never set in the
  // web build and would leave this panel permanently in its "no platform" empty state
  // while the header GlobalBalanceButton shows a live balance). status.configured is
  // the authoritative answer from GET /api/platform/status.
  const configured = Boolean(status?.configured)

  const load = useCallback(async () => {
    setLoading(true)
    const [s, b, h] = await Promise.all([
      ApiClient.getPlatformStatus(),
      ApiClient.getPlatformBalance(),
      ApiClient.getPlatformHistory(),
    ])
    if (s.ok) setStatus(s.data)
    if (b.ok) setBalance(b.data)
    if (h.ok) setHistory(h.data)
    setLoading(false)
  }, [])

  // Load once a platform server is configured (and whenever it changes, so a
  // just-saved URL immediately reflects balance).
  useEffect(() => {
    void load()
  }, [load])

  const topUp = async (cents: number) => {
    setBusyTier(cents)
    setMessage(null)
    const result = await ApiClient.createPlatformCheckout(cents)
    setBusyTier(null)
    if (!result.ok) {
      setMessage({ kind: 'err', text: result.error.message })
      return
    }
    void window.electronAPI.openExternalUrl({ url: result.data.url })
    setMessage({ kind: 'ok', text: 'Opening secure checkout in your browser. Refresh when you’re done.' })
  }

  // Compact sidebar variant (main dashboard): just the wallet icon + live $$ balance,
  // with a quiet refresh button on the right. No explanatory text (the settings
  // Account tab carries the detail).
  if (compact) {
    return (
      <div className="flex items-center justify-between gap-2">
        <div className="flex items-center gap-2 min-w-0">
          <Wallet className="h-4 w-4 text-amber-400 shrink-0" />
          <span className="text-sm font-semibold text-white truncate" title="Platform credits">
            {balance ? formatMicros(Number(balance.balanceUsdMicros)) : '—'}
          </span>
        </div>
        <button
          onClick={() => void load()}
          disabled={loading}
          className="inline-flex items-center text-zinc-400 hover:text-white disabled:text-zinc-600 shrink-0"
          title="Refresh balance"
        >
          <RefreshCw className={`h-3.5 w-3.5 ${loading ? 'animate-spin' : ''}`} />
        </button>
      </div>
    )
  }

  // Full variant (settings Account tab).
  return (
    <div className="space-y-4 pt-4 border-t border-zinc-800">
      <div>
        <p className="text-xs text-zinc-500 leading-relaxed">
          Your platform account. Balance is charged for remote generation (inference is pass-through);
          a small platform fee is added at checkout.
        </p>
      </div>

      {!configured ? (
        <div className="bg-zinc-800/50 rounded-lg p-4">
          <div className="text-xs text-zinc-500 bg-zinc-900/60 rounded-lg px-3 py-2 inline-flex items-center gap-1.5">
            <AlertCircle className="h-3 w-3" /> No platform connected — set a Platform server URL in General to see your
            balance.
          </div>
        </div>
      ) : (
        <>
          {/* Balance */}
          <div className="bg-zinc-800/50 rounded-lg p-4 space-y-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <CreditCard className="h-4 w-4 text-emerald-400" />
                <span className="text-sm font-medium text-white">Credits</span>
              </div>
              <button
                onClick={() => { setMessage(null); void load() }}
                disabled={loading}
                className="inline-flex items-center gap-1.5 text-xs text-emerald-400 hover:text-emerald-300 disabled:text-zinc-500"
              >
                <RefreshCw className={`h-3.5 w-3.5 ${loading ? 'animate-spin' : ''}`} /> Refresh
              </button>
            </div>

            {balance?.hasAccess === false && (
              <div className="bg-amber-500/10 border border-amber-500/30 rounded-lg px-3 py-2 text-xs text-amber-300 flex items-center gap-2">
                <AlertCircle className="h-4 w-4 flex-shrink-0" />
                Insufficient credits — top up below to continue remote generation.
              </div>
            )}

            <div className="grid grid-cols-3 gap-3 text-xs">
              <div>
                <p className="text-zinc-500">Balance</p>
                <p className="text-lg font-semibold text-white">{balance ? formatMicros(Number(balance.balanceUsdMicros)) : '—'}</p>
              </div>
              <div>
                <p className="text-zinc-500">Consumed</p>
                <p className="text-lg font-semibold text-zinc-300">{balance ? formatMicros(Number(balance.consumedUsdMicros)) : '—'}</p>
              </div>
              <div>
                <p className="text-zinc-500">Lifetime granted</p>
                <p className="text-lg font-semibold text-zinc-300">{balance ? formatMicros(Number(balance.lifetimeGrantedUsdMicros)) : '—'}</p>
              </div>
            </div>

            {/* Top up */}
            <div className="space-y-2">
              <p className="text-xs text-zinc-300 font-medium">Top up credits</p>
              <div className="grid grid-cols-4 gap-2">
                {TIERS.map((tier) => (
                  <button
                    key={tier.cents}
                    onClick={() => topUp(tier.cents)}
                    disabled={busyTier !== null}
                    className="px-2 py-2 bg-zinc-900 border border-zinc-700 rounded-lg text-center hover:bg-zinc-800 disabled:opacity-50 transition-colors"
                  >
                    <span className="block text-sm font-semibold text-white">{tier.label}</span>
                    <span className="block text-[10px] text-zinc-500">{tier.charge}</span>
                  </button>
                ))}
              </div>
              <p className="text-[10px] text-zinc-600">Opens a secure checkout in your browser. Refresh after paying to see your new balance.</p>
            </div>
          </div>

          {/* Account status (only non-hash facts) */}
          <div className="bg-zinc-800/50 rounded-lg p-4 space-y-2">
            <div className="flex items-center gap-2">
              <Wallet className="h-4 w-4 text-blue-400" />
              <span className="text-sm font-medium text-white">Account status</span>
            </div>
            <div className="flex items-center gap-2 text-xs">
              <span className={`w-1.5 h-1.5 rounded-full ${configured ? 'bg-emerald-400' : 'bg-zinc-600'}`} />
              <span className="text-zinc-300">{configured ? 'Connected' : 'Not connected'}</span>
              {status?.configured && status.baseUrl && (
                <span className="text-zinc-600 truncate">· {status.baseUrl}</span>
              )}
            </div>
          </div>

          {/* Spending history + per-project breakdown */}
          <div className="bg-zinc-800/50 rounded-lg p-4 space-y-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Receipt className="h-4 w-4 text-amber-400" />
                <span className="text-sm font-medium text-white">Spending history</span>
              </div>
              <span className="text-[10px] text-zinc-500">charged at generation</span>
            </div>

            {!history || history.history.length === 0 ? (
              <p className="text-[11px] text-zinc-500 leading-relaxed">
                No spend recorded yet. Every paid generation is listed here the moment its payment
                ticket is signed (per-project breakdown below).
              </p>
            ) : (
              <>
                {/* Per-project totals */}
                <div className="space-y-1.5">
                  {history.perProject
                    .filter((p) => p.projectId !== null)
                    .map((p) => (
                      <div key={p.projectId} className="flex items-center justify-between text-xs">
                        <span className="text-zinc-400 truncate pr-2">{p.projectId}</span>
                        <span className="text-zinc-200 font-medium whitespace-nowrap">
                          {formatMicros(Number(p.totalUsdMicros))}
                          <span className="text-zinc-600 font-normal"> · {p.count}</span>
                        </span>
                      </div>
                    ))}
                </div>

                {/* Recent transactions */}
                <div className="border-t border-zinc-700/60 pt-2 space-y-1.5 max-h-48 overflow-y-auto">
                  {history.history.slice(0, 20).map((entry) => (
                    <div key={entry.requestId} className="flex items-center justify-between gap-2 text-xs">
                      <div className="flex items-center gap-2 min-w-0">
                        {entry.projectId ? (
                          <span className="text-[10px] text-zinc-600 truncate">{entry.projectId}</span>
                        ) : (
                          <span className="text-[10px] text-zinc-600">—</span>
                        )}
                        <span className="text-zinc-500 shrink-0">
                          {new Date(entry.createdAt.replace(' ', 'T') + 'Z').toLocaleDateString(undefined, {
                            month: 'short',
                            day: 'numeric',
                          })}
                        </span>
                      </div>
                      <span className="text-zinc-200 font-medium shrink-0">
                        {formatMicros(Number(entry.amountUsdMicros))}
                      </span>
                    </div>
                  ))}
                </div>
                <p className="text-[10px] text-zinc-600">
                  Amount is the ticket's expected value charged by PymtHouse at generation time.
                </p>
              </>
            )}
          </div>
        </>
      )}

      {message && (
        <div className={`text-xs px-3 py-2 rounded-lg ${message.kind === 'ok' ? 'bg-green-500/10 text-green-400' : 'bg-red-500/10 text-red-400'}`}>
          {message.text}
        </div>
      )}
    </div>
  )
}
