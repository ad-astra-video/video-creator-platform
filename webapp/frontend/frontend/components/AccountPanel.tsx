import { useCallback, useEffect, useState } from 'react'
import { AlertCircle, CreditCard, RefreshCw, Wallet } from 'lucide-react'
import { useAppSettings } from '../contexts/AppSettingsContext'
import { ApiClient, type PlatformBalance, type PlatformStatus } from '../lib/api-client'

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
  const { settings } = useAppSettings()

  const [status, setStatus] = useState<PlatformStatus | null>(null)
  const [balance, setBalance] = useState<PlatformBalance | null>(null)
  const [loading, setLoading] = useState(false)
  const [busyTier, setBusyTier] = useState<number | null>(null)
  const [message, setMessage] = useState<{ kind: 'ok' | 'err'; text: string } | null>(null)

  const configured = Boolean(settings.hasPlatformBaseUrl) && Boolean(status?.configured)

  const load = useCallback(async () => {
    setLoading(true)
    const [s, b] = await Promise.all([
      ApiClient.getPlatformStatus(),
      ApiClient.getPlatformBalance(),
    ])
    if (s.ok) setStatus(s.data)
    if (b.ok) setBalance(b.data)
    setLoading(false)
  }, [])

  // Load once a platform server is configured (and whenever it changes, so a
  // just-saved URL immediately reflects balance).
  useEffect(() => {
    if (!settings.hasPlatformBaseUrl) return
    void load()
  }, [settings.hasPlatformBaseUrl, load])

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

  // Compact sidebar variant: just the essentials (balance + access), with a
  // hint to open Settings for the full account view.
  if (compact) {
    return (
      <div className="space-y-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Wallet className="h-4 w-4 text-amber-400" />
            <span className="text-xs font-semibold text-white uppercase tracking-wider">Account</span>
          </div>
          <button
            onClick={() => { setMessage(null); void load() }}
            disabled={loading}
            className="inline-flex items-center text-[10px] text-zinc-400 hover:text-white disabled:text-zinc-600"
            title="Refresh balance"
          >
            <RefreshCw className={`h-3 w-3 ${loading ? 'animate-spin' : ''}`} />
          </button>
        </div>

        {!configured ? (
          <p className="text-[11px] text-zinc-500 leading-relaxed">
            No platform connected. Balance and credits will appear here once a platform server is set in Settings.
          </p>
        ) : (
          <>
            {balance?.hasAccess === false && (
              <div className="bg-amber-500/10 border border-amber-500/30 rounded-lg px-2.5 py-1.5 text-[11px] text-amber-300 flex items-center gap-1.5">
                <AlertCircle className="h-3 w-3 flex-shrink-0" />
                Insufficient credits
              </div>
            )}
            <div className="flex items-end justify-between">
              <div>
                <p className="text-[10px] text-zinc-500">Balance</p>
                <p className="text-xl font-semibold text-white leading-tight">
                  {balance ? formatMicros(Number(balance.balanceUsdMicros)) : '—'}
                </p>
                {balance && Number(balance.pendingUsdMicros ?? 0) > 0 && (
                  <p className="text-[10px] text-amber-400/80 leading-tight">
                    {formatMicros(Number(balance.pendingUsdMicros))} in-flight (updating)
                  </p>
                )}
              </div>
              <div className="text-right">
                <p className="text-[10px] text-zinc-500">Remaining</p>
                <p className="text-sm font-medium text-zinc-300 leading-tight">
                  {balance ? formatMicros(Number(balance.remainingUsdMicros)) : '—'}
                </p>
              </div>
            </div>
          </>
        )}

        {message && (
          <p className={`text-[11px] px-2 py-1 rounded ${message.kind === 'ok' ? 'bg-green-500/10 text-green-400' : 'bg-red-500/10 text-red-400'}`}>
            {message.text}
          </p>
        )}
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

            <div className="grid grid-cols-2 gap-3 text-xs">
              <div>
                <p className="text-zinc-500">Balance</p>
                <p className="text-lg font-semibold text-white">{balance ? formatMicros(Number(balance.balanceUsdMicros)) : '—'}</p>
              </div>
              <div>
                <p className="text-zinc-500">Remaining</p>
                <p className="text-lg font-semibold text-zinc-300">{balance ? formatMicros(Number(balance.remainingUsdMicros)) : '—'}</p>
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
              {settings.hasPlatformBaseUrl && (
                <span className="text-zinc-600 truncate">· {settings.platformBaseUrl}</span>
              )}
            </div>
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
