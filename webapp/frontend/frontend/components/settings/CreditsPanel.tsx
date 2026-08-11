import { useCallback, useEffect, useState } from 'react'
import { AlertCircle, CreditCard, KeyRound, RefreshCw, Wallet } from 'lucide-react'
import { useAppSettings } from '../../contexts/AppSettingsContext'
import { ApiClient, type PlatformBalance, type PlatformStatus } from '../../lib/api-client'

const TIERS: Array<{ label: string; cents: number; charge: string }> = [
  { label: '$10', cents: 1000, charge: 'pays $11' },
  { label: '$25', cents: 2500, charge: 'pays $26.50' },
  { label: '$50', cents: 5000, charge: 'pays $53' },
  { label: '$100', cents: 10000, charge: 'pays $105' },
]

function formatMicros(micros: number): string {
  return `$${(micros / 1_000_000).toFixed(2)}`
}

export function CreditsPanel() {
  const { settings, updateSettings, refreshSettings } = useAppSettings()

  const [baseUrlInput, setBaseUrlInput] = useState('')
  const [status, setStatus] = useState<PlatformStatus | null>(null)
  const [balance, setBalance] = useState<PlatformBalance | null>(null)
  const [loading, setLoading] = useState(false)
  const [busyTier, setBusyTier] = useState<number | null>(null)
  const [message, setMessage] = useState<{ kind: 'ok' | 'err'; text: string } | null>(null)

  // Recovery inputs
  const [emailInput, setEmailInput] = useState('')
  const [codeInput, setCodeInput] = useState('')
  const [confirming, setConfirming] = useState(false)

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

  useEffect(() => {
    if (!settings.hasPlatformBaseUrl) return
    void load()
  }, [settings.hasPlatformBaseUrl, load])

  const saveBaseUrl = async () => {
    const trimmed = baseUrlInput.trim()
    if (!trimmed) return
    updateSettings({ platformBaseUrl: trimmed })
    try {
      await refreshSettings()
      setMessage({ kind: 'ok', text: 'Platform server saved.' })
      await load()
    } catch (err) {
      setMessage({ kind: 'err', text: errorText(err) })
    } finally {
      setBaseUrlInput('')
    }
  }

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
    setMessage({ kind: 'ok', text: 'Opening secure checkout in your browser. Click Refresh when you’re done.' })
  }

  const linkEmail = async () => {
    if (!emailInput.trim()) return
    const result = await ApiClient.linkPlatformEmail(emailInput.trim())
    if (!result.ok) {
      setMessage({ kind: 'err', text: result.error.message })
      return
    }
    setMessage({ kind: 'ok', text: 'Recovery email linked. You can now recover a lost key using it.' })
    setEmailInput('')
  }

  const requestRecovery = async () => {
    if (!emailInput.trim()) return
    const result = await ApiClient.requestPlatformRecovery(emailInput.trim())
    if (!result.ok) {
      setMessage({ kind: 'err', text: result.error.message })
      return
    }
    setMessage({ kind: 'ok', text: 'Recovery code sent to your email.' })
  }

  const confirmRecovery = async () => {
    if (!emailInput.trim() || !codeInput.trim()) return
    setConfirming(true)
    const result = await ApiClient.confirmPlatformRecovery(emailInput.trim(), codeInput.trim())
    setConfirming(false)
    if (!result.ok) {
      setMessage({ kind: 'err', text: result.error.message })
      return
    }
    setMessage({ kind: 'ok', text: 'Recovery complete — your key was rotated. You’re signed back in.' })
    setCodeInput('')
    await refreshSettings()
  }

  const configured = Boolean(settings.hasPlatformBaseUrl) && Boolean(status?.configured)

  return (
    <div className="space-y-4 pt-4 border-t border-zinc-800">
      <div className="flex items-center gap-2">
        <Wallet className="h-4 w-4 text-amber-400" />
        <h3 className="text-sm font-semibold text-white">Platform Credits</h3>
        <span className="text-[10px] px-1.5 py-0.5 rounded bg-amber-500/10 text-amber-400">Credit-gated remote generation</span>
      </div>

      <p className="text-xs text-zinc-500 leading-relaxed">
        Remote generation is credit-gated through a platform backend. Set the platform server URL, and add credits
        to keep remote jobs running. Inference is charged against your balance (pass-through); a small platform fee is
        added at checkout.
      </p>

      {/* Platform server URL */}
      <div className="bg-zinc-800/50 rounded-lg p-4 space-y-3">
        <div className="space-y-2">
          <label className="block text-xs text-zinc-300 font-medium">Platform server URL</label>
          <div className="flex gap-2">
            <input
              type="url"
              value={baseUrlInput}
              onChange={(e) => setBaseUrlInput(e.target.value)}
              placeholder="https://your-platform.workers.dev"
              className="flex-1 px-3 py-2 bg-zinc-800 border border-zinc-700 rounded-lg text-sm text-white placeholder-zinc-500 focus:outline-none focus:ring-2 focus:ring-amber-500"
            />
            <button
              onClick={saveBaseUrl}
              disabled={!baseUrlInput.trim() || loading}
              className="px-3 py-2 bg-amber-600 text-white text-sm rounded-lg hover:bg-amber-500 disabled:bg-zinc-700 disabled:text-zinc-500 disabled:cursor-not-allowed transition-colors whitespace-nowrap"
            >
              Save
            </button>
          </div>
          {settings.hasPlatformBaseUrl && (
            <p className="text-[11px] text-zinc-500">
              Saved: <span className="text-zinc-300">{settings.platformBaseUrl}</span>
              {status?.userId ? <span className="ml-2 text-zinc-500">ID: <span className="font-mono text-zinc-400">{status.userId.slice(0, 8)}…</span></span> : null}
            </p>
          )}
        </div>

        {!configured ? (
          <div className="text-xs text-zinc-500 bg-zinc-900/60 rounded-lg px-3 py-2 inline-flex items-center gap-1.5">
            <AlertCircle className="h-3 w-3" /> Platform not configured — remote generation isn’t credit-gated yet.
          </div>
        ) : null}
      </div>

      {configured && (
        <>
          {/* Balance */}
          <div className="bg-zinc-800/50 rounded-lg p-4 space-y-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <CreditCard className="h-4 w-4 text-emerald-400" />
                <span className="text-sm font-medium text-white">Credits</span>
              </div>
              <button
                onClick={load}
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
                <p className="text-zinc-500">Consumed</p>
                <p className="text-lg font-semibold text-zinc-300">{balance ? formatMicros(Number(balance.consumedUsdMicros)) : '—'}</p>
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

          {/* Recovery */}
          <div className="bg-zinc-800/50 rounded-lg p-4 space-y-3">
            <div className="flex items-center gap-2">
              <KeyRound className="h-4 w-4 text-blue-400" />
              <span className="text-sm font-medium text-white">Recovery</span>
            </div>
            <p className="text-xs text-zinc-500 leading-relaxed">
              Link an email so you can recover a lost API key. Request a code, then confirm it to rotate to a fresh key.
            </p>
            <div className="flex gap-2">
              <input
                type="email"
                value={emailInput}
                onChange={(e) => setEmailInput(e.target.value)}
                placeholder="you@example.com"
                className="flex-1 px-3 py-2 bg-zinc-800 border border-zinc-700 rounded-lg text-sm text-white placeholder-zinc-500 focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
              <button
                onClick={linkEmail}
                disabled={!emailInput.trim()}
                className="px-3 py-2 bg-blue-600 text-white text-sm rounded-lg hover:bg-blue-500 disabled:bg-zinc-700 disabled:text-zinc-500 transition-colors whitespace-nowrap"
              >
                Link
              </button>
              <button
                onClick={requestRecovery}
                disabled={!emailInput.trim()}
                className="px-3 py-2 bg-zinc-700 text-white text-sm rounded-lg hover:bg-zinc-600 disabled:bg-zinc-800 disabled:text-zinc-600 transition-colors whitespace-nowrap"
              >
                Send code
              </button>
            </div>
            <div className="flex gap-2">
              <input
                type="text"
                value={codeInput}
                onChange={(e) => setCodeInput(e.target.value)}
                placeholder="Recovery code"
                className="flex-1 px-3 py-2 bg-zinc-800 border border-zinc-700 rounded-lg text-sm text-white placeholder-zinc-500 focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
              <button
                onClick={confirmRecovery}
                disabled={confirming || !codeInput.trim()}
                className="px-3 py-2 bg-blue-600 text-white text-sm rounded-lg hover:bg-blue-500 disabled:bg-zinc-700 disabled:text-zinc-500 transition-colors whitespace-nowrap"
              >
                {confirming ? 'Rotating…' : 'Confirm'}
              </button>
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

function errorText(err: unknown): string {
  if (err instanceof Error) return err.message
  return String(err)
}
