import { useEffect, useState } from 'react'
import { Loader2, Wallet } from 'lucide-react'
import { ApiClient, type PlatformBalance } from '../lib/api-client'

function fmtMicros(micros: number | undefined): string {
  return micros == null ? '—' : `$${(micros / 1_000_000).toFixed(2)}`
}

/** Small header pill showing the live platform-credit balance. Click opens the
 *  Credits modal. Refetches on mount, when `refreshKey` changes (e.g. after the
 *  modal closes), and on a light 60s poll so the number stays live. */
export function GlobalBalanceButton({
  onClick,
  refreshKey,
}: {
  onClick: () => void
  refreshKey: number
}) {
  const [balance, setBalance] = useState<PlatformBalance | null>(null)
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    let cancelled = false
    const load = async () => {
      setLoading(true)
      const result = await ApiClient.getPlatformBalance()
      if (cancelled) return
      if (result.ok) setBalance(result.data)
      setLoading(false)
    }
    void load()
    const t = setInterval(() => void load(), 60_000)
    return () => {
      cancelled = true
      clearInterval(t)
    }
  }, [refreshKey])

  const label = balance?.configured ? fmtMicros(balance.balanceUsdMicros) : 'Credits'

  return (
    <button
      onClick={onClick}
      title="Platform credits — click for balance & top up"
      className="inline-flex h-8 items-center gap-1.5 rounded-md border border-zinc-800 px-2 text-xs font-medium text-zinc-300 transition-colors hover:bg-zinc-800 hover:text-white"
    >
      <Wallet className="h-4 w-4 text-amber-400" />
      {loading ? <Loader2 className="h-3 w-3 animate-spin" /> : <span>{label}</span>}
    </button>
  )
}
