import { useEffect, useState, type ReactNode } from 'react'
import { AlertCircle, Check, KeyRound, Loader2, RotateCcw, Save } from 'lucide-react'
import { useAppSettings } from '../../contexts/AppSettingsContext'
import type { CustomPrompts } from '../../lib/llm-messages'

const FEATURES: { key: keyof CustomPrompts; title: string; desc: string }[] = [
  { key: 'enhancerT2V', title: 'Text-to-video enhancement', desc: 'Prompt for turning text into a video.' },
  { key: 'enhancerI2V', title: 'Image-to-video enhancement', desc: 'Prompt when animating a source image.' },
  { key: 'enhancerExtend', title: 'Video extension', desc: 'Prompt when extending an existing clip.' },
  { key: 'enhancerRetake', title: 'Video retake', desc: 'Prompt when re-rendering a selected segment.' },
  { key: 'layerSuggestRubric', title: 'Layer-count suggestion', desc: 'Rubric for choosing how many decomposition layers an image needs.' },
  { key: 'gapFillRubric', title: 'Timeline gap-fill', desc: 'Rubric for writing the clip that fills a timeline gap.' },
]

/**
 * Prompt Builder tab — write/edit the exact system prompts every LLM feature uses.
 *
 * Each field is optional: a blank field falls back to the built-in default prompt.
 * Written prompts are used by the browser's OpenRouter DIRECT path AND (byte-for-byte)
 * by the runner's gemma worker as pure executor, so a custom prompt takes effect
 * everywhere at once.
 *
 * Storage is two-mode (user decision):
 *  - no encryption key  -> prompts saved as PLAINTEXT in D1 + browser localStorage
 *  - an encryption key  -> prompts ENVELOPE-encrypted (AES-GCM); the key lives ONLY in
 *    the browser (never sent to the server); server stores ciphertext it cannot read.
 */
export function PromptBuilderTab() {
  const { settings, saveCustomPrompts } = useAppSettings()
  const [drafts, setDrafts] = useState<Record<string, string>>({})
  const [passphrase, setPassphrase] = useState('')
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [savedAt, setSavedAt] = useState<number | null>(null)

  useEffect(() => {
    const init: Record<string, string> = {}
    for (const f of FEATURES) {
      init[f.key] = settings.customPrompts?.[f.key] ?? ''
    }
    setDrafts(init)
  }, [settings.customPrompts])

  const setDraft = (key: string, value: string) => {
    setDrafts((d) => ({ ...d, [key]: value }))
  }

  const persist = async () => {
    setSaving(true)
    setError(null)
    try {
      const plain: CustomPrompts = {}
      for (const f of FEATURES) if (drafts[f.key]) plain[f.key] = drafts[f.key]
      await saveCustomPrompts(plain, passphrase ? { passphrase } : undefined)
      setPassphrase('')
      setSavedAt(Date.now())
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to save prompts')
    } finally {
      setSaving(false)
    }
  }

  const revert = () => {
    const init: Record<string, string> = {}
    for (const f of FEATURES) init[f.key] = settings.customPrompts?.[f.key] ?? ''
    setDrafts(init)
  }

  const encrypted = settings.hasPromptEncryptionKey

  return (
    <div className="space-y-6">
      {/* Encryption card */}
      <div className="bg-zinc-800/50 rounded-lg p-4 space-y-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <KeyRound className="h-4 w-4 text-blue-400" />
            <span className="text-sm font-medium text-white">Protect these prompts</span>
          </div>
          <div className={`text-xs px-2 py-1 rounded inline-flex items-center gap-1.5 ${
            encrypted ? 'bg-green-500/10 text-green-400' : 'bg-zinc-800 text-zinc-400'
          }`}>
            {encrypted ? <><Check className="h-3 w-3" /> Encrypted</> : <><AlertCircle className="h-3 w-3" /> Plain text</>}
          </div>
        </div>
        <p className="text-xs text-zinc-500 leading-relaxed">
          {encrypted
            ? 'Your prompts are encrypted before being stored — they can only be read from this browser with your key.'
            : 'Prompts are saved in plain text. Set an encryption key to store them encrypted (readable only from this browser).'}
        </p>
        <div className="flex gap-2">
          <input
            type="password"
            value={passphrase}
            onChange={(e) => setPassphrase(e.target.value)}
            placeholder={encrypted ? 'Set a new encryption key (or leave empty to keep current)' : 'Optional encryption key…'}
            onKeyDown={(e) => e.stopPropagation()}
            className="flex-1 px-3 py-2 bg-zinc-800 border border-zinc-700 rounded-lg text-sm text-white placeholder-zinc-500 focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
          <button
            onClick={() => { setPassphrase(''); void persist() }}
            disabled={saving}
            className="px-3 py-2 bg-zinc-700 text-white text-sm rounded-lg hover:bg-zinc-600 disabled:opacity-40 transition-colors whitespace-nowrap"
            title="Remove encryption — store as plain text"
          >
            {encrypted ? 'Remove protection' : 'No key'}
          </button>
        </div>
      </div>

      {/* Per-feature prompt cards */}
      <div className="space-y-3">
        {FEATURES.map((f) => {
          const value = drafts[f.key] ?? ''
          const customized = value.trim().length > 0
          return (
            <div key={f.key} className="bg-zinc-800/50 rounded-lg p-4 space-y-2">
              <div className="flex items-start justify-between gap-3">
                <div className="flex-1">
                  <div className="flex items-center gap-2 mb-1">
                    <span className="text-sm font-semibold text-white">{f.title}</span>
                    {customized && (
                      <span className="text-[10px] px-1.5 py-0.5 rounded bg-blue-500/10 text-blue-400 font-medium uppercase tracking-wide">Custom</span>
                    )}
                  </div>
                  <p className="text-xs text-zinc-500 leading-relaxed">{f.desc}</p>
                </div>
                <button
                  type="button"
                  onClick={() => setDraft(f.key, customized ? '' : value || '')}
                  className={`relative inline-flex h-5 w-9 flex-shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors ${
                    customized ? 'bg-orange-500' : 'bg-zinc-700'
                  }`}
                >
                  <span className={`pointer-events-none inline-block h-4 w-4 transform rounded-full bg-white shadow transition duration-200 ${customized ? 'translate-x-4' : 'translate-x-0'}`} />
                </button>
              </div>
              {customized && (
                <textarea
                  value={value}
                  onChange={(e) => setDraft(f.key, e.target.value)}
                  rows={4}
                  className="w-full px-3 py-2 bg-zinc-900 border border-zinc-700 rounded-lg text-sm text-white placeholder-zinc-600 focus:outline-none focus:ring-2 focus:ring-blue-500 resize-y"
                />
              )}
            </div>
          )
        })}
      </div>

      {error && <p className="text-xs text-amber-400">{error}</p>}

      <div className="flex items-center justify-end gap-2 pt-2">
        <Button outline onClick={revert} className="inline-flex items-center gap-1.5">
          <RotateCcw className="h-3.5 w-3.5" /> Revert
        </Button>
        <Button onClick={() => void persist()} disabled={saving} className="inline-flex items-center gap-1.5">
          {saving ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Save className="h-3.5 w-3.5" />}
          Save prompt settings
        </Button>
      </div>
      {savedAt && <p className="text-xs text-zinc-500 text-right">Saved {new Date(savedAt).toLocaleTimeString()}</p>}
    </div>
  )
}

/** Minimal local button so this file stays dependency-free. */
function Button({ children, onClick, disabled, className, outline }: {
  children: ReactNode
  onClick?: () => void
  disabled?: boolean
  className?: string
  outline?: boolean
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      className={`px-3 py-2 text-sm rounded-lg transition-colors whitespace-nowrap disabled:opacity-40 disabled:cursor-not-allowed ${
        outline ? 'bg-zinc-800 border border-zinc-700 text-white hover:bg-zinc-700' : 'bg-blue-600 text-white hover:bg-blue-500'
      } ${className ?? ''}`}
    >
      {children}
    </button>
  )
}
