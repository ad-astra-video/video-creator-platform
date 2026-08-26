import { useEffect, useState, type ReactNode } from 'react'
import { AlertCircle, Check, KeyRound, Loader2, RotateCcw, Save } from 'lucide-react'
import { useAppSettings } from '../../contexts/AppSettingsContext'
import {
  DEFAULT_T2V_SYSTEM_PROMPT,
  DEFAULT_I2V_SYSTEM_PROMPT,
  DEFAULT_EXTEND_SYSTEM_PROMPT,
  DEFAULT_RETAKE_SYSTEM_PROMPT,
  LAYER_SUGGEST_RUBRIC,
  GAP_FILL_RUBRIC,
  type CustomPrompts,
} from '../../lib/llm-messages'

const FEATURES: { key: keyof CustomPrompts; title: string; desc: string }[] = [
  { key: 'enhancerT2V', title: 'Text-to-video enhancement', desc: 'Prompt for turning text into a video.' },
  { key: 'enhancerI2V', title: 'Image-to-video enhancement', desc: 'Prompt when animating a source image.' },
  { key: 'enhancerExtend', title: 'Video extension', desc: 'Prompt when extending an existing clip.' },
  { key: 'enhancerRetake', title: 'Video retake', desc: 'Prompt when re-rendering a selected segment.' },
  { key: 'layerSuggestRubric', title: 'Layer-count suggestion', desc: 'Rubric for choosing how many decomposition layers an image needs.' },
  { key: 'gapFillRubric', title: 'Timeline gap-fill', desc: 'Rubric for writing the clip that fills a timeline gap.' },
]

/** The built-in default prompt used for each feature when you haven't customized it. */
const DEFAULTS: Record<string, string> = {
  enhancerT2V: DEFAULT_T2V_SYSTEM_PROMPT,
  enhancerI2V: DEFAULT_I2V_SYSTEM_PROMPT,
  enhancerExtend: DEFAULT_EXTEND_SYSTEM_PROMPT,
  enhancerRetake: DEFAULT_RETAKE_SYSTEM_PROMPT,
  layerSuggestRubric: LAYER_SUGGEST_RUBRIC,
  gapFillRubric: GAP_FILL_RUBRIC,
}

/**
 * Prompt Builder tab — see AND edit the exact system prompt every LLM feature uses.
 *
 * Each feature shows the prompt it will actually send: the built-in default (read-only)
 * or your custom text (editable). A two-option dropdown ("Default" / "Custom") picks
 * which. Written prompts are used by the browser's OpenRouter DIRECT path AND
 * (byte-for-byte) by the runner's gemma worker, so a custom prompt takes effect
 * everywhere at once.
 *
 * Storage is two-mode (user decision):
 *  - no encryption key  -> prompts saved as PLAINTEXT in D1 + browser localStorage
 *  - an encryption key  -> prompts ENVELOPE-encrypted (AES-GCM); the key lives ONLY in
 *    the browser (never sent to the server); the server stores ciphertext it can't read.
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
    for (const f of FEATURES) init[f.key] = settings.customPrompts?.[f.key] ?? ''
    setDrafts(init)
  }, [settings.customPrompts])

  const customized = (key: string) => (drafts[key] ?? '').trim().length > 0
  const activePrompt = (key: string) => (customized(key) ? drafts[key] : DEFAULTS[key])

  const setDraft = (key: string, value: string) => setDrafts((d) => ({ ...d, [key]: value }))

  const switchMode = (key: string, mode: 'default' | 'custom') => {
    if (mode === 'custom') {
      // Seed with the existing custom text, else the default as a starting point,
      // so choosing Custom always gives you a non-empty prompt to edit.
      setDraft(key, customized(key) ? drafts[key]! : DEFAULTS[key])
    } else {
      setDraft(key, '') // clear -> uses the built-in default
    }
  }

  const persist = async () => {
    setSaving(true)
    setError(null)
    try {
      const plain: CustomPrompts = {}
      for (const f of FEATURES) if (customized(f.key)) plain[f.key] = drafts[f.key]!
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
          const custom = customized(f.key)
          const prompt = activePrompt(f.key)
          return (
            <div key={f.key} className="bg-zinc-800/50 rounded-lg p-4 space-y-3">
              <div className="flex items-start justify-between gap-3">
                <div className="flex-1">
                  <div className="flex items-center gap-2 mb-1">
                    <span className="text-sm font-semibold text-white">{f.title}</span>
                    <span className={`text-[10px] px-1.5 py-0.5 rounded font-medium uppercase tracking-wide ${
                      custom ? 'bg-blue-500/10 text-blue-400' : 'bg-zinc-800 text-zinc-500'
                    }`}>
                      {custom ? 'Custom' : 'Default'}
                    </span>
                  </div>
                  <p className="text-xs text-zinc-500 leading-relaxed">{f.desc}</p>
                </div>
                <select
                  value={custom ? 'custom' : 'default'}
                  onChange={(e) => switchMode(f.key, e.target.value as 'default' | 'custom')}
                  className="flex-shrink-0 px-2.5 py-1.5 bg-zinc-900 border border-zinc-700 rounded-lg text-sm text-white focus:outline-none focus:ring-2 focus:ring-blue-500 cursor-pointer"
                  title={custom ? 'Uses your custom prompt. Switch to Default to use the built-in prompt.' : 'Uses the built-in prompt. Switch to Custom to write your own.'}
                >
                  <option value="default">Default</option>
                  <option value="custom">Custom</option>
                </select>
              </div>

              {/* The exact prompt that will be used — always visible.
                  Default: read-only. Custom: editable. */}
              {custom ? (
                <textarea
                  value={drafts[f.key] ?? ''}
                  onChange={(e) => setDraft(f.key, e.target.value)}
                  rows={4}
                  className="w-full px-3 py-2 bg-zinc-900 border border-zinc-700 rounded-lg text-sm text-white placeholder-zinc-600 focus:outline-none focus:ring-2 focus:ring-blue-500 resize-y"
                  aria-label={`Custom ${f.title} prompt`}
                />
              ) : (
                <div className="relative">
                  <pre className="max-h-24 overflow-y-auto whitespace-pre-wrap px-3 py-2 bg-zinc-900/60 border border-zinc-800 rounded-lg text-xs text-zinc-400 leading-relaxed">
                    {prompt}
                  </pre>
                  <span className="absolute top-1.5 right-2 text-[10px] text-zinc-600 uppercase tracking-wide">
                    Built-in default
                  </span>
                </div>
              )}

              {custom && (
                <div className="flex justify-end">
                  <button
                    type="button"
                    onClick={() => setDraft(f.key, '')}
                    className="text-xs text-zinc-400 hover:text-zinc-200 inline-flex items-center gap-1 transition-colors"
                  >
                    <RotateCcw className="h-3 w-3" /> Reset to default
                  </button>
                </div>
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
