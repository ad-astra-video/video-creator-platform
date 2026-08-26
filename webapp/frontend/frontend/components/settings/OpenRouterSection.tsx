import { useEffect, useState } from 'react'
import { AlertCircle, Check, Loader2 } from 'lucide-react'
import { useAppSettings } from '../../contexts/AppSettingsContext'
import { ApiClient } from '../../lib/api-client'
import { fetchVisionModels, type OpenRouterModel } from '../../lib/openrouter'

/**
 * OpenRouter API key + vision-model picker (API Keys tab).
 *
 * Saving a key fetches OpenRouter's live model list (vision, in OpenRouter's own
 * order) so the user can pick the exact model for prompt enhancement. "Free (auto)"
 * clears the explicit pick — the model is then resolved LIVE at each call to the
 * most popular free+vision model (no hardcoded constant; see resolveOpenRouterModel).
 */
export function OpenRouterSection() {
  const { settings, saveOpenRouterApiKey, setOpenRouterModel } = useAppSettings()
  const [input, setInput] = useState('')
  const [models, setModels] = useState<OpenRouterModel[] | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [customModel, setCustomModel] = useState('')

  const loadModels = async (key: string) => {
    setLoading(true)
    setError(null)
    try {
      setModels(await fetchVisionModels(key))
    } catch (e) {
      setModels(null)
      setError(e instanceof Error ? e.message : 'Failed to load models')
    } finally {
      setLoading(false)
    }
  }

  // When a key is already configured, fetch the raw key and populate the model list.
  useEffect(() => {
    if (!settings.hasOpenRouterApiKey) return
    let cancelled = false
    void ApiClient.getOpenRouterApiKey().then(async (res) => {
      if (cancelled || !res.ok) return
      const key = (res.data as { openrouterApiKey?: string }).openrouterApiKey
      if (key) await loadModels(key)
    })
    return () => { cancelled = true }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const save = async () => {
    const key = input.trim()
    if (!key) return
    setLoading(true)
    setError(null)
    try {
      await saveOpenRouterApiKey(key)
      setInput('')
      await loadModels(key)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Failed to save key')
      setLoading(false)
    }
  }

  const pick = (id: string) => void setOpenRouterModel(id)
  const freeAuto = !settings.openrouterModel

  return (
    <div className="space-y-4 pt-4 border-t border-zinc-800">
      <div className="flex items-center gap-2">
        <svg className="h-4 w-4 text-zinc-400" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
          <circle cx="12" cy="12" r="10" />
          <path d="M2 12h20M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z" />
        </svg>
        <h3 className="text-sm font-semibold text-white">OpenRouter (LLM)</h3>
      </div>

      <p className="text-xs text-zinc-500 leading-relaxed">
        Add an OpenRouter API key to run prompt enhancement, layer-suggest and gap-fill with a
        hosted vision model, called directly from your browser. Without a key these features use your own runner.
      </p>

      <div className="bg-zinc-800/50 rounded-lg p-4 space-y-3">
        <div className="flex gap-2">
          <input
            type="password"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder={settings.hasOpenRouterApiKey ? 'Enter new key to replace...' : 'Enter your OpenRouter API key...'}
            onKeyDown={(e) => e.stopPropagation()}
            className="flex-1 px-3 py-2 bg-zinc-800 border border-zinc-700 rounded-lg text-sm text-white placeholder-zinc-500 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          />
          <button
            onClick={() => void save()}
            disabled={!input.trim() || loading}
            className="px-3 py-2 bg-blue-600 text-white text-sm rounded-lg hover:bg-blue-500 disabled:bg-zinc-700 disabled:text-zinc-500 disabled:cursor-not-allowed transition-colors whitespace-nowrap inline-flex items-center gap-1.5"
          >
            {loading ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : null}
            Save Key
          </button>
        </div>

        <div className="flex items-center justify-between">
          <div className={`text-xs px-2 py-1 rounded inline-flex items-center gap-1.5 ${
            settings.hasOpenRouterApiKey ? 'bg-green-500/10 text-green-400' : 'bg-amber-500/10 text-amber-400'
          }`}>
            {settings.hasOpenRouterApiKey ? (
              <><Check className="h-3 w-3" /> Key configured</>
            ) : (
              <><AlertCircle className="h-3 w-3" /> No key — runner fallback active</>
            )}
          </div>
        </div>

        <div className="flex items-center gap-2 text-xs">
          <a
            href="https://openrouter.ai/keys"
            target="_blank"
            rel="noopener noreferrer"
            className="text-blue-400 hover:text-blue-300 transition-colors underline underline-offset-2"
            onClick={(e) => e.stopPropagation()}
          >
            Get OpenRouter API key →
          </a>
        </div>
      </div>

      {settings.hasOpenRouterApiKey && (
        <div className="bg-zinc-800/50 rounded-lg p-4 space-y-3">
          <div className="flex items-center gap-2">
            <span className="text-xs font-semibold text-zinc-200">Vision model for prompt enhancement</span>
          </div>

          {loading && <p className="text-xs text-zinc-400 inline-flex items-center gap-1.5"><Loader2 className="h-3 w-3 animate-spin" /> Loading models…</p>}
          {error && <p className="text-xs text-amber-400">{error}</p>}

          <div className="flex gap-2">
            <select
              value={freeAuto ? '' : settings.openrouterModel}
              onChange={(e) => pick(e.target.value)}
              className="flex-1 px-3 py-2 bg-zinc-800 border border-zinc-700 rounded-lg text-sm text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              <option value="">Free (auto) — most popular free vision model</option>
              {models?.map((m) => (
                <option key={m.id} value={m.id}>
                  {m.name} ({m.id}){m.free ? ' — Free' : ''}
                </option>
              ))}
            </select>
          </div>

          <div className="flex gap-2 items-center">
            <input
              value={customModel}
              onChange={(e) => setCustomModel(e.target.value)}
              placeholder="Or type a custom model id (e.g. openai/gpt-4o-mini)"
              onKeyDown={(e) => e.stopPropagation()}
              className="flex-1 px-3 py-2 bg-zinc-800 border border-zinc-700 rounded-lg text-sm text-white placeholder-zinc-500 focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
            <button
              onClick={() => { const v = customModel.trim(); if (v) pick(v) }}
              disabled={!customModel.trim()}
              className="px-3 py-2 bg-zinc-700 text-white text-sm rounded-lg hover:bg-zinc-600 disabled:opacity-40 transition-colors whitespace-nowrap"
            >
              Use
            </button>
          </div>
        </div>
      )}
    </div>
  )
}
