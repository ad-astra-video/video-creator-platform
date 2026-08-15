import { useEffect, useState } from 'react'
import { ApiClient } from '../lib/api-client'
import { resolveRunner } from '../lib/direct-transport'
import { useAppSettings } from '../contexts/AppSettingsContext'

export type EnhanceProvider = 'local' | 'api' | 'runner'

interface UsePromptEnhancerProviderResult {
  // Local requires the Gemma text-encoder checkpoint to be downloaded AND local generation to
  // actually be usable this run (e.g. not memory-constrained into API-only mode); API requires a
  // stored Gemini key (presence only — an invalid key still surfaces via the enhance call's own
  // error response, so no separate validity check is worth the extra backend round-trip).
  hasLocalTextEncoder: boolean
  hasGeminiApiKey: boolean
  // A Livepeer runner that advertises prompt-enhance is resolvable. This is the web-app path:
  // the runner's own gemma-worker (Gemma 4) does the enhancement, falling back to the LTX
  // pipeline's Gemma text encoder when no gemma-worker is present. No Gemini key or local
  // checkpoint is needed for this to work.
  hasEnhanceRunner: boolean
  // Whether at least one provider is usable.
  isAvailable: boolean
  // The provider Enhance will actually use: the persisted preference when it's currently
  // available, otherwise whichever single option works right now. A provider that's temporarily
  // unavailable (e.g. local on a memory-constrained run) falls back silently — it does NOT
  // overwrite the persisted preference, which only an explicit setProviderPreference call changes.
  provider: EnhanceProvider
  // Only true (and thus only worth showing a picker for) when the user actually has a choice
  // between two currently-usable providers.
  canToggleProvider: boolean
  setProviderPreference: (provider: EnhanceProvider) => void
}

// Single source of truth for which prompt-enhancer provider (local Gemma text encoder, Gemini's
// hosted API, or a Livepeer runner's gemma/ltx text encoder) is available and which one Enhance
// should use. `enabled` gates the lookups so they only fire once the enhancer could plausibly be
// shown for the current mode.
export function usePromptEnhancerProvider(enabled: boolean): UsePromptEnhancerProviderResult {
  const {
    settings: {
      hasGeminiApiKey,
      promptEnhancerProviderPreference,
      livepeerDiscoveryUrl,
    },
    updateSettings,
    forceApiGenerations,
  } = useAppSettings()

  const [isTextEncoderDownloaded, setIsTextEncoderDownloaded] = useState(false)
  const [hasEnhanceRunner, setHasEnhanceRunner] = useState(false)
  useEffect(() => {
    if (!enabled) return
    let cancelled = false
    void ApiClient.getTextEncoderRecommendation().then((result) => {
      if (!cancelled) setIsTextEncoderDownloaded(result.ok && result.data.cp_to_download === null)
    })
    return () => { cancelled = true }
  }, [enabled])

  // A capable runner only helps when a Livepeer Discovery URL is configured — that's the only
  // route where the browser can reach a runner's /prompt-enhance at all. When configured, probe
  // it (the same resolveRunner the enhance call uses) so the button is enabled when a gemma/ltx
  // text-encoder runner is actually available, without requiring a Gemini key.
  useEffect(() => {
    if (!enabled) return
    if (!livepeerDiscoveryUrl?.trim()) {
      setHasEnhanceRunner(false)
      return
    }
    let cancelled = false
    setHasEnhanceRunner(false)
    resolveRunner(['prompt-enhance'])
      .then((runner) => { if (!cancelled) setHasEnhanceRunner(Boolean(runner)) })
      .catch(() => { if (!cancelled) setHasEnhanceRunner(false) })
    return () => { cancelled = true }
  }, [enabled, livepeerDiscoveryUrl])

  // Downloaded isn't enough on its own — forceApiGenerations is the pure "insufficient memory
  // for local models this run" signal (deliberately NOT shouldVideoGenerateWithLtxApi, which
  // also folds in the user's own preference to use the LTX API for VIDEO specifically — that's
  // unrelated to whether the much smaller Gemma text encoder can run locally right now).
  const hasLocalTextEncoder = isTextEncoderDownloaded && !forceApiGenerations
  const canToggleProvider = hasLocalTextEncoder && hasGeminiApiKey

  // Default to local (the first available option) when the user hasn't made an explicit choice,
  // or when their choice isn't currently usable — a silent, non-persisted fallback. A resolvable
  // prompt-enhance runner wins over Gemini so the web label reads plain "Enhance" (not "Enhance
  // (API)") and the request is transparently routed through the runner.
  const provider: EnhanceProvider =
    promptEnhancerProviderPreference === 'api' && hasGeminiApiKey ? 'api'
    : promptEnhancerProviderPreference === 'local' && hasLocalTextEncoder ? 'local'
    : hasLocalTextEncoder ? 'local'
    : hasEnhanceRunner ? 'runner'
    : hasGeminiApiKey ? 'api'
    : 'api'

  const setProviderPreference = (next: EnhanceProvider) => {
    // 'runner' is a transient fallback, never a persisted choice — the toggle
    // dropdown only offers local/api, so a runner-backed enhance stays silent.
    if (next === 'runner') return
    updateSettings({ promptEnhancerProviderPreference: next })
  }

  return {
    hasLocalTextEncoder,
    hasGeminiApiKey,
    hasEnhanceRunner,
    isAvailable: hasLocalTextEncoder || hasGeminiApiKey || hasEnhanceRunner,
    provider,
    canToggleProvider,
    setProviderPreference,
  }
}
