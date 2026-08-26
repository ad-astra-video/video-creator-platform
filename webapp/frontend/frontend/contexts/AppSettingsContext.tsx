import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from 'react'
import { resetBackendCredentials } from '../lib/backend'
import { ApiClient, type ApiSuccessOf } from '../lib/api-client'
import { isWebPlatform, setRunnerDiscoveryConfig } from '../lib/livepeer-discovery'
const CUSTOM_PROMPT_VAULT_KEY = 'custom-prompt-passphrase'
import { encryptPromptsKeyed, decryptPromptsKeyed, newSalt } from '../lib/prompt-crypto'
import type { CustomPrompts } from '../lib/llm-messages'

export interface AppSettings {
  useTorchCompile: boolean
  diffusionStageCacheEnabled: boolean
  hasLtxApiKey: boolean
  userPrefersLtxApiVideoGenerations: boolean
  hasFalApiKey: boolean
  userPrefersFalApiImageGenerations: boolean
  hasGeminiApiKey: boolean
  hasOpenRouterApiKey: boolean
  openrouterModel: string
  hasPromptEncryptionKey: boolean
  customPrompts: CustomPrompts | null
  customPromptsEnc: string | null
  customPromptsKeyEnc: string | null
  customPromptsKdfSalt: string | null
  useLocalTextEncoder: boolean
  promptCacheSize: number
  // The user's explicit prompt-enhancer provider choice, persisted so it survives restarts.
  // null means no active choice yet — the enhancer defaults to whichever provider is available
  // without writing that default back here; only an explicit pick (never an automatic fallback
  // when the preferred provider is temporarily unavailable) sets this.
  promptEnhancerProviderPreference: 'local' | 'api' | null
  seedLocked: boolean
  lockedSeed: number
  modelsDir: string
  // Remote inference via Livepeer
  remoteInferenceEnabled: boolean
  livepeerVideoEnabled: boolean
  livepeerImageEnabled: boolean
  livepeerTextEncodingEnabled: boolean
  livepeerDiscoveryUrl: string
  hasLivepeerDiscoveryUrl: boolean
  hasLivepeerApiKey: boolean
  livepeerSelectedRunnerId: string
  livepeerExcludedRunnerIds: string[]
  // Remote "platform" credits API (per-install identity + credit top-up)
  platformBaseUrl: string
  hasPlatformBaseUrl: boolean
  platformUserId: string
  hasPlatformApiKey: boolean
  platformRecoveryEmail: string
}

export const DEFAULT_APP_SETTINGS: AppSettings = {
  useTorchCompile: false,
  diffusionStageCacheEnabled: false,
  hasLtxApiKey: false,
  userPrefersLtxApiVideoGenerations: false,
  hasFalApiKey: false,
  userPrefersFalApiImageGenerations: false,
  hasGeminiApiKey: false,
  hasOpenRouterApiKey: false,
  openrouterModel: '',
  hasPromptEncryptionKey: false,
  customPrompts: null,
  customPromptsEnc: null,
  customPromptsKeyEnc: null,
  customPromptsKdfSalt: null,
  useLocalTextEncoder: false,
  promptCacheSize: 1,
  promptEnhancerProviderPreference: null,
  seedLocked: false,
  lockedSeed: 42,
  modelsDir: '',
  remoteInferenceEnabled: false,
  livepeerVideoEnabled: true,
  livepeerImageEnabled: true,
  livepeerTextEncodingEnabled: true,
  livepeerDiscoveryUrl: '',
  hasLivepeerDiscoveryUrl: false,
  hasLivepeerApiKey: false,
  livepeerSelectedRunnerId: '',
  livepeerExcludedRunnerIds: [],
  platformBaseUrl: '',
  hasPlatformBaseUrl: false,
  platformUserId: '',
  hasPlatformApiKey: false,
  platformRecoveryEmail: '',
}

type BackendProcessStatus = 'alive' | 'restarting' | 'dead'

interface AppSettingsContextValue {
  settings: AppSettings
  isLoaded: boolean
  runtimePolicyLoaded: boolean
  updateSettings: (patch: Partial<AppSettings> | ((prev: AppSettings) => AppSettings)) => void
  refreshSettings: () => Promise<void>
  saveLtxApiKey: (value: string) => Promise<void>
  saveFalApiKey: (value: string) => Promise<void>
  saveGeminiApiKey: (value: string) => Promise<void>
  saveOpenRouterApiKey: (value: string) => Promise<void>
  setOpenRouterModel: (value: string) => Promise<void>
  saveCustomPrompts: (plain: CustomPrompts, opts?: { passphrase?: string }) => Promise<void>
  saveLivepeerDiscoveryUrl: (value: string) => Promise<void>
  saveLivepeerApiKey: (value: string) => Promise<void>
  forceApiGenerations: boolean
  shouldVideoGenerateWithLtxApi: boolean
  shouldImageGenerateWithFalApi: boolean
  cudaAvailable: boolean
}

const AppSettingsContext = createContext<AppSettingsContextValue | null>(null)

function toBackendProcessStatus(value: unknown): BackendProcessStatus | null {
  if (!value || typeof value !== 'object') {
    return null
  }

  const record = value as { status?: unknown }
  if (record.status === 'alive' || record.status === 'restarting' || record.status === 'dead') {
    return record.status
  }
  return null
}

function normalizeAppSettings(data: Partial<AppSettings>): AppSettings {
  return {
    useTorchCompile: data.useTorchCompile ?? DEFAULT_APP_SETTINGS.useTorchCompile,
    diffusionStageCacheEnabled: data.diffusionStageCacheEnabled ?? DEFAULT_APP_SETTINGS.diffusionStageCacheEnabled,
    hasLtxApiKey: data.hasLtxApiKey ?? DEFAULT_APP_SETTINGS.hasLtxApiKey,
    userPrefersLtxApiVideoGenerations: data.userPrefersLtxApiVideoGenerations ?? DEFAULT_APP_SETTINGS.userPrefersLtxApiVideoGenerations,
    hasFalApiKey: data.hasFalApiKey ?? DEFAULT_APP_SETTINGS.hasFalApiKey,
    userPrefersFalApiImageGenerations: data.userPrefersFalApiImageGenerations ?? DEFAULT_APP_SETTINGS.userPrefersFalApiImageGenerations,
    hasGeminiApiKey: data.hasGeminiApiKey ?? DEFAULT_APP_SETTINGS.hasGeminiApiKey,
    hasOpenRouterApiKey: data.hasOpenRouterApiKey ?? DEFAULT_APP_SETTINGS.hasOpenRouterApiKey,
    openrouterModel: data.openrouterModel ?? DEFAULT_APP_SETTINGS.openrouterModel,
    hasPromptEncryptionKey: data.hasPromptEncryptionKey ?? DEFAULT_APP_SETTINGS.hasPromptEncryptionKey,
    customPrompts: data.customPrompts ?? DEFAULT_APP_SETTINGS.customPrompts,
    customPromptsEnc: data.customPromptsEnc ?? DEFAULT_APP_SETTINGS.customPromptsEnc,
    customPromptsKeyEnc: data.customPromptsKeyEnc ?? DEFAULT_APP_SETTINGS.customPromptsKeyEnc,
    customPromptsKdfSalt: data.customPromptsKdfSalt ?? DEFAULT_APP_SETTINGS.customPromptsKdfSalt,
    useLocalTextEncoder: data.useLocalTextEncoder ?? DEFAULT_APP_SETTINGS.useLocalTextEncoder,
    promptCacheSize: data.promptCacheSize ?? DEFAULT_APP_SETTINGS.promptCacheSize,
    promptEnhancerProviderPreference: data.promptEnhancerProviderPreference ?? DEFAULT_APP_SETTINGS.promptEnhancerProviderPreference,
    seedLocked: data.seedLocked ?? DEFAULT_APP_SETTINGS.seedLocked,
    lockedSeed: data.lockedSeed ?? DEFAULT_APP_SETTINGS.lockedSeed,
    modelsDir: data.modelsDir ?? DEFAULT_APP_SETTINGS.modelsDir,
    remoteInferenceEnabled: data.remoteInferenceEnabled ?? DEFAULT_APP_SETTINGS.remoteInferenceEnabled,
    livepeerVideoEnabled: data.livepeerVideoEnabled ?? DEFAULT_APP_SETTINGS.livepeerVideoEnabled,
    livepeerImageEnabled: data.livepeerImageEnabled ?? DEFAULT_APP_SETTINGS.livepeerImageEnabled,
    livepeerTextEncodingEnabled: data.livepeerTextEncodingEnabled ?? DEFAULT_APP_SETTINGS.livepeerTextEncodingEnabled,
    livepeerDiscoveryUrl: data.livepeerDiscoveryUrl ?? DEFAULT_APP_SETTINGS.livepeerDiscoveryUrl,
    hasLivepeerDiscoveryUrl: data.hasLivepeerDiscoveryUrl ?? DEFAULT_APP_SETTINGS.hasLivepeerDiscoveryUrl,
    hasLivepeerApiKey: data.hasLivepeerApiKey ?? DEFAULT_APP_SETTINGS.hasLivepeerApiKey,
    livepeerSelectedRunnerId: data.livepeerSelectedRunnerId ?? DEFAULT_APP_SETTINGS.livepeerSelectedRunnerId,
    livepeerExcludedRunnerIds: data.livepeerExcludedRunnerIds ?? DEFAULT_APP_SETTINGS.livepeerExcludedRunnerIds,
    platformBaseUrl: data.platformBaseUrl ?? DEFAULT_APP_SETTINGS.platformBaseUrl,
    hasPlatformBaseUrl: data.hasPlatformBaseUrl ?? DEFAULT_APP_SETTINGS.hasPlatformBaseUrl,
    platformUserId: data.platformUserId ?? DEFAULT_APP_SETTINGS.platformUserId,
    hasPlatformApiKey: data.hasPlatformApiKey ?? DEFAULT_APP_SETTINGS.hasPlatformApiKey,
    platformRecoveryEmail: data.platformRecoveryEmail ?? DEFAULT_APP_SETTINGS.platformRecoveryEmail,
  }
}

type RuntimePolicyPayload = ApiSuccessOf<'getRuntimePolicy'>
type GpuInfoPayload = ApiSuccessOf<'getGpuInfo'>

export function AppSettingsProvider({ children }: { children: ReactNode }) {
  const [settings, setSettings] = useState<AppSettings>(DEFAULT_APP_SETTINGS)
  const [isLoaded, setIsLoaded] = useState(false)
  const [runtimePolicyLoaded, setRuntimePolicyLoaded] = useState(false)
  const [forceApiGenerations, setForceApiGenerations] = useState(true)
  const [cudaAvailable, setCudaAvailable] = useState(false)
  const [backendProcessStatus, setBackendProcessStatus] = useState<BackendProcessStatus | null>(null)

  useEffect(() => {
    if (backendProcessStatus !== 'alive') return

    let cancelled = false
    setRuntimePolicyLoaded(false)

    const fetchRuntimePolicy = async () => {
      const result = await ApiClient.getRuntimePolicy()
      if (!result.ok) {
        if (!cancelled) {
          // Fail closed until policy can be read.
          setForceApiGenerations(true)
          setRuntimePolicyLoaded(true)
        }
        return
      }

      const payload = result.data as RuntimePolicyPayload
      if (typeof payload.force_api_generations !== 'boolean') {
        if (!cancelled) {
          setForceApiGenerations(true)
        }
      } else if (!cancelled) {
        setForceApiGenerations(payload.force_api_generations)
      }

      if (!cancelled) {
        setRuntimePolicyLoaded(true)
      }
    }

    void fetchRuntimePolicy()

    return () => {
      cancelled = true
    }
  }, [backendProcessStatus])

  useEffect(() => {
    if (backendProcessStatus !== 'alive') return
    // Local GPU info is served by the desktop backend (under /api/local/gpu-info), not the
    // Cloudflare Worker (serverless has no CUDA). The web build never has local inference, so
    // skip the call entirely — its only consumer (CUDA-only settings toggles) can't render here.
    if (isWebPlatform()) return

    let cancelled = false

    const fetchGpuInfo = async () => {
      const result = await ApiClient.getGpuInfo()
      if (!result.ok || cancelled) return

      const payload = result.data as GpuInfoPayload
      setCudaAvailable(Boolean(payload.cuda_available))
    }

    void fetchGpuInfo()

    return () => {
      cancelled = true
    }
  }, [backendProcessStatus])

  useEffect(() => {
    let cancelled = false

    const applyStatus = (value: unknown) => {
      const nextStatus = toBackendProcessStatus(value)
      if (!nextStatus || cancelled) {
        return
      }
      if (nextStatus === 'alive') {
        resetBackendCredentials()
      }
      setBackendProcessStatus(nextStatus)
    }

    const unsubscribe = window.electronAPI.onBackendHealthStatus((data) => {
      applyStatus(data)
    })

    void window.electronAPI.getBackendHealthStatus()
      .then((snapshot) => {
        applyStatus(snapshot)
      })
      .catch(() => {
        // Snapshot is optional at startup; subscription continues to listen for pushes.
      })

    return () => {
      cancelled = true
      unsubscribe()
    }
  }, [])

  const refreshSettings = useCallback(async () => {
    const result = await ApiClient.getSettings()
    if (!result.ok) {
      throw new Error(result.error.message)
    }
    setSettings(normalizeAppSettings(result.data))
    setIsLoaded(true)
  }, [])

  useEffect(() => {
    if (isLoaded || backendProcessStatus !== 'alive') return

    let cancelled = false
    let retryTimer: ReturnType<typeof setTimeout> | null = null

    const fetchSettings = async () => {
      try {
        await refreshSettings()
        if (cancelled) return
      } catch {
        if (!cancelled) {
          retryTimer = setTimeout(fetchSettings, 1000)
        }
      }
    }

    fetchSettings()

    return () => {
      cancelled = true
      if (retryTimer) clearTimeout(retryTimer)
    }
  }, [backendProcessStatus, isLoaded, refreshSettings])

  useEffect(() => {
    if (!isLoaded || backendProcessStatus !== 'alive') return
    const syncTimer = setTimeout(async () => {
      const {
        hasLtxApiKey: _ltxKey,
        hasFalApiKey: _falKey,
        hasGeminiApiKey: _geminiKey,
        hasLivepeerDiscoveryUrl: _lpDiscovery,
        hasLivepeerApiKey: _lpKey,
        modelsDir: _modelsDir,
        ...syncPayload
      } = settings
      const result = await ApiClient.updateSettings(syncPayload)
      if (!result.ok) {
        // Best-effort settings sync.
      }
    }, 150)
    return () => clearTimeout(syncTimer)
  }, [backendProcessStatus, isLoaded, settings])

  // Push the runner-discovery inputs into the shared lib so resolveRunner() can do client-side
  // discovery against the user's Discovery URL without threading them through every call site.
  useEffect(() => {
    setRunnerDiscoveryConfig({
      discoveryUrl: settings.livepeerDiscoveryUrl,
      selectedRunnerId: settings.livepeerSelectedRunnerId,
    })
  }, [settings.livepeerDiscoveryUrl, settings.livepeerSelectedRunnerId])

  // If prompts were stored encrypted (a passphrase exists in this browser's localStorage),
  // decrypt them into local plaintext state so the Prompt Builder + dispatch can use them.
  // Without a cached passphrase -> mark needsKey (hasPromptEncryptionKey false) so the UI
  // prompts; the cloud copy is unrecoverable by design.
  useEffect(() => {
    if (!isLoaded) return
    const e = settings.customPromptsEnc
    if (!e || !settings.customPromptsKeyEnc || !settings.customPromptsKdfSalt) return
    if (settings.customPrompts) return // already have local plaintext
    let cancelled = false
    let pass = ''
    try { pass = localStorage.getItem(CUSTOM_PROMPT_VAULT_KEY) || '' } catch { /* non-fatal */ }
    if (!pass) {
      setSettings((prev) => ({ ...prev, hasPromptEncryptionKey: false }))
      return
    }
    void decryptPromptsKeyed(pass, { enc: e, keyEnc: settings.customPromptsKeyEnc, kdfSalt: settings.customPromptsKdfSalt })
      .then((plain) => {
        if (cancelled) return
        try { setSettings((prev) => ({ ...prev, customPrompts: JSON.parse(plain), hasPromptEncryptionKey: true })) } catch { /* unparseable */ }
      })
      .catch(() => { if (!cancelled) setSettings((prev) => ({ ...prev, hasPromptEncryptionKey: false })) })
    return () => { cancelled = true }
  }, [isLoaded, settings.customPrompts, settings.customPromptsEnc, settings.customPromptsKeyEnc, settings.customPromptsKdfSalt])

  const updateSettings = useCallback((patch: Partial<AppSettings> | ((prev: AppSettings) => AppSettings)) => {
    if (typeof patch === 'function') {
      setSettings((prev) => patch(prev))
      return
    }
    setSettings((prev) => ({ ...prev, ...patch }))
  }, [])

  const saveLtxApiKey = useCallback(async (value: string) => {
    const result = await ApiClient.updateSettings({ ltxApiKey: value })
    if (!result.ok) {
      throw new Error(result.error.message)
    }
    await refreshSettings()
  }, [refreshSettings])

  const saveGeminiApiKey = useCallback(async (value: string) => {
    const result = await ApiClient.updateSettings({ geminiApiKey: value })
    if (!result.ok) {
      throw new Error(result.error.message)
    }
    await refreshSettings()
  }, [refreshSettings])

  const saveFalApiKey = useCallback(async (value: string) => {
    const result = await ApiClient.updateSettings({ falApiKey: value })
    if (!result.ok) {
      throw new Error(result.error.message)
    }
    await refreshSettings()
  }, [refreshSettings])

  const saveOpenRouterApiKey = useCallback(async (value: string) => {
    const result = await ApiClient.updateSettings({ openrouterApiKey: value })
    if (!result.ok) {
      throw new Error(result.error.message)
    }
    await refreshSettings()
  }, [refreshSettings])

  const setOpenRouterModel = useCallback(async (value: string) => {
    const result = await ApiClient.updateSettings({ openrouterModel: value })
    if (!result.ok) {
      throw new Error(result.error.message)
    }
    await refreshSettings()
  }, [refreshSettings])

  // Persist custom prompts. passphrase set -> envelope-encrypt and store ciphertext (the
  // passphrase lives in THIS browser's localStorage only, never sent to the server);
  // no passphrase -> store plaintext in D1 (server-readable). Always keeps a browser-local
  // plaintext draft so the Prompt Builder can seed the editor without a network hit.
  const saveCustomPrompts = useCallback(async (plain: CustomPrompts, opts?: { passphrase?: string }) => {
    try { localStorage.setItem('custom-prompts-draft', JSON.stringify(plain)) } catch { /* non-fatal */ }
    if (opts?.passphrase) {
      const salt = newSalt()
      const c = await encryptPromptsKeyed(opts.passphrase, salt, JSON.stringify(plain))
      try { localStorage.setItem(CUSTOM_PROMPT_VAULT_KEY, opts.passphrase) } catch { /* non-fatal */ }
      const result = await ApiClient.updateSettings({
        customPrompts: null,
        customPromptsEnc: c.enc,
        customPromptsKeyEnc: c.keyEnc,
        customPromptsKdfSalt: c.kdfSalt,
        hasPromptEncryptionKey: true,
      })
      if (!result.ok) throw new Error(result.error.message)
      await refreshSettings()
    } else {
      try { localStorage.setItem(CUSTOM_PROMPT_VAULT_KEY, '') } catch { /* non-fatal */ }
      const result = await ApiClient.updateSettings({
        customPrompts: plain,
        customPromptsEnc: null,
        customPromptsKeyEnc: null,
        customPromptsKdfSalt: null,
        hasPromptEncryptionKey: false,
      })
      if (!result.ok) throw new Error(result.error.message)
      await refreshSettings()
    }
  }, [refreshSettings])

  const saveLivepeerDiscoveryUrl = useCallback(async (value: string) => {
    // Default every per-section generation method to Livepeer whenever a Discovery
    // URL is configured. Persisted atomically with the URL so the subsequent
    // refreshSettings() returns the new defaults. Users can still switch each
    // section off individually in settings.
    const result = await ApiClient.updateSettings({
      livepeerDiscoveryUrl: value,
      livepeerVideoEnabled: true,
      livepeerImageEnabled: true,
      livepeerTextEncodingEnabled: true,
    })
    if (!result.ok) {
      throw new Error(result.error.message)
    }
    await refreshSettings()
  }, [refreshSettings])

  const saveLivepeerApiKey = useCallback(async (value: string) => {
    const result = await ApiClient.updateSettings({ livepeerApiKey: value })
    if (!result.ok) {
      throw new Error(result.error.message)
    }
    await refreshSettings()
  }, [refreshSettings])

  const shouldVideoGenerateWithLtxApi =
    forceApiGenerations || (settings.userPrefersLtxApiVideoGenerations && settings.hasLtxApiKey)
  const shouldImageGenerateWithFalApi =
    forceApiGenerations || (settings.userPrefersFalApiImageGenerations && settings.hasFalApiKey)

  const contextValue = useMemo<AppSettingsContextValue>(
    () => ({
      settings,
      isLoaded,
      runtimePolicyLoaded,
      updateSettings,
      refreshSettings,
      saveLtxApiKey,
      saveFalApiKey,
      saveGeminiApiKey,
      saveOpenRouterApiKey,
      setOpenRouterModel,
      saveCustomPrompts,
      saveLivepeerDiscoveryUrl,
      saveLivepeerApiKey,
      forceApiGenerations,
      shouldVideoGenerateWithLtxApi,
      shouldImageGenerateWithFalApi,
      cudaAvailable,
    }),
    [cudaAvailable, forceApiGenerations, isLoaded, refreshSettings, runtimePolicyLoaded, saveCustomPrompts, saveFalApiKey, saveGeminiApiKey, saveLtxApiKey, saveLivepeerDiscoveryUrl, saveLivepeerApiKey, saveOpenRouterApiKey, setOpenRouterModel, settings, shouldVideoGenerateWithLtxApi, shouldImageGenerateWithFalApi, updateSettings],
  )

  return <AppSettingsContext.Provider value={contextValue}>{children}</AppSettingsContext.Provider>
}

export function useAppSettings() {
  const context = useContext(AppSettingsContext)
  if (!context) {
    throw new Error('useAppSettings must be used within AppSettingsProvider')
  }
  return context
}
