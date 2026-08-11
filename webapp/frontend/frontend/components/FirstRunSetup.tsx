import { useState, useEffect, useRef, useCallback } from 'react'
import { ApiClient, type ApiRequestBodyOf, type ApiSuccessOf } from '../lib/api-client'
import { formatBytes } from '../lib/format'
import { logger } from '../lib/logger'
import { useHfAuth } from '../hooks/use-hf-auth'
import { useAppSettings } from '../contexts/AppSettingsContext'
import './FirstRunSetup.css'

interface LaunchGateProps {
  licenseOnly?: boolean
  showLicenseStep?: boolean
  onComplete: () => Promise<void>
  onAcceptLicense?: () => Promise<void>
}

type Step = 'license' | 'location' | 'installing' | 'complete'
type StartModelDownloadBody = NonNullable<ApiRequestBodyOf<'startModelDownload'>>
type ModelCheckpointID = NonNullable<StartModelDownloadBody['cp_ids']>[number]
type LtxRecommendation = ApiSuccessOf<'getLtxRecommendation'>
type ImgGenRecommendation = ApiSuccessOf<'getImgGenRecommendation'>
type DownloadProgress = ApiSuccessOf<'getModelDownloadProgress'>
type CheckpointDescriptor = ApiSuccessOf<'describeCheckpoints'>['checkpoints'][number]
type DownloadStepSpec = {
  type: StartModelDownloadBody['type']
  cpIds: ModelCheckpointID[]
}

// Fun loading messages
const INSTALL_MESSAGES = [
  "Downloading model weights...",
  "Teaching AI to dream in 4K...",
  "Loading neural pathways...",
  "Calibrating inference engine...",
  "Almost there...",
  "Unpacking the magic...",
  "Configuring parameters...",
  "Finalizing installation..."
]

function uniqueCpIds(cpIds: readonly ModelCheckpointID[]): ModelCheckpointID[] {
  return [...new Set(cpIds)]
}

// User-facing explanation per checkpoint role (moved out of the backend so wording/i18n
// iterates without a backend deploy). Keyed by the stable `role` the backend ships.
const CP_INFO_BY_ROLE: Record<CheckpointDescriptor['role'], string> = {
  base: 'The core LTX video model that turns your prompt into video frames. This is the largest download.',
  upscaler:
    'Doubles the resolution of generated video for sharper, more detailed output. This updated version also ' +
    'fixes glitches and stray text or overlay artifacts that could appear near the end of longer clips, and ' +
    'keeps detail consistent through the final frames — recommended for long videos.',
  text_encoder:
    'Reads your text prompt so the model understands it. You can skip this large download by entering an LTX ' +
    'API key, which encodes prompts via the API instead.',
  image: 'Generates still images from text prompts (used for image-to-video and image tools).',
  support: 'A supporting model used for guided generation (depth, edges, or pose control).',
}

// One line in the first-run "What will be downloaded" list: checkpoint name, an
// info-tooltip icon, and the size (or a "skipped" note when an API key covers it).
function DownloadItem({ item, skipped }: { item: CheckpointDescriptor; skipped: boolean }) {
  return (
    <div
      style={{
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        fontSize: 13,
        padding: '6px 0',
        color: skipped ? '#666' : '#e0e0e0',
        opacity: skipped ? 0.7 : 1,
      }}
    >
      <span style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
        <span style={{ textDecoration: skipped ? 'line-through' : 'none' }}>{item.name}</span>
        <span
          title={CP_INFO_BY_ROLE[item.role]}
          style={{
            display: 'inline-flex',
            alignItems: 'center',
            justifyContent: 'center',
            width: 15,
            height: 15,
            borderRadius: '50%',
            border: '1px solid #6b7280',
            color: '#9ca3af',
            fontSize: 10,
            fontStyle: 'italic',
            fontFamily: 'Georgia, serif',
            cursor: 'help',
            flexShrink: 0,
          }}
        >
          i
        </span>
      </span>
      <span style={{ color: skipped ? '#666' : '#a0a0a0', flexShrink: 0, marginLeft: 12 }}>
        {skipped ? 'Skipped (API key)' : formatBytes(item.size_bytes)}
      </span>
    </div>
  )
}

function buildDownloadSteps(
  ltxRecommendation: LtxRecommendation,
  imgGenRecommendation: ImgGenRecommendation,
): DownloadStepSpec[] {
  const cpIds: ModelCheckpointID[] = []
  if (ltxRecommendation.status === 'download') {
    cpIds.push(...ltxRecommendation.cps_to_download)
  }
  if (imgGenRecommendation.cp_to_download) {
    cpIds.push(imgGenRecommendation.cp_to_download)
  }
  const unique = uniqueCpIds(cpIds)
  return unique.length > 0 ? [{ type: 'download', cpIds: unique }] : []
}


export function LaunchGate({
  licenseOnly,
  showLicenseStep = true,
  onComplete,
  onAcceptLicense,
}: LaunchGateProps) {
  const [currentStep, setCurrentStep] = useState<Step>(showLicenseStep ? 'license' : 'location')
  const [installPath, setInstallPath] = useState('')
  const [downloadProgress, setDownloadProgress] = useState<DownloadProgress | null>(null)
  const [downloadError, setDownloadError] = useState<string | null>(null)
  const [downloadSessionId, setDownloadSessionId] = useState<string | null>(null)
  const [installMessage, setInstallMessage] = useState(INSTALL_MESSAGES[0])
  const [availableSpace, setAvailableSpace] = useState('...')
  const [downloadItems, setDownloadItems] = useState<CheckpointDescriptor[]>([])
  const [videoPath, setVideoPath] = useState('/splash/splash.mp4')
  const [ltxApiKey, setLtxApiKey] = useState('')
  const [licenseAccepted, setLicenseAccepted] = useState(false)
  const [licenseText, setLicenseText] = useState<string | null>(null)
  const [licenseError, setLicenseError] = useState<string | null>(null)
  const [actionError, setActionError] = useState<string | null>(null)
  const [isActionPending, setIsActionPending] = useState(false)
  const { hfAuthStatus, hfAuthPolling, startHuggingFaceLogin } = useHfAuth(currentStep === 'location')
  const { saveLtxApiKey } = useAppSettings()
  const downloadQueueRef = useRef<DownloadStepSpec[]>([])
  const runningDownloadProgress = downloadProgress?.status === 'downloading' ? downloadProgress : null
  const totalProgress = runningDownloadProgress?.total_progress ?? (downloadProgress?.status === 'complete' ? 100 : 0)

  // The text encoder download is skipped when an API key is entered (the backend
  // omits it once a key is saved); reflect that live in the preview + total.
  const isItemSkipped = (item: CheckpointDescriptor): boolean =>
    item.role === 'text_encoder' && ltxApiKey.trim().length > 0
  const totalDownloadBytes = downloadItems
    .filter((item) => !isItemSkipped(item))
    .reduce((sum, item) => sum + item.size_bytes, 0)

  // Format time remaining
  const formatTimeRemaining = (seconds: number): string => {
    if (!seconds || !isFinite(seconds) || seconds <= 0) return '--'
    if (seconds < 60) return `${Math.round(seconds)}s`
    if (seconds < 3600) return `${Math.round(seconds / 60)}m`
    return `${Math.round(seconds / 3600)}h ${Math.round((seconds % 3600) / 60)}m`
  }

  // Calculate ETA based on speed and remaining bytes
  const getTimeRemaining = (): string => {
    if (!runningDownloadProgress || runningDownloadProgress.speed_bytes_per_sec <= 0) return '--'
    const remainingBytes = runningDownloadProgress.expected_total_bytes - runningDownloadProgress.total_downloaded_bytes
    if (remainingBytes <= 0) return '--'
    const secondsRemaining = remainingBytes / runningDownloadProgress.speed_bytes_per_sec
    return formatTimeRemaining(secondsRemaining)
  }

  // Fetch license text
  const fetchLicense = async () => {
    setLicenseError(null)
    setLicenseText(null)
    try {
      const text = await window.electronAPI.fetchLicenseText()
      setLicenseText(text)
    } catch (e) {
      setLicenseError(e instanceof Error ? e.message : 'Failed to fetch license text.')
    }
  }

  const refreshModelRecommendations = useCallback(async () => {
    if (licenseOnly) return

    const [settingsResult, ltxResult, imgGenResult] = await Promise.all([
      ApiClient.getSettings(),
      ApiClient.getLtxRecommendation(),
      ApiClient.getImgGenRecommendation(),
    ])
    if (!settingsResult.ok) {
      logger.error(`Failed to fetch model recommendations: ${settingsResult.error.message}`)
      return
    }
    if (!ltxResult.ok) {
      logger.error(`Failed to fetch model recommendations: ${ltxResult.error.message}`)
      return
    }
    if (!imgGenResult.ok) {
      logger.error(`Failed to fetch model recommendations: ${imgGenResult.error.message}`)
      return
    }

    setInstallPath(settingsResult.data.modelsDir ?? '')

    // Surface exactly what the install will download (base model, upscaler, text
    // encoder, image model) with per-checkpoint sizes and info. Same cp set the
    // installer actually downloads, so the preview can't drift from reality.
    const cpIds = buildDownloadSteps(ltxResult.data, imgGenResult.data).flatMap((step) => step.cpIds)
    if (cpIds.length === 0) {
      setDownloadItems([])
      return
    }
    const describeResult = await ApiClient.describeCheckpoints({ cp_ids: cpIds })
    if (!describeResult.ok) {
      logger.error(`Failed to describe checkpoints: ${describeResult.error.message}`)
      return
    }
    setDownloadItems(describeResult.data.checkpoints)
  }, [licenseOnly])

  const startDownloadStep = useCallback(async (step: DownloadStepSpec) => {
    setDownloadProgress(null)
    setDownloadError(null)
    const result = await ApiClient.startModelDownload({
      type: step.type,
      cp_ids: step.cpIds,
    })
    if (!result.ok) {
      throw new Error(result.error.message)
    }
    const downloadData = result.data
    if (downloadData.status === 'started') {
      setDownloadSessionId(downloadData.sessionId)
      return
    }
    throw new Error('Unexpected response while starting model download.')
  }, [])

  // Initialize
  useEffect(() => {
    const init = async () => {
      try {
        // Get video path for production (unpacked from asar)
        try {
          const resourcePath = await window.electronAPI.getResourcePath?.()
          if (resourcePath) {
            setVideoPath(`file://${resourcePath}/app.asar.unpacked/dist/splash/splash.mp4`)
          }
        } catch {
          // Dev mode: use relative path
          setVideoPath('/splash/splash.mp4')
        }

        await refreshModelRecommendations()

        // TODO: Get actual available space
        setAvailableSpace('1.8 TB')
      } catch (e) {
        logger.error(`Init error: ${e}`)
      }
    }
    init()
    if (showLicenseStep) {
      void fetchLicense()
    }
  }, [refreshModelRecommendations, showLicenseStep])

  // Cycle install messages
  useEffect(() => {
    if (currentStep !== 'installing') return
    let index = 0
    const interval = setInterval(() => {
      index = (index + 1) % INSTALL_MESSAGES.length
      setInstallMessage(INSTALL_MESSAGES[index])
    }, 4000)
    return () => clearInterval(interval)
  }, [currentStep])

  // Poll download progress during installation
  useEffect(() => {
    if (currentStep !== 'installing' || !downloadSessionId) return

    const pollProgress = async () => {
      const result = await ApiClient.getModelDownloadProgress({ sessionId: downloadSessionId })
      if (!result.ok) {
        logger.error(`Progress poll error: ${result.error.message}`)
        return
      }

      const progress = result.data
      setDownloadProgress(progress)

      if (progress.status === 'error') {
        downloadQueueRef.current = []
        setDownloadError(progress.error || 'Download failed.')
      } else if (progress.status === 'complete') {
        const nextStep = downloadQueueRef.current.shift() ?? null
        if (nextStep) {
          await startDownloadStep(nextStep)
          return
        }
        setTimeout(() => setCurrentStep('complete'), 600)
      }
    }

    pollProgress()
    const interval = setInterval(pollProgress, 500)
    return () => clearInterval(interval)
  }, [currentStep, downloadSessionId, startDownloadStep])

  // Start installation
  const startInstallation = async () => {
    setCurrentStep('installing')
    try {
      if (ltxApiKey.trim()) {
        try {
          await saveLtxApiKey(ltxApiKey.trim())
        } catch (e) {
          logger.error(`Failed to save API key: ${e instanceof Error ? e.message : String(e)}`)
        }
      }

      const [ltxResult, imgGenResult] = await Promise.all([
        ApiClient.getLtxRecommendation(),
        ApiClient.getImgGenRecommendation(),
      ])
      if (!ltxResult.ok) {
        throw new Error(ltxResult.error.message)
      }
      if (!imgGenResult.ok) {
        throw new Error(imgGenResult.error.message)
      }
      const nextLtxRecommendation = ltxResult.data
      const nextImgGenRecommendation = imgGenResult.data

      const downloadSteps = buildDownloadSteps(nextLtxRecommendation, nextImgGenRecommendation)
      if (downloadSteps.length === 0) {
        setCurrentStep('complete')
        return
      }

      downloadQueueRef.current = downloadSteps.slice(1)
      await startDownloadStep(downloadSteps[0])
    } catch (e) {
      logger.error(`Download start error: ${e}`)
      setDownloadError(e instanceof Error ? e.message : 'Failed to start model download.')
    }
  }

  const retryInstallation = () => {
    setDownloadError(null)
    downloadQueueRef.current = []
    startInstallation()
  }

  // Handle next button
  const handleNext = async () => {
    setActionError(null)
    if (currentStep === 'license') {
      if (!licenseAccepted) return
      setIsActionPending(true)
      try {
        if (onAcceptLicense) {
          await onAcceptLicense()
        }
        if (licenseOnly) {
          await onComplete()
          return
        }
        setCurrentStep('location')
      } catch (e) {
        setActionError(e instanceof Error ? e.message : 'Failed to accept license.')
      } finally {
        setIsActionPending(false)
      }
      return
    }
    if (currentStep === 'location') {
      startInstallation()
      return
    }
    if (currentStep === 'complete') {
      await handleFinish()
    }
  }

  const handleFinish = async () => {
    setActionError(null)
    setIsActionPending(true)
    try {
      await onComplete()
    } catch (e) {
      setActionError(e instanceof Error ? e.message : 'Failed to complete setup.')
    } finally {
      setIsActionPending(false)
    }
  }

  // Get button text
  const getNextButtonText = () => {
    if (currentStep === 'license') return licenseOnly ? 'Accept' : 'Next'
    if (currentStep === 'location') return 'Install'
    if (currentStep === 'complete') return 'Finish'
    return 'Continue'
  }

  // Check if next button should be disabled
  const isNextDisabled = () => {
    if (currentStep === 'license') return !licenseAccepted || isActionPending
    // HF sign-in is optional (base models are public) — don't block setup on it.
    if (currentStep === 'location') return false
    if (currentStep === 'complete') return isActionPending
    return false
  }

  return (
    <div className="h-screen flex flex-col" style={{
      background: '#000000',
      fontFamily: 'Arial, Helvetica, sans-serif',
      color: '#ffffff'
    }}>
      {/* Custom Title Bar */}
      <div style={{
        height: 32,
        background: '#000000',
        display: 'flex',
        alignItems: 'center',
        paddingLeft: 80,
        borderBottom: '1px solid #1a1a1a',
        // @ts-expect-error - Electron-specific CSS property
        WebkitAppRegion: 'drag'
      }}>
      </div>

      {/* Main Container */}
      <div style={{
        display: 'flex',
        flexDirection: 'column',
        flex: 1,
        overflow: 'hidden',
        minHeight: 0,
        // @ts-expect-error - Electron-specific CSS property
        WebkitAppRegion: 'no-drag'
      }}>
        {/* Header */}
        <div style={{
          padding: currentStep === 'installing' ? '12px 32px' : '16px 32px',
          borderBottom: '1px solid #1a1a1a'
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
            {/* Video Creator wordmark */}
            <span style={{ display: 'flex', alignItems: 'baseline', gap: 6, fontWeight: 800, letterSpacing: '-0.02em', fontSize: 24 }}>
              <span style={{ color: '#ffffff' }}>V</span>
              <span style={{ color: '#3b82f6' }}>C</span>
            </span>
            <span style={{ fontSize: 14, color: '#71717a', fontWeight: 500, letterSpacing: '0.02em', paddingTop: 2, paddingLeft: 6 }}>Video Creator</span>
          </div>
        </div>

        {/* Content Area */}
        <div style={{
          flex: 1,
          padding: currentStep === 'installing' ? 0 : '28px 32px',
          display: 'flex',
          flexDirection: 'column',
          overflow: 'hidden'
        }}>
          {/* Step 1: Model License */}
          {currentStep === 'license' && (
            <div style={{ animation: 'fadeIn 0.25s ease', display: 'flex', flexDirection: 'column', overflow: 'hidden', flex: 1 }}>
              <h2 style={{
                fontFamily: "'Miriam Libre', serif",
                fontSize: 24,
                fontWeight: 700,
                marginBottom: 6
              }}>
                LTX-2 Model License
              </h2>
              <p style={{ color: '#a0a0a0', fontSize: 14, marginBottom: 16 }}>
                The LTX-2 model is subject to the following license agreement. Please review and accept before downloading.
              </p>

              <div style={{
                flex: 1,
                display: 'flex',
                flexDirection: 'column',
                overflow: 'hidden',
                minHeight: 0
              }}>
                <div style={{
                  flex: 1,
                  overflow: 'hidden',
                  borderRadius: 8,
                  minHeight: 0
                }}>
                  {licenseError ? (
                    <div style={{
                      display: 'flex',
                      flexDirection: 'column',
                      alignItems: 'center',
                      justifyContent: 'center',
                      height: '100%',
                      gap: 12
                    }}>
                      <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#f87171" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <circle cx="12" cy="12" r="10"/>
                        <line x1="12" y1="8" x2="12" y2="12"/>
                        <line x1="12" y1="16" x2="12.01" y2="16"/>
                      </svg>
                      <span style={{ color: '#f87171', fontSize: 13, textAlign: 'center' }}>{licenseError}</span>
                      <button
                        onClick={fetchLicense}
                        style={{
                          padding: '6px 20px',
                          borderRadius: 9999,
                          fontSize: 13,
                          fontWeight: 600,
                          cursor: 'pointer',
                          background: 'linear-gradient(125deg, #A98BD9, #6D28D9)',
                          border: 'none',
                          color: '#ffffff',
                        }}
                      >
                        Retry
                      </button>
                    </div>
                  ) : licenseText === null ? (
                    <div style={{
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      height: '100%',
                      gap: 10
                    }}>
                      <svg width="20" height="20" viewBox="0 0 24 24" style={{ animation: 'spin 1s linear infinite' }}>
                        <circle cx="12" cy="12" r="10" stroke="#6D28D9" strokeWidth="3" fill="none" strokeDasharray="31.4 31.4" strokeLinecap="round" />
                      </svg>
                      <span style={{ color: '#a0a0a0', fontSize: 13 }}>Loading license...</span>
                    </div>
                  ) : (
                    <div style={{
                      overflowY: 'auto',
                      height: '100%',
                      background: '#1a1a1a',
                      borderRadius: 8,
                      padding: 40
                    }}>
                      <pre style={{
                        fontFamily: "'Consolas', 'Monaco', monospace",
                        fontSize: 11,
                        lineHeight: 1.5,
                        color: '#d0d0d0',
                        margin: 0,
                        whiteSpace: 'pre-line',
                        wordWrap: 'break-word',
                        width: '100%'
                      }}>
                        {licenseText?.replace(/([^\n])\n([^\n])/g, '$1 $2')}
                      </pre>
                    </div>
                  )}
                </div>

                <label style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: 10,
                  marginTop: 14,
                  cursor: 'pointer',
                  fontSize: 13,
                  userSelect: 'none'
                }}>
                  <input
                    type="checkbox"
                    checked={licenseAccepted}
                    onChange={(e) => setLicenseAccepted(e.target.checked)}
                    style={{
                      width: 16,
                      height: 16,
                      accentColor: '#2B61FF',
                      cursor: 'pointer',
                      flexShrink: 0
                    }}
                  />
                  <span>I have read and agree to the LTX-2 Community License Agreement</span>
                </label>
              </div>
            </div>
          )}

          {/* Step 2: Choose Location */}
          {currentStep === 'location' && (
            <div style={{ animation: 'fadeIn 0.25s ease', overflowY: 'auto', flex: 1, minHeight: 0 }}>
              <h2 style={{
                fontFamily: "'Miriam Libre', serif",
                fontSize: 24,
                fontWeight: 700,
                marginBottom: 6
              }}>
                Choose Location
              </h2>
              <p style={{ color: '#a0a0a0', fontSize: 14, marginBottom: 24 }}>
                Select where to install the model files.
              </p>

              <div style={{
                background: '#2e3445',
                borderRadius: 12,
                padding: '14px 18px'
              }}>
                <div style={{ display: 'flex', gap: 12, alignItems: 'center' }}>
                  <input
                    type="text"
                    value={installPath}
                    readOnly
                    style={{
                      flex: 1,
                      background: '#1a1a1a',
                      border: '1px solid #333',
                      borderRadius: 8,
                      padding: '12px 14px',
                      color: '#ffffff',
                      fontSize: 13,
                      fontFamily: "'Consolas', 'Monaco', monospace"
                    }}
                  />
                  <button
                    onClick={async () => {
                      const result = await window.electronAPI?.openModelsDirChangeDialog()
                      if (result?.success) {
                        setInstallPath(result.path)
                      }
                    }}
                    style={{
                      padding: '10px 28px',
                      borderRadius: 9999,
                      fontSize: 13,
                      fontWeight: 600,
                      cursor: 'pointer',
                      background: 'transparent',
                      border: '1px solid #444',
                      color: '#ffffff',
                      transition: 'all 0.2s ease'
                    }}
                  >
                    Browse
                  </button>
                </div>

                <div style={{
                  display: 'flex',
                  justifyContent: 'flex-end',
                  fontSize: 12,
                  color: '#a0a0a0',
                  marginTop: 10
                }}>
                  <span>Available: <strong style={{ color: '#fff' }}>{availableSpace}</strong></span>
                </div>
              </div>

              {/* What will be downloaded */}
              {downloadItems.length > 0 && (
                <div style={{
                  marginTop: 24,
                  background: '#2e3445',
                  borderRadius: 12,
                  padding: '14px 18px'
                }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', marginBottom: 10 }}>
                    <label style={{ fontSize: 13, fontWeight: 600, color: '#ffffff' }}>What will be downloaded</label>
                    <span style={{ fontSize: 12, color: '#a0a0a0' }}>
                      Total: <strong style={{ color: '#fff' }}>{formatBytes(totalDownloadBytes)}</strong>
                    </span>
                  </div>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                    {downloadItems.map((item) => (
                      <DownloadItem key={item.cp_id} item={item} skipped={isItemSkipped(item)} />
                    ))}
                  </div>
                </div>
              )}

              {/* LTX API Key - Optional but saves ~25 GB download */}
              <div style={{
                marginTop: 24,
                background: '#2e3445',
                borderRadius: 12,
                padding: '14px 18px'
              }}>
                <div style={{ marginBottom: 8 }}>
                  <label style={{ fontSize: 13, fontWeight: 600, color: '#ffffff' }}>
                    LTX API Key
                    <span style={{
                      fontSize: 11,
                      color: '#A98BD9',
                      marginLeft: 8,
                      fontWeight: 400
                    }}>
                      Optional - Saves ~25 GB download
                    </span>
                  </label>
                </div>
                <input
                  type="password"
                  value={ltxApiKey}
                  onChange={(e) => setLtxApiKey(e.target.value)}
                  placeholder="Enter API key to skip text encoder download..."
                  style={{
                    width: '100%',
                    background: '#1a1a1a',
                    border: '1px solid #333',
                    borderRadius: 8,
                    padding: '12px 14px',
                    color: '#ffffff',
                    fontSize: 13,
                    boxSizing: 'border-box'
                  }}
                />
                <p style={{ fontSize: 11, color: '#888', marginTop: 8 }}>
                  {ltxApiKey ? (
                    <span style={{ color: '#6D28D9' }}>
                      ✓ Text encoder download will be skipped (using API instead)
                    </span>
                  ) : (
                    'If you have an LTX API key, entering it here skips the 25 GB text encoder download. ' +
                    'The API provides faster text encoding (~1s vs 23s local).'
                  )}
                </p>
              </div>

              {/* HuggingFace Authentication */}
              <div style={{
                marginTop: 24,
                background: '#2e3445',
                borderRadius: 12,
                padding: '14px 18px'
              }}>
                <div style={{ marginBottom: 8 }}>
                  <label style={{ fontSize: 13, fontWeight: 600, color: '#ffffff' }}>
                    HuggingFace Account
                    <span style={{
                      fontSize: 11,
                      color: hfAuthStatus === 'authenticated' ? '#22c55e' : '#888',
                      marginLeft: 8,
                      fontWeight: 400
                    }}>
                      {hfAuthStatus === 'authenticated' ? 'Signed in' : 'Optional'}
                    </span>
                  </label>
                </div>
                {hfAuthStatus === 'authenticated' ? (
                  <p style={{ fontSize: 12, color: '#22c55e' }}>
                    ✓ Authenticated — gated models will download with your account.
                  </p>
                ) : (
                  <>
                    <p style={{ fontSize: 11, color: '#888', marginBottom: 12 }}>
                      Optional. The base models download without an account — sign in only to
                      download gated models (some catalog LoRAs / IC-LoRAs require it). You can
                      also do this later in Settings.
                    </p>
                    <button
                      onClick={startHuggingFaceLogin}
                      disabled={hfAuthPolling}
                      style={{
                        padding: '10px 28px',
                        borderRadius: 9999,
                        fontSize: 13,
                        fontWeight: 600,
                        cursor: hfAuthPolling ? 'default' : 'pointer',
                        background: hfAuthPolling ? '#333' : '#4f46e5',
                        border: 'none',
                        color: '#ffffff',
                        transition: 'all 0.2s ease',
                        opacity: hfAuthPolling ? 0.7 : 1
                      }}
                    >
                      {hfAuthPolling ? 'Waiting for sign in...' : 'Sign in with HuggingFace'}
                    </button>
                  </>
                )}
              </div>

            </div>
          )}

          {/* Step 3: Installing */}
          {currentStep === 'installing' && (
            <div style={{
              position: 'relative',
              height: '100%',
              animation: 'fadeIn 0.25s ease'
            }}>
              {/* Video Section - fills container but leaves room for progress */}
              <div style={{
                position: 'absolute',
                top: 0,
                left: 0,
                right: 0,
                bottom: 140,
                background: '#0a0a0a',
                overflow: 'hidden'
              }}>
                {/* Splash Video */}
                <video
                  key={videoPath}
                  autoPlay
                  loop
                  muted
                  playsInline
                  style={{
                    width: '100%',
                    height: '100%',
                    objectFit: 'cover',
                    display: 'block'
                  }}
                >
                  <source src={videoPath} type="video/mp4" />
                </video>

                {/* Video Credit */}
                <div style={{
                  position: 'absolute',
                  bottom: 20,
                  left: 24,
                  fontFamily: "'Miriam Libre', serif",
                  fontSize: 13,
                  color: 'rgba(255,255,255,0.75)',
                  textShadow: '0 1px 4px rgba(0,0,0,0.9)',
                  zIndex: 10
                }}>
                  Generated by PongFlongo
                </div>
              </div>

              {/* Progress Section - fixed at bottom */}
              <div style={{
                position: 'absolute',
                left: 0,
                right: 0,
                bottom: 0,
                height: 140,
                background: '#0d0d0d',
                padding: '16px 24px',
                borderTop: '1px solid #2a2a2a'
              }}>
              {downloadError ? (
                <div style={{
                  display: 'flex',
                  flexDirection: 'column',
                  alignItems: 'center',
                  justifyContent: 'center',
                  height: '100%',
                  gap: 10,
                }}>
                  <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#f87171" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <circle cx="12" cy="12" r="10"/>
                    <line x1="12" y1="8" x2="12" y2="12"/>
                    <line x1="12" y1="16" x2="12.01" y2="16"/>
                  </svg>
                  <span style={{ color: '#f87171', fontSize: 13, textAlign: 'center', maxWidth: 400 }}>{downloadError}</span>
                  <div style={{ display: 'flex', gap: 10 }}>
                    <button
                      onClick={() => { setDownloadError(null); setCurrentStep('location') }}
                      style={{
                        padding: '6px 20px',
                        borderRadius: 9999,
                        fontSize: 13,
                        fontWeight: 600,
                        cursor: 'pointer',
                        background: 'transparent',
                        border: '1px solid #444',
                        color: '#ffffff',
                      }}
                    >
                      Back
                    </button>
                    <button
                      onClick={retryInstallation}
                      style={{
                        padding: '6px 20px',
                        borderRadius: 9999,
                        fontSize: 13,
                        fontWeight: 600,
                        cursor: 'pointer',
                        background: 'linear-gradient(125deg, #A98BD9, #6D28D9)',
                        border: 'none',
                        color: '#ffffff',
                      }}
                    >
                      Retry
                    </button>
                  </div>
                </div>
              ) : (
              <>
                {/* Header row with status and percentage */}
                <div style={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                  marginBottom: 8
                }}>
                  <span style={{ fontSize: 13, fontWeight: 500 }}>
                    {totalProgress > 85 ? 'Installing...' : 'Downloading...'}
                  </span>
                  <span style={{ fontSize: 13, color: '#A98BD9', fontWeight: 600 }}>
                    {Math.round(totalProgress)}%
                  </span>
                </div>

                {/* Progress Bar */}
                <div style={{
                  height: 6,
                  background: '#1a1a1a',
                  borderRadius: 3,
                  overflow: 'hidden'
                }}>
                  <div style={{
                    height: '100%',
                    background: 'linear-gradient(125deg, #A98BD9, #6D28D9, #194DF9)',
                    backgroundSize: '200% 200%',
                    animation: 'gradientShift 3s ease infinite',
                    borderRadius: 3,
                    width: `${totalProgress}%`,
                    transition: 'width 0.3s ease'
                  }} />
                </div>

                {/* Download stats row */}
                <div style={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                  marginTop: 10,
                  fontSize: 12,
                  color: '#a0a0a0'
                }}>
                  {/* Current file */}
                  <span style={{ flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                    {runningDownloadProgress?.current_downloading_file || installMessage}
                  </span>

                  {/* Speed and ETA */}
                  <div style={{ display: 'flex', gap: 16, marginLeft: 16, flexShrink: 0 }}>
                    {runningDownloadProgress && runningDownloadProgress.speed_bytes_per_sec > 0 && (
                      <span style={{ color: '#6D28D9', fontWeight: 500 }}>
                        {(runningDownloadProgress.speed_bytes_per_sec / (1024 * 1024)).toFixed(1)} MB/s
                      </span>
                    )}
                    {runningDownloadProgress && runningDownloadProgress.expected_total_bytes > 0 && (
                      <span>
                        {formatBytes(runningDownloadProgress.total_downloaded_bytes)} / {formatBytes(runningDownloadProgress.expected_total_bytes)}
                      </span>
                    )}
                    {runningDownloadProgress && runningDownloadProgress.speed_bytes_per_sec > 0 && (
                      <span>
                        ETA: {getTimeRemaining()}
                      </span>
                    )}
                  </div>
                </div>

                {/* Files progress */}
                {runningDownloadProgress && runningDownloadProgress.all_files.length > 0 && (
                  <div style={{
                    marginTop: 6,
                    fontSize: 11,
                    color: '#666'
                  }}>
                    File {runningDownloadProgress.completed_files.length + 1} of {runningDownloadProgress.all_files.length}
                  </div>
                )}
              </>
              )}
              </div>
            </div>
          )}

          {/* Step 4: Complete */}
          {currentStep === 'complete' && (
            <div style={{
              flex: 1,
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              textAlign: 'center',
              animation: 'fadeIn 0.25s ease'
            }}>
              {/* Success Icon */}
              <div style={{
                width: 72,
                height: 72,
                background: 'linear-gradient(125deg, #A98BD9, #6D28D9, #194DF9)',
                borderRadius: '50%',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                marginBottom: 20
              }}>
                <svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="2.5">
                  <polyline points="20 6 9 17 4 12"/>
                </svg>
              </div>

              <h2 style={{
                fontFamily: "'Miriam Libre', serif",
                fontSize: 26,
                fontWeight: 700,
                marginBottom: 8
              }}>
                Ready to Create
              </h2>
              <p style={{ color: '#a0a0a0', fontSize: 14, maxWidth: 320 }}>
                LTX Video is installed. Start generating.
              </p>

              {/* Install Summary */}
              <div style={{
                background: '#2e3445',
                borderRadius: 12,
                padding: '16px 28px',
                marginTop: 20,
                textAlign: 'left',
                minWidth: 260
              }}>
                <div style={{
                  display: 'flex',
                  justifyContent: 'space-between',
                  padding: '8px 0',
                  fontSize: 13
                }}>
                  <span style={{ color: '#a0a0a0' }}>Location</span>
                  <span style={{ fontWeight: 500, maxWidth: 150, overflow: 'hidden', textOverflow: 'ellipsis' }}>
                    {installPath.split('\\').pop() || installPath}
                  </span>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Footer */}
        <div style={{
          padding: currentStep === 'installing' ? '12px 24px' : '16px 32px',
          borderTop: '1px solid #1a1a1a',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center'
        }}>
          <div style={{ fontSize: 11, color: '#666' }}>© 2026 ad-astra-video · Fork of Lightricks' LTX Desktop</div>

          <div style={{ display: 'flex', gap: 10 }}>
            {/* Next/Install/Finish Button */}
            {currentStep !== 'installing' && (
              <button
                onClick={() => void handleNext()}
                disabled={isNextDisabled()}
                style={{
                  padding: '10px 28px',
                  borderRadius: 8,
                  fontSize: 13,
                  fontWeight: 700,
                  cursor: isNextDisabled() ? 'not-allowed' : 'pointer',
                  background: isNextDisabled() ? '#555' : '#2B61FF',
                  border: 'none',
                  color: '#ffffff',
                  transition: 'all 0.2s ease',
                  opacity: isNextDisabled() ? 0.6 : 1
                }}
              >
                {getNextButtonText()}
              </button>
            )}
          </div>
        </div>
        {actionError && (
          <div style={{ padding: '0 32px 12px 32px', color: '#fca5a5', fontSize: 12 }}>
            {actionError}
          </div>
        )}
      </div>

    </div>
  )
}
