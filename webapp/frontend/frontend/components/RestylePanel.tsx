import { useState, useEffect, useCallback, useRef, useMemo, forwardRef, useImperativeHandle } from 'react'
import { VideoPreviewPanel } from './VideoPreviewPanel'
import { validateVideoSource } from '../lib/video-constraints'
import { webAssetUrl } from '../lib/file-url'
import { isWebPath, getBlob } from '../lib/runtime/web-store'
import { ApiClient } from '../lib/api-client'
import { resolveRunner, segmentSubjectViaRunner, styleFrameViaRunner } from '../lib/direct-transport'
import { Image, X, Loader2, Check, Film, Wand2, Upload } from 'lucide-react'
import { logger } from '../lib/logger'
import type { Asset } from '../types/project-model'

// Restyle panel for the two-step identity-preserving workflow:
//   1. Drop a video  ->  its first frame is extracted automatically
//   2. A style prompt drives a FLUX.2 [klein] 4B image edit of that frame (via the
//      id-v2v worker's /style-frame) producing a stylized first frame
//   3. The user accepts it  ->  the accepted image becomes the stylized first
//      frame passed to id-v2v /restyle.
// The parent (GenSpace) prefills the main prompt with the default "restyle this
// video" once the stylized frame is accepted. The panel still lets the user drop
// their own stylized image as a direct shortcut.
//
// Unified layout: the source-video preview and the extracted/stylized first frame
// live in a TABBED view (Image | Video). The first RESTYLE step (frame edit) is
// driven from GenSpace's main prompt bar — the panel exposes an imperative handle
// (restyleFrame) that GenSpace calls when the user presses "Restyle Image". There
// is no inline First-Frame settings panel here: the style prompt + FLUX.2 model
// live in the main prompt window.

export const DEFAULT_RESTYLE_PROMPT = 'restyle this video'

/** Read a browser Blob (e.g. a web:// image) as a base64 data payload for the runner. */
function blobToBase64(blob: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const fr = new FileReader()
    fr.onload = () => {
      const url = fr.result
      if (typeof url === 'string') {
        const comma = url.indexOf(',')
        resolve(comma >= 0 ? url.slice(comma + 1) : url)
      } else {
        reject(new Error('Could not read image blob'))
      }
    }
    fr.onerror = () => reject(fr.error ?? new Error('Could not read image blob'))
    fr.readAsDataURL(blob)
  })
}

type RestyleTab = 'image' | 'video'

export interface RestyleFrameOptions {
  prompt: string
  /** Optional per-call seed; omitted -> a fresh rotated seed each run. */
  seed?: number
  /** Run the style prompt through the worker's Gemma LLM (enhance-before-evict). */
  enhance?: boolean
}

export interface RestylePanelHandle {
  /** Run the first step of the restyle workflow (edit of the extracted frame). */
  restyleFrame: (opts: RestyleFrameOptions) => Promise<boolean>
  /** Accept the current candidate/extracted frame as the stylized first frame. */
  acceptCurrent: () => void
}

export interface RestyleFrameState {
  extractedFramePath: string | null
  isStyling: boolean
  canRestyle: boolean
  hasStylized: boolean
}

interface RestylePanelProps {
  initialVideoPath?: string | null
  initialImagePath?: string | null
  // Quick-pick candidates drawn from the project's asset library. Videos load as the
  // restyle source; images set the stylized first-frame directly.
  assets?: Asset[]
  resetKey?: number
  isProcessing?: boolean
  processingStatus?: string
  fillHeight?: boolean
  enforceApiConstraints?: boolean
  /** Controlled active tab (Image | Video). */
  activeTab?: RestyleTab
  onTabChange?: (tab: RestyleTab) => void
  /** Reports first-frame workflow state so the parent can gate/enable "Restyle Image". */
  onStateChange?: (state: RestyleFrameState) => void
  onChange?: (data: {
    videoPath: string | null
    stylizedImagePath: string | null
    ready: boolean
  }) => void
  // Called when the user accepts a generated/first-frame stylized image, with the
  // accepted image path. The parent uses it to prefill the default restyle prompt.
  onAccept?: (acceptedImagePath: string, videoPath: string | null) => void
}

export const RestylePanel = forwardRef<RestylePanelHandle, RestylePanelProps>(function RestylePanel(
  {
    initialVideoPath,
    initialImagePath,
    assets,
    resetKey,
    isProcessing = false,
    processingStatus = '',
    enforceApiConstraints = true,
    activeTab: activeTabProp = 'video',
    onTabChange,
    onStateChange,
    onChange,
    onAccept,
  },
  ref,
) {
  const [videoPath, setVideoPath] = useState<string | null>(initialVideoPath || null)
  const [stylizedImagePath, setStylizedImagePath] = useState<string | null>(initialImagePath || null)
  const [dimensions, setDimensions] = useState<{ width: number; height: number }>({ width: 0, height: 0 })
  const [videoDuration, setVideoDuration] = useState(0)

  // Which preview is shown: the first-frame edit UI or the source video.
  const [activeTab, setActiveTab] = useState<RestyleTab>(activeTabProp)

  // First-frame workflow state.
  const [extractedFramePath, setExtractedFramePath] = useState<string | null>(null)
  // Auto SAM3 segmentation of the subject to keep, run when the first frame is pulled.
  const [segmentingSubject, setSegmentingSubject] = useState(false)
  const [subjectMaskB64, setSubjectMaskB64] = useState<string | null>(null)
  // Every generated first-frame candidate is KEPT so the user can iterate (each
  // re-run rotates the seed) and pick which stylized frame to accept. The active
  // one is what's shown / what Accept uses.
  const [candidates, setCandidates] = useState<string[]>([])
  const [activeCandidate, setActiveCandidate] = useState<string | null>(null)
  const [isStyling, setIsStyling] = useState(false)
  const [isExtracting, setIsExtracting] = useState(false)
  const [stylingError, setStylingError] = useState<string | null>(null)

  const videoKnownRef = useRef<string | null>(null)
  const stylizedInputRef = useRef<HTMLInputElement>(null)

  // Use an existing image file as the stylized first frame directly (skip the
  // generate-&-accept flow). Useful for comparing against authored samples that
  // ship a pre-made stylized frame, or for reusing an earlier frame.
  const handleManualStylized = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file || !file.type.startsWith('image/')) return
    const filePath = window.electronAPI?.getPathForFile(file)
    const path = filePath || URL.createObjectURL(file)
    if (!path) return
    e.target.value = ''
    setStylizedImagePath(path)
    setCandidates([])
    setActiveCandidate(null)
    onAccept?.(path, videoPath)
    setTab('video')
  }

  // Quick-pick: bump the token with a chosen video path to force VideoPreviewPanel
  // to load that source (mirrors drop/browse).
  const [videoRequest, setVideoRequest] = useState<{ path: string; token: number } | null>(null)

  const quickPickVideos = useMemo(
    () => (assets || []).filter(a => a.type === 'video'),
    [assets],
  )
  const quickPickImages = useMemo(
    () => (assets || []).filter(a => a.type === 'image'),
    [assets],
  )

  // Sync the controlled activeTab from the parent (e.g. when entering restyle mode).
  useEffect(() => {
    setActiveTab(activeTabProp)
  }, [activeTabProp])

  const setTab = useCallback((tab: RestyleTab) => {
    setActiveTab(tab)
    onTabChange?.(tab)
  }, [onTabChange])

  const pickQuickVideo = useCallback((path: string) => {
    setVideoRequest(prev => ({ path, token: (prev?.token ?? 0) + 1 }))
  }, [])

  const pickQuickImage = useCallback((asset: Asset) => {
    // Setting the stylized first frame directly (bypasses extract+restyle) and treat
    // it as accepted so the panel is ready to run the restyle.
    setStylizedImagePath(asset.path)
    setCandidates([])
    setActiveCandidate(null)
    setExtractedFramePath(null)
    // Jump to the accepted image first, then let the user switch to the video.
    setTab('image')
    onAccept?.(asset.path, videoPath)
  }, [videoPath, onAccept, setTab])

  const handleSourceChange = useCallback(async (data: { videoPath: string | null; videoDuration: number; width: number; height: number }) => {
    setVideoPath(data.videoPath)
    setVideoDuration(data.videoDuration)
    setDimensions({ width: data.width, height: data.height })

    // New/updated source video: keep the user on the Video tab; they switch to
    // Image to run the first-frame restyle.

    // Auto-extract the first frame whenever a new source video lands.
    if (data.videoPath && data.videoPath !== videoKnownRef.current) {
      videoKnownRef.current = data.videoPath
      setStylizedImagePath(null)
      setCandidates([])
      setActiveCandidate(null)
      setExtractedFramePath(null)
      setSubjectMaskB64(null)
      await extractFirstFrame(data.videoPath)
    }
  }, [setTab])

  const extractFirstFrame = useCallback(async (path: string) => {
    setIsExtracting(true)
    setStylingError(null)
    try {
      let imagePath: string | null = null
      if (isWebPath(path)) {
        // Browser: draw the first frame with the canvas web API (HTMLVideoElement +
        // drawImage) — no local Python / ffmpeg needed. Returns a new web:// blob key.
        const res = await window.electronAPI.extractVideoFrame({ videoPath: path, seekTime: 0 })
        imagePath = res?.path || null
        if (!imagePath) {
          setStylingError('Could not extract the first frame')
          return
        }
      } else {
        // Desktop: extract via the local backend.
        const res = await ApiClient.extractFirstFrame({ video_path: path })
        if (!res.ok) {
          logger.error(`First-frame extraction failed: ${res.error?.message}`)
          setStylingError(res.error?.message ?? 'Could not extract the first frame')
          return
        }
        imagePath = res.data.imagePath
      }
      setExtractedFramePath(imagePath)
      // Auto-segment the subject to keep as soon as the first frame is pulled.
      setSegmentingSubject(true)
      let mask: string | null = null
      try {
        // Browser path: the source image is a web:// blob only the browser can read — a
        // remote runner can't fetch it, so the Worker rail would skip SAM3 ("auto subject
        // segmentation unavailable in browser"). Instead hand the actual bytes to the
        // runner over the direct-transport (paid Livepeer) rail when a sam3 runner is up.
        if (isWebPath(imagePath)) {
          const runner = await resolveRunner(['sam3'])
          if (runner) {
            const blob = getBlob(imagePath)
            if (blob) {
              const seg = await segmentSubjectViaRunner(runner, await blobToBase64(blob))
              mask = seg.maskB64
              logger.info(`SAM3 segmentation via direct runner ${runner.runner_id}`)
            }
          }
        }
      } catch (e2) {
        logger.warn(`Direct SAM3 failed (${e2}); falling back to Worker rail`)
      }
      if (!mask) {
        // Desktop real path (or browser with no direct sam3 runner): Worker rail. For a
        // web:// key with no runner this returns the graceful { ok, skipped } no-op.
        try {
          const segRes = await ApiClient.segmentSubject({ image_path: imagePath })
          if (segRes.ok && segRes.data.mask_b64) mask = segRes.data.mask_b64
        } catch (e3) {
          logger.error(`Subject segmentation failed: ${e3}`)
        }
      }
      setSubjectMaskB64(mask)
      setSegmentingSubject(false)
    } catch (e) {
      logger.error(`First-frame extraction exception: ${e}`)
      setStylingError(e instanceof Error ? e.message : 'Could not extract the first frame')
    } finally {
      setIsExtracting(false)
    }
  }, [])

  // Precedence for what's shown in the first-frame panel: a freshly generated
  // candidate (not yet accepted) > the accepted/stylized image > the raw extracted frame.
  const displayFramePath = activeCandidate || stylizedImagePath || extractedFramePath

  // First step of the restyle workflow: Z-Image edit of the extracted first frame,
  // driven by the style prompt provided by the main prompt bar.
  const restyleFrame = useCallback(async (opts: RestyleFrameOptions): Promise<boolean> => {
    if (!extractedFramePath) return false
    if (!opts.prompt.trim()) return false
    setIsStyling(true)
    setStylingError(null)
    try {
      // Browser path: the extracted frame is a web:// blob a remote runner can't read, so the
      // Worker rail would skip ("style-frame unavailable in browser without asset upload"). Hand
      // the actual bytes to the runner over the direct paid rail when one is up (same as SAM3).
      if (isWebPath(extractedFramePath)) {
        const runner = await resolveRunner(['restyle'])
        const blob = getBlob(extractedFramePath)
        if (runner && blob) {
          try {
            const styled = await styleFrameViaRunner(
              runner,
              await blobToBase64(blob),
              opts.prompt.trim(),
              { seed: opts.seed ?? Math.floor(Math.random() * 2 ** 31), enhance: opts.enhance ?? false },
            )
            setCandidates(prev => prev.includes(styled.styledImageUrl) ? prev : [...prev, styled.styledImageUrl])
            setActiveCandidate(styled.styledImageUrl)
            setTab('image')
            return true
          } catch (e) {
            logger.warn(`Direct style-frame failed (${e}); falling back to Worker rail`)
          }
        }
      }
      // First-frame styling routes to FLUX.2 [klein] 4B on the id-v2v worker
      // (/api/restyle/style-frame). FLUX.2 klein is a fixed 4-step distilled
      // single-reference editor: correct inputs are just the reference frame +
      // prompt + seed (+ optional enhance). No strength / guidance / steps /
      // keep-subject mask (those were Z-Image img2img concepts and do not exist
      // for FLUX.2 klein).
      const res = await ApiClient.styleFirstFrame({
        image_path: extractedFramePath,
        prompt: opts.prompt.trim(),
        seed: opts.seed ?? Math.floor(Math.random() * 2 ** 31),
        enhance_prompt: opts.enhance ?? false,
      })
      if (!res.ok) {
        logger.error(`First-frame restyle failed: ${res.error?.message}`)
        setStylingError(res.error?.message ?? 'Restyle failed')
        return false
      }
      const path = res.data?.imagePath
      if (!path) {
        setStylingError('Restyle produced no image')
        return false
      }
      setCandidates(prev => prev.includes(path) ? prev : [...prev, path])
      // Show the freshly styled candidate.
      setActiveCandidate(path)
      setTab('image')
      return true
    } catch (e) {
      logger.error(`First-frame restyle exception: ${e}`)
      setStylingError(e instanceof Error ? e.message : 'Restyle failed')
      return false
    } finally {
      setIsStyling(false)
    }
  }, [extractedFramePath, setTab])

  const acceptCurrent = useCallback(() => {
    const accepted = activeCandidate || extractedFramePath
    if (!accepted) return
    setStylizedImagePath(accepted)
    setActiveCandidate(null)
    onAccept?.(accepted, videoPath)
    // Once a stylized frame is accepted, flip to the source-video preview.
    setTab('video')
  }, [activeCandidate, extractedFramePath, videoPath, onAccept, setTab])

  useImperativeHandle(ref, () => ({ restyleFrame, acceptCurrent }), [restyleFrame, acceptCurrent])

  // Report first-frame workflow state to the parent.
  useEffect(() => {
    const canRestyle = !!extractedFramePath && !isStyling && !isProcessing
    onStateChange?.({
      extractedFramePath,
      isStyling,
      canRestyle,
      hasStylized: !!stylizedImagePath,
    })
  }, [extractedFramePath, isStyling, stylizedImagePath, isProcessing, onStateChange])

  const error = enforceApiConstraints && videoPath
    ? validateVideoSource({ width: dimensions.width, height: dimensions.height, duration: videoDuration })
    : null

  useEffect(() => {
    if (resetKey === undefined) return
    setStylizedImagePath(initialImagePath || null)
    setCandidates(initialImagePath ? [initialImagePath] : [])
    setActiveCandidate(null)
  }, [resetKey, initialImagePath])

  useEffect(() => {
    onChange?.({
      videoPath,
      stylizedImagePath,
      ready: !!videoPath && !!stylizedImagePath && !error,
    })
  }, [videoPath, stylizedImagePath, error, onChange])

  const tabButton = (tab: RestyleTab, icon: React.ReactNode, label: string, title: string) => (
    <button
      type="button"
      onClick={() => setTab(tab)}
      title={title}
      className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium transition-colors border ${
        activeTab === tab
          ? 'bg-zinc-800 text-white border-zinc-700'
          : 'bg-transparent text-zinc-400 border-transparent hover:text-zinc-200 hover:bg-zinc-800/50'
      }`}
    >
      {icon}
      {label}
    </button>
  )

  return (
    <div className="flex-1 flex flex-col min-h-0 gap-3">
      {/* Quick-pick from the asset library */}
      {(quickPickVideos.length > 0 || quickPickImages.length > 0) && (
        <div className="flex-shrink-0 flex gap-4">
          {quickPickVideos.length > 0 && (
            <div className="flex-1 min-w-0">
              <div className="text-[10px] uppercase tracking-wide text-zinc-500 mb-1">Quick pick video</div>
              <div className="flex gap-2 overflow-x-auto pb-1">
                {quickPickVideos.map(a => (
                  <button
                    key={a.id}
                    onClick={() => pickQuickVideo(a.path)}
                    title={a.path.split(/[/\\]/).pop()}
                    className={`w-24 flex-shrink-0 aspect-video rounded-lg overflow-hidden border-2 transition-colors ${
                      videoPath === a.path ? 'border-emerald-500' : 'border-zinc-800 hover:border-zinc-600'
                    }`}
                  >
                    {a.bigThumbnailPath ? (
                      <img src={webAssetUrl(a.bigThumbnailPath)} alt="" className="w-full h-full object-cover" />
                    ) : (
                      <div className="w-full h-full bg-zinc-800 flex items-center justify-center">
                        <Film className="h-4 w-4 text-zinc-500" />
                      </div>
                    )}
                  </button>
                ))}
              </div>
            </div>
          )}
          {quickPickImages.length > 0 && (
            <div className="flex-1 min-w-0">
              <div className="text-[10px] uppercase tracking-wide text-zinc-500 mb-1">Quick pick image</div>
              <div className="flex gap-2 overflow-x-auto pb-1">
                {quickPickImages.map(a => (
                  <button
                    key={a.id}
                    onClick={() => pickQuickImage(a)}
                    title={a.path.split(/[/\\]/).pop()}
                    className={`w-24 flex-shrink-0 aspect-video rounded-lg overflow-hidden border-2 transition-colors ${
                      stylizedImagePath === a.path ? 'border-emerald-500' : 'border-zinc-800 hover:border-zinc-600'
                    }`}
                  >
                    <img src={webAssetUrl(a.path)} alt="" className="w-full h-full object-cover" />
                  </button>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Unified tabbed view: Image (first-frame edit) | Video (source preview) */}
      <div className="flex-shrink-0 flex items-center gap-1 border-b border-zinc-800 pb-2">
        {tabButton('image', <Image className="h-4 w-4" />, 'Image', 'Stylize the first frame')}
        {tabButton('video', <Film className="h-4 w-4" />, 'Video', 'Preview the source video')}
        <div className="ml-auto flex items-center gap-2">
          {displayFramePath && (
            <span className="text-[10px] text-zinc-500">
              {stylizedImagePath ? 'Stylized frame set' : extractedFramePath ? 'First frame extracted' : ''}
            </span>
          )}
          <button
            type="button"
            onClick={() => stylizedInputRef.current?.click()}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium border border-zinc-700 bg-zinc-800/60 text-zinc-200 hover:bg-zinc-700/60 transition-colors"
            title="Use an existing image as the stylized first frame (skip generation)"
          >
            <Upload className="h-3.5 w-3.5" />
            Add stylized frame…
          </button>
          <input
            ref={stylizedInputRef}
            type="file"
            accept="image/*"
            onChange={handleManualStylized}
            className="hidden"
          />
        </div>
      </div>

      <div className="flex-1 min-h-0">
        {/* VideoPreviewPanel stays mounted (hidden while the Image tab is active) so
            quick-pick / drop source selection is still processed regardless of tab. */}
        <div className={activeTab === 'video' ? 'h-full min-h-0' : 'hidden'}>
          <div className="flex gap-3 h-full min-h-0">
            <div className="flex-1 min-w-0 h-full min-h-0">
              <VideoPreviewPanel
                title="Restyle"
                initialVideoPath={initialVideoPath}
                resetKey={resetKey}
                isProcessing={isProcessing}
                processingStatus={processingStatus}
                processingDefault="Restyling video..."
                fillHeight
                emptyTitle="Drop a video to restyle"
                hint={{ title: 'Restyle your video in a new style' }}
                errorMessage={error ?? undefined}
                onSourceChange={handleSourceChange}
                showFilmstrip={false}
                externalVideoRequest={videoRequest}
              />
            </div>

            {/* Accepted (styled) first frame shown alongside the source video. */}
            <div className="flex-1 min-w-0 h-full min-h-0 flex flex-col justify-center">
              <div className="text-[10px] uppercase tracking-wide text-zinc-500 mb-1">
                {stylizedImagePath ? 'First frame (styled)' : 'First frame'}
              </div>
              <div className="flex-1 min-h-0 rounded-xl border border-zinc-800 bg-black flex items-center justify-center overflow-hidden">
                {displayFramePath ? (
                  <img src={webAssetUrl(displayFramePath)} alt="" className="w-full h-full object-contain" />
                ) : (
                  <div className="flex flex-col items-center gap-2 text-zinc-600 px-3 text-center">
                    <Image className="h-6 w-6" />
                    <span className="text-[11px]">No first frame yet</span>
                  </div>
                )}
              </div>
              {stylizedImagePath && (
                <span className="mt-1 flex items-center gap-1 text-[10px] text-emerald-400">
                  <Check className="h-3 w-3" /> Accepted
                </span>
              )}
            </div>
          </div>
        </div>

        {activeTab === 'image' && (
          <div className="h-full min-h-0 flex flex-col gap-2">
            {/* Main UI: the extracted/stylized first frame */}
            <div className="flex-1 min-h-0 relative rounded-xl border border-zinc-800 bg-black flex items-center justify-center overflow-hidden">
              {isExtracting ? (
                <div className="flex flex-col items-center gap-2 text-zinc-500">
                  <Loader2 className="h-5 w-5 animate-spin" />
                  <span className="text-xs">Extracting first frame...</span>
                </div>
              ) : displayFramePath ? (
                <>
                  <img src={webAssetUrl(displayFramePath)} alt="" className="w-full h-full object-contain" />
                  {subjectMaskB64 && (
                    <img
                      src={`data:image/png;base64,${subjectMaskB64}`}
                      alt=""
                      className="absolute inset-0 w-full h-full object-contain pointer-events-none mix-blend-screen opacity-40"
                    />
                  )}
                </>
              ) : (
                <div className="flex flex-col items-center gap-2 text-zinc-600">
                  <Image className="h-6 w-6" />
                  <span className="text-[11px] px-3 text-center">Drop a video to extract its first frame</span>
                </div>
              )}
              {displayFramePath && !isExtracting && (activeCandidate || stylizedImagePath) && (
                <button
                  onClick={(e) => { e.stopPropagation(); if (activeCandidate) { setCandidates(prev => prev.filter(c => c !== activeCandidate)); setActiveCandidate(null); } else { setStylizedImagePath(null) } }}
                  className="absolute top-1.5 right-1.5 p-1 rounded-full bg-zinc-800/80 text-zinc-400 hover:text-white"
                  title="Clear"
                >
                  <X className="h-3.5 w-3.5" />
                </button>
              )}
              {displayFramePath !== extractedFramePath && displayFramePath && (
                <span className="absolute top-1.5 left-1.5 text-[10px] text-emerald-400 bg-zinc-900/70 rounded px-1.5 py-0.5">
                  {activeCandidate && stylizedImagePath !== activeCandidate ? `candidate · take ${candidates.indexOf(activeCandidate) + 1}` : 'stylized'}
                </span>
              )}
              {isStyling && (
                <div className="absolute inset-0 bg-black/40 flex items-center justify-center">
                  <div className="flex items-center gap-2 text-zinc-200 text-sm">
                    <Loader2 className="h-5 w-5 animate-spin" />
                    <span>Restyling frame...</span>
                  </div>
                </div>
              )}
            </div>

            {/* Minimal footer: segmenting status + accept frame */}
            <div className="flex-shrink-0 flex flex-col gap-2 rounded-2xl border border-zinc-800 bg-zinc-900/60 p-3">
              <div className="flex items-center gap-2">
                <Film className="h-4 w-4 text-zinc-400" />
                <span className="text-xs text-zinc-300 font-medium">
                  {isExtracting ? 'Extracting first frame...' : 'First frame'}
                </span>
                <span className="ml-auto text-[10px] text-zinc-500">
                  Use the prompt bar above to describe the style
                </span>
              </div>

              {(segmentingSubject || subjectMaskB64) && (
                <p className={segmentingSubject ? "text-[10px] text-zinc-300 flex items-center gap-1" : "text-[10px] text-emerald-400"}>
                  {segmentingSubject ? <><Wand2 className="h-3 w-3 animate-pulse" /> Segmenting subject (SAM3)…</> : <>Light region = detected subject, kept unchanged</>}
                </p>
              )}

              {extractedFramePath && (
                <button
                  onClick={acceptCurrent}
                  disabled={isStyling}
                  className="self-start flex items-center gap-1.5 px-3 py-1.5 rounded-md bg-emerald-600 text-white text-xs font-medium hover:bg-emerald-500 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
                >
                  <Check className="h-3.5 w-3.5" />
                  {activeCandidate && stylizedImagePath !== activeCandidate ? `Accept styled frame (take ${candidates.indexOf(activeCandidate) + 1})` : 'Accept frame'}
                </button>
              )}

              {stylingError && (
                <p className="text-[11px] text-red-400">{stylingError}</p>
              )}
              {activeCandidate && stylizedImagePath !== activeCandidate && (
                <p className="text-[10px] text-emerald-400">Each re-run rotates the seed and adds a new take below — pick one, then accept.</p>
              )}
              {/* Kept first-frame candidates: rotate seed per re-run, keep every output. */}
              {candidates.length > 0 && (
                <div className="flex items-center gap-1.5 overflow-x-auto pb-0.5">
                  {candidates.map((c, i) => (
                    <button
                      key={c}
                      onClick={() => setActiveCandidate(c)}
                      title={`take ${i + 1}`}
                      className={`relative w-16 flex-shrink-0 aspect-video rounded-md overflow-hidden border-2 transition-colors ${
                        activeCandidate === c ? 'border-emerald-500' : 'border-zinc-700 hover:border-zinc-500'
                      }`}
                    >
                      <img src={webAssetUrl(c)} alt="" className="w-full h-full object-cover" />
                      <span className="absolute bottom-0 inset-x-0 bg-black/60 text-[8px] text-zinc-300 text-center">#{i + 1}</span>
                    </button>
                  ))}
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  )
})
