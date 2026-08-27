import { useState, useEffect, useCallback, useRef, forwardRef, useImperativeHandle } from 'react'
import { VideoPreviewPanel } from './VideoPreviewPanel'
import { PostProcessControls } from './PostProcessControls'
import { useBerniniEdit, type BerniniEditOutcome } from '../hooks/use-bernini-edit'
import type { RunnerProgressEvent } from '../lib/direct-transport'
import { getBlobUrl, isWebPath, registerFile } from '../lib/runtime/web-store'
import { measureVideoFps } from '../lib/video-fps'
import { Clapperboard, Wand2, Image as ImageIcon, Plus, Zap, Sparkles } from 'lucide-react'
import {
  BERNINI_NATIVE_FPS,
  BERNINI_NATIVE_RESOLUTION,
  type BerniniEngine,
  type BerniniProcessPayload,
  type BerniniResolution,
} from '../lib/bernini-delivery'

// Edit Video master-task panel (Bernini rail). The frontend decides the goal ->
// endpoint (v2v motion-preserving edit | r2v reference-image -> video) with NO
// server-side goal-intent. Source video is edited in place; references (r2v, 1.3B)
// are optional additional images. Above-native delivery is requested via the shared
// PostProcessControls (RIFE fps-boost + FlashVSR upscale). The prompt bar drives the
// edit through the imperative handle: ref.runEdit(prompt, {goal}).

export type EditGoal = 'v2v' | 'r2v'

export interface EditVideoPanelHandle {
  runEdit: (prompt: string, opts?: {
    goal?: EditGoal
    engine?: BerniniEngine
    // Forwarded to submitBerniniEdit so the chat-dock task card can mirror the
    // live edit status (message + progress + step) instead of this panel showing it.
    onProgress?: (ev: RunnerProgressEvent) => void
  }) => Promise<BerniniEditOutcome>
}

interface EditVideoPanelProps {
  initialVideoPath?: string | null
  resetKey?: number
  isProcessing?: boolean
  fillHeight?: boolean
  // The 1.3B engine is the only one natively multi-reference (r2v); 14B rejects >1 ref.
  engine?: BerniniEngine
  onSourceChange?: (data: { videoPath: string | null; ready: boolean }) => void
  onResult?: (videoPath: string) => void
}

export const EditVideoPanel = forwardRef<EditVideoPanelHandle, EditVideoPanelProps>(function EditVideoPanel(
  {
    initialVideoPath,
    resetKey,
    isProcessing = false,
    fillHeight = false,
    engine = '1.3b',
    onSourceChange,
    onResult,
  },
  ref,
) {
  const { submitBerniniEdit, isEditing, berniniEditResult } = useBerniniEdit()
  // Parent's onSourceChange may be an unstable inline callback (a fresh identity every
  // render). Holding it in a ref keeps handleSourceChange stable, so VideoPreviewPanel's
  // onSourceChange effect dep never churns and we avoid a setState-in-effect update loop.
  const onSourceChangeRef = useRef(onSourceChange)
  useEffect(() => {
    onSourceChangeRef.current = onSourceChange
  }, [onSourceChange])

  const [videoPath, setVideoPath] = useState<string | null>(initialVideoPath || null)
  const [goal, setGoal] = useState<EditGoal>('v2v')
  // Engine override: fast (1.3B) | detailed (14B), defaulting to the parent's
  // model prop but user-toggleable per edit. Only 14B r2v is single-reference.
  const [eng, setEng] = useState<BerniniEngine>(engine)
  // Switching to 14B trims multi-refs down to one (14B r2v rejects >1 ref).
  const applyEngine = useCallback((m: BerniniEngine) => {
    setEng(m)
    if (m === '14b') setReferencePaths((prev) => (prev.length > 1 ? prev.slice(0, 1) : prev))
  }, [])
  useEffect(() => { applyEngine(engine) }, [engine, applyEngine])
  // r2v reference images as web:// asset paths (1.3B multi-reference).
  const [referencePaths, setReferencePaths] = useState<string[]>([])
  const referenceInputRef = useRef<HTMLInputElement>(null)
  const [post, setPost] = useState<BerniniProcessPayload>({})
  // Source clip's measured frame rate (best-effort). When > native and the user
  // hasn't set an explicit fps_boost, we deliver at this rate so the edit MATCHES
  // the input (motion-preserving RIFE, not a fixed 16fps output).
  const [sourceFps, setSourceFps] = useState<number | null>(null)

  // Probe the source video's fps whenever the source changes (web:// -> blob URL,
  // or a plain http/blob URL). Pure best-effort: null falls back to native 16fps.
  useEffect(() => {
    let cancelled = false
    const src = videoPath ?? initialVideoPath
    if (!src) {
      setSourceFps(null)
      return
    }
    const url = isWebPath(src) ? getBlobUrl(src) : src
    if (!url || !/^(blob:|https?:|data:)/.test(String(url))) {
      setSourceFps(null)
      return
    }
    measureVideoFps(url).then((fps) => {
      if (!cancelled) setSourceFps(fps)
    })
    return () => {
      cancelled = true
    }
  }, [videoPath, initialVideoPath])

  const effectiveProcessing = isProcessing || isEditing

  useEffect(() => {
    if (berniniEditResult) onResult?.(berniniEditResult.videoPath)
  }, [berniniEditResult, onResult])

  const handleSourceChange = useCallback((data: { videoPath: string | null }) => {
    setVideoPath(data.videoPath)
    onSourceChangeRef.current?.({ videoPath: data.videoPath, ready: !!data.videoPath })
  }, [])

  // r2v reference-image audio/visual: register picked files into the web:// asset
  // store and append their keys. The 14B engine is single-reference (rejects >1),
  // so it replaces the current reference instead of appending.
  const addReferences = useCallback((files: FileList | File[]) => {
    const images = Array.from(files).filter((f) => f.type.startsWith('image/'))
    if (images.length === 0) return
    const keys = images.map(registerFile)
    setReferencePaths((prev) => {
      if (eng === '14b') return keys.length ? [keys[keys.length - 1]] : prev
      return [...prev, ...keys]
    })
  }, [eng])

  const handleReferenceInput = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files
    if (files) addReferences(files)
    e.target.value = ''
  }

  const handleReferenceDrop = (e: React.DragEvent) => {
    e.preventDefault()
    const files = e.dataTransfer?.files
    if (files && files.length) addReferences(files)
  }

  useImperativeHandle(ref, () => ({
    runEdit: async (prompt, opts) => {
      const op = opts?.goal ?? goal
      const effEngine = opts?.engine ?? eng
      // Default the delivery fps to the SOURCE clip's fps so an edit's output matches
      // the input (motion-preserving RIFE boost). An explicit fps_boost (user set it)
      // always wins; otherwise fall back to native 16fps.
      const fps = post.fps_boost?.target_fps
        ?? (sourceFps && sourceFps > BERNINI_NATIVE_FPS ? sourceFps : BERNINI_NATIVE_FPS)
      const resolution = post.upscale ? resolutionForPost(post) : BERNINI_NATIVE_RESOLUTION
      return submitBerniniEdit({
        operation: op,
        videoPath: videoPath ?? initialVideoPath ?? '',
        referencePaths: op === 'r2v' ? referencePaths : [],
        prompt,
        engine: effEngine,
        fps,
        resolution,
        duration: 3,
        onProgress: opts?.onProgress,
      })
    },
  }), [goal, eng, post, videoPath, initialVideoPath, referencePaths, sourceFps, submitBerniniEdit])

  return (
    <div className="flex flex-col gap-3 h-full min-h-0">
      {/* The video preview grows to fill the column; the goal selector + post
          rails below stay at their natural height. Wrapping in a flex-1 child
          (rather than letting VideoPreviewPanel's fillHeight resolve against
          the auto-height root) keeps the filmstrip fully visible above the
          prompt bar — otherwise it overflows the panel and clips under it. */}
      <div className="flex-1 min-h-0">
        <VideoPreviewPanel
          title="Edit Video"
          initialVideoPath={initialVideoPath ?? videoPath}
          resetKey={resetKey}
          fillHeight={fillHeight}
          emptyTitle="Drop a video to edit"
          hint={{ title: 'Pick an edit goal, then describe the edit in the bar below' }}
          onSourceChange={handleSourceChange}
        />
      </div>

      {/* Edit goal + model selector */}
      <div className="flex flex-wrap items-center gap-1.5">
        <button
          type="button"
          onClick={() => setGoal('v2v')}
          className={`flex items-center gap-1.5 px-2.5 py-1.5 rounded-md text-[11px] font-medium transition-colors ${
            goal === 'v2v' ? 'bg-emerald-700/70 text-white' : 'bg-zinc-800 text-zinc-400 hover:bg-zinc-700'
          }`}
        >
          <Wand2 className="h-3.5 w-3.5" />
          Motion-preserving edit
        </button>
        <button
          type="button"
          onClick={() => setGoal('r2v')}
          className={`flex items-center gap-1.5 px-2.5 py-1.5 rounded-md text-[11px] font-medium transition-colors ${
            goal === 'r2v' ? 'bg-emerald-700/70 text-white' : 'bg-zinc-800 text-zinc-400 hover:bg-zinc-700'
          }`}
        >
          <ImageIcon className="h-3.5 w-3.5" />
          Reference-guided
        </button>
        <div className="w-px h-4 bg-zinc-700 mx-0.5" />
        <button
          type="button"
          onClick={() => applyEngine('1.3b')}
          className={`flex items-center gap-1.5 px-2.5 py-1.5 rounded-md text-[11px] font-medium transition-colors ${
            eng === '1.3b' ? 'bg-emerald-700/70 text-white' : 'bg-zinc-800 text-zinc-400 hover:bg-zinc-700'
          }`}
        >
          <Zap className="h-3.5 w-3.5" />
          Fast (1.3B)
        </button>
        <button
          type="button"
          onClick={() => applyEngine('14b')}
          className={`flex items-center gap-1.5 px-2.5 py-1.5 rounded-md text-[11px] font-medium transition-colors ${
            eng === '14b' ? 'bg-emerald-700/70 text-white' : 'bg-zinc-800 text-zinc-400 hover:bg-zinc-700'
          }`}
        >
          <Sparkles className="h-3.5 w-3.5" />
          Detailed (14B)
        </button>
      </div>

      {goal === 'r2v' && (
        <div
          className="rounded-xl border border-zinc-800 bg-zinc-900/60 p-2 flex flex-wrap items-center gap-2"
          onDragOver={(e) => e.preventDefault()}
          onDrop={handleReferenceDrop}
        >
          <Clapperboard className="h-3.5 w-3.5 text-zinc-500" />
          <span className="text-[11px] text-zinc-400">
            Reference images ({eng === '14b' ? '14B · 1' : '1.3B · multi'}):
          </span>
          {referencePaths.map((p, i) => (
            <span key={p} className="relative group flex items-center gap-1 text-[10px] text-zinc-300 bg-zinc-800 rounded-md px-1 py-0.5 pr-5">
              <img
                src={isWebPath(p) ? getBlobUrl(p) : p}
                alt={`Reference ${i + 1}`}
                className="h-6 w-6 rounded object-cover bg-zinc-900"
              />
              Ref {i + 1}
              <button
                type="button"
                aria-label={`Remove reference ${i + 1}`}
                onClick={() => setReferencePaths((prev) => prev.filter((_, idx) => idx !== i))}
                className="absolute top-0.5 right-1 text-zinc-500 hover:text-white"
              >×</button>
            </span>
          ))}
          <button
            type="button"
            onClick={() => referenceInputRef.current?.click()}
            className="flex items-center gap-1 text-[10px] text-zinc-400 hover:text-white bg-zinc-800 hover:bg-zinc-700 rounded-md px-1.5 py-1"
            title="Add reference image (drop an image here too)"
          >
            <Plus className="h-3 w-3" /> Add
          </button>
          <input
            ref={referenceInputRef}
            type="file"
            accept="image/*"
            multiple={eng !== '14b'}
            onChange={handleReferenceInput}
            className="hidden"
          />
        </div>
      )}

      {/* Shared post-process rails */}
      <PostProcessControls value={post} onChange={setPost} disabled={effectiveProcessing} />

    </div>
  )
})

/** Map a post upscale selection to the delivery resolution tier the runner expects. */
function resolutionForPost(post: BerniniProcessPayload): BerniniResolution {
  switch (post.upscale?.final) {
    case 'raw':
      return 'raw-4x'
    case '1440':
      return '1440p'
    case '1080':
    default:
      return '1080p'
  }
}
