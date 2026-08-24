import { useState, useEffect, useCallback, forwardRef, useImperativeHandle } from 'react'
import { VideoPreviewPanel } from './VideoPreviewPanel'
import { PostProcessControls } from './PostProcessControls'
import { useBerniniEdit } from '../hooks/use-bernini-edit'
import { Loader2, Clapperboard, Wand2, Image as ImageIcon } from 'lucide-react'
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
  runEdit: (prompt: string, opts?: { goal?: EditGoal; engine?: BerniniEngine }) => Promise<void>
}

interface EditVideoPanelProps {
  initialVideoPath?: string | null
  resetKey?: number
  isProcessing?: boolean
  processingStatus?: string
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
    processingStatus = '',
    fillHeight = false,
    engine = 'bernini-1.3b',
    onSourceChange,
    onResult,
  },
  ref,
) {
  const { submitBerniniEdit, isEditing, editStatus, editError, berniniEditResult } = useBerniniEdit()
  const [videoPath, setVideoPath] = useState<string | null>(initialVideoPath || null)
  const [goal, setGoal] = useState<EditGoal>('v2v')
  // r2v reference images as web:// asset paths (1.3B multi-reference).
  const [referencePaths, setReferencePaths] = useState<string[]>([])
  const [post, setPost] = useState<BerniniProcessPayload>({})

  const effectiveProcessing = isProcessing || isEditing
  const effectiveStatus = processingStatus || editStatus

  useEffect(() => {
    if (berniniEditResult) onResult?.(berniniEditResult.videoPath)
  }, [berniniEditResult, onResult])

  const handleSourceChange = useCallback((data: { videoPath: string | null }) => {
    setVideoPath(data.videoPath)
    onSourceChange?.({ videoPath: data.videoPath, ready: !!data.videoPath })
  }, [onSourceChange])

  useImperativeHandle(ref, () => ({
    runEdit: async (prompt, opts) => {
      const op = opts?.goal ?? goal
      const eng = opts?.engine ?? engine
      const fps = post.fps_boost?.target_fps ?? BERNINI_NATIVE_FPS
      const resolution = post.upscale ? resolutionForPost(post) : BERNINI_NATIVE_RESOLUTION
      await submitBerniniEdit({
        operation: op,
        videoPath: videoPath ?? initialVideoPath ?? '',
        referencePaths: op === 'r2v' ? referencePaths : [],
        prompt,
        engine: eng,
        fps,
        resolution,
        duration: 3,
      })
    },
  }), [goal, engine, post, videoPath, initialVideoPath, referencePaths, submitBerniniEdit])

  return (
    <div className="flex flex-col gap-3">
      <VideoPreviewPanel
        title="Edit Video"
        initialVideoPath={initialVideoPath ?? videoPath}
        resetKey={resetKey}
        isProcessing={effectiveProcessing}
        processingStatus={effectiveStatus}
        processingDefault="Processing edit..."
        fillHeight={fillHeight}
        emptyTitle="Drop a video to edit"
        hint={{ title: 'Pick an edit goal, then describe the edit in the bar below' }}
        onSourceChange={handleSourceChange}
      />

      {/* Edit goal selector */}
      <div className="flex items-center gap-1.5">
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
      </div>

      {goal === 'r2v' && (
        <div className="rounded-xl border border-zinc-800 bg-zinc-900/60 p-2 flex flex-wrap items-center gap-2">
          <Clapperboard className="h-3.5 w-3.5 text-zinc-500" />
          <span className="text-[11px] text-zinc-400">Reference images (1.3B):</span>
          {referencePaths.map((p, i) => (
            <span key={p} className="flex items-center gap-1 text-[10px] text-zinc-300 bg-zinc-800 rounded px-1.5 py-0.5">
              Ref {i + 1}
              <button
                type="button"
                onClick={() => setReferencePaths((prev) => prev.filter((_, idx) => idx !== i))}
                className="text-zinc-500 hover:text-white"
              >×</button>
            </span>
          ))}
        </div>
      )}

      {/* Shared post-process rails */}
      <PostProcessControls value={post} onChange={setPost} disabled={effectiveProcessing} />

      {isEditing && (
        <div className="flex items-center gap-2 text-[11px] text-zinc-400">
          <Loader2 className="h-3.5 w-3.5 animate-spin" /> {editStatus || 'Editing...'}
        </div>
      )}
      {editError && <p className="text-[11px] text-red-400">{editError}</p>}
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
