import { useState } from 'react'
import { Film, ChevronLeft, ChevronRight } from 'lucide-react'
import { webAssetUrl } from '../lib/file-url'
import type { Asset } from '../types/project-model'

/**
 * Shared collapsible quick-pick rail for the Edit master-task subtasks (Edit
 * Video, Restyle, Retake, Extend, IC-LoRA). It owns the left asset sidebar that
 * used to live inside RestylePanel and hosts the active edit-task panel on the
 * right. The parent wires selection per active mode:
 *   - onPickVideo: pick the subtask's source video (assets[].type === 'video')
 *   - onPickImage: restyle-only — set the stylized first frame
 *   - showImages: only Restyle surface an Images section today.
 */
export interface EditTaskContainerProps {
  assets?: Asset[]
  activeVideoPath?: string | null
  activeImagePath?: string | null
  /** Show the Images section (only Restyle uses a stylized frame today). */
  showImages?: boolean
  onPickVideo: (asset: Asset) => void
  onPickImage?: (asset: Asset) => void
  children: React.ReactNode
}

export const EditTaskContainer = ({
  assets = [],
  activeVideoPath,
  activeImagePath,
  showImages = false,
  onPickVideo,
  onPickImage,
  children,
}: EditTaskContainerProps) => {
  const [quickPickOpen, setQuickPickOpen] = useState(true)
  const quickPickVideos = assets.filter(a => a.type === 'video')
  const quickPickImages = showImages ? assets.filter(a => a.type === 'image') : []

  return (
    <div className="flex-1 flex min-h-0 gap-3">
      {/* Collapsible quick-pick asset sidebar (left) */}
      <div
        className={`flex-shrink-0 flex flex-col min-h-0 rounded-xl border border-zinc-800 bg-zinc-900/40 overflow-hidden transition-all ${
          quickPickOpen ? 'w-40' : 'w-9'
        }`}
      >
        <div className="flex items-center justify-between px-1.5 py-1.5 flex-shrink-0">
          {quickPickOpen && (
            <span className="text-[10px] uppercase tracking-wide text-zinc-500 truncate">Quick pick</span>
          )}
          <button
            type="button"
            onClick={() => setQuickPickOpen(o => !o)}
            title={quickPickOpen ? 'Hide quick-pick assets' : 'Show quick-pick assets'}
            className="p-1 rounded-md text-zinc-400 hover:text-white hover:bg-zinc-800 transition-colors"
          >
            {quickPickOpen ? <ChevronLeft className="h-3.5 w-3.5" /> : <ChevronRight className="h-3.5 w-3.5" />}
          </button>
        </div>
        {quickPickOpen && (
          <div className="flex-1 min-h-0 overflow-y-auto flex flex-col gap-2 px-1.5 pb-2">
            {quickPickVideos.length > 0 && (
              <div className="flex flex-col gap-1">
                <div className="text-[10px] uppercase tracking-wide text-zinc-500 px-0.5">Videos</div>
                <div className="grid grid-cols-2 gap-1.5">
                  {quickPickVideos.map(a => (
                    <button
                      key={a.id}
                      onClick={() => onPickVideo(a)}
                      title={a.path.split(/[/\\]/).pop()}
                      className={`aspect-video rounded-lg overflow-hidden border-2 transition-colors ${
                        activeVideoPath === a.path ? 'border-emerald-500' : 'border-zinc-800 hover:border-zinc-600'
                      }`}
                    >
                      {a.path ? (
                        <video
                          src={webAssetUrl(a.path)}
                          muted
                          playsInline
                          preload="auto"
                          disablePictureInPicture
                          className="w-full h-full object-cover"
                        />
                      ) : (
                        <div className="w-full h-full bg-zinc-800 flex items-center justify-center">
                          <Film className="h-3.5 w-3.5 text-zinc-500" />
                        </div>
                      )}
                    </button>
                  ))}
                </div>
              </div>
            )}
            {quickPickImages.length > 0 && (
              <div className="flex flex-col gap-1">
                <div className="text-[10px] uppercase tracking-wide text-zinc-500 px-0.5">Images</div>
                <div className="grid grid-cols-2 gap-1.5">
                  {quickPickImages.map(a => (
                    <button
                      key={a.id}
                      onClick={() => onPickImage?.(a)}
                      title={a.path.split(/[/\\]/).pop()}
                      className={`aspect-video rounded-lg overflow-hidden border-2 transition-colors ${
                        activeImagePath === a.path ? 'border-emerald-500' : 'border-zinc-800 hover:border-zinc-600'
                      }`}
                    >
                      <img src={webAssetUrl(a.path)} alt="" className="w-full h-full object-cover" />
                    </button>
                  ))}
                </div>
              </div>
            )}
            {quickPickVideos.length === 0 && quickPickImages.length === 0 && (
              <div className="text-[10px] text-zinc-600 px-1">No assets yet</div>
            )}
          </div>
        )}
      </div>

      {/* Main content column: the active edit-task panel */}
      <div className="flex-1 min-w-0 min-h-0 flex flex-col overflow-hidden">{children}</div>
    </div>
  )
}
