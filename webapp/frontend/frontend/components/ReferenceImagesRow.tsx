import { useState, useRef, useEffect } from 'react'
import { webAssetUrl, pathToFileUrl } from '../lib/file-url'
import { isWebPath } from '../lib/runtime/web-store'
import type { Asset } from '../types/project-model'
import { Image as ImageIcon, Plus, X, ChevronDown } from 'lucide-react'

/** Resolve an asset path to something an <img> can render (web:// key or disk path). */
function srcFor(path: string): string {
  return isWebPath(path) ? webAssetUrl(path) : pathToFileUrl(path)
}

/**
 * ReferenceImagesRow — a horizontal strip shown directly above the prompt bar when an
 * image edit or a restyle first-frame edit is active. Lets the user attach 1..n EXTRA
 * reference images (in addition to the primary edit target) for multi-image edits.
 *
 * The runner engine honors the list for Qwen-Image-Edit-2511 / HiDream-O1
 * (multi-reference conditioning); masked / klein / z-image edits are single-image
 * and simply ignore the extras.
 */
export function ReferenceImagesRow({
  primaryKey,
  references,
  assets,
  onAdd,
  onRemove,
}: {
  /** web:// key (or resolvable path) of the primary edit target (shown for context, not addable). */
  primaryKey?: string | null
  /** Paths of the extra reference images currently attached (web:// keys). */
  references: string[]
  /** Project assets to offer as candidate references (image-type assets only are shown). */
  assets?: Asset[] | null
  onAdd: (path: string) => void
  onRemove: (path: string) => void
}) {
  const [open, setOpen] = useState(false)
  const popRef = useRef<HTMLDivElement>(null)

  // Click-outside closes the add-dropdown.
  useEffect(() => {
    if (!open) return
    const onClick = (e: MouseEvent) => {
      if (popRef.current && !popRef.current.contains(e.target as Node)) setOpen(false)
    }
    document.addEventListener('mousedown', onClick)
    return () => document.removeEventListener('mousedown', onClick)
  }, [open])

  const candidates = (assets ?? []).filter(
    a => a.type === 'image' && !!a.path && a.path !== primaryKey && !references.includes(a.path),
  )

  return (
    <div className="flex items-center gap-2 px-1 py-1.5">
      <span className="flex shrink-0 items-center gap-1.5 text-[11px] text-zinc-400 font-medium">
        <ImageIcon className="h-3.5 w-3.5" />
        Reference images
        <span className="text-zinc-600 font-normal hidden md:inline">(qwen-edit / hidream multi-image)</span>
      </span>

      {/* Attached references */}
      {references.map(p => (
        <div key={p} className="relative group shrink-0">
          <img
            src={srcFor(p)}
            alt="reference"
            className="h-9 w-9 rounded-md border border-zinc-700 bg-zinc-900 object-cover"
          />
          <button
            type="button"
            onClick={() => onRemove(p)}
            aria-label="Remove reference image"
            className="absolute -top-1.5 -right-1.5 h-4 w-4 rounded-full bg-zinc-800 border border-zinc-600 text-zinc-300 hover:bg-red-500 hover:text-white flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity"
          >
            <X className="h-2.5 w-2.5" />
          </button>
        </div>
      ))}

      {/* Add-reference dropdown */}
      <div ref={popRef} className="relative shrink-0">
        <button
          type="button"
          onClick={() => setOpen(o => !o)}
          disabled={candidates.length === 0}
          className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-md border border-dashed border-zinc-700 text-zinc-300 hover:bg-zinc-800 hover:text-white text-xs font-medium disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
          title={candidates.length ? 'Add a reference image from project assets' : 'No more image assets to add'}
        >
          <Plus className="h-3.5 w-3.5" />
          Add reference
          <ChevronDown className={`h-3 w-3 transition-transform ${open ? 'rotate-180' : ''}`} />
        </button>
        {open && (
          <div className="absolute bottom-full mb-1.5 left-0 w-56 max-h-72 overflow-y-auto rounded-md border border-zinc-700 bg-zinc-800 shadow-xl z-[9999] py-1">
            <div className="px-3 pt-1.5 pb-1 text-[10px] text-zinc-500 uppercase tracking-wider">Project images</div>
            {candidates.map(a => (
              <button
                key={a.path}
                type="button"
                onClick={() => { onAdd(a.path); setOpen(false) }}
                className="w-full flex items-center gap-2.5 px-2.5 py-1.5 text-left text-sm text-zinc-300 hover:bg-zinc-700 hover:text-white transition-colors"
              >
                <img src={srcFor(a.path)} alt="" className="h-8 w-8 rounded object-cover bg-zinc-900" />
                <span className="truncate">{a.path.split('/').pop() || 'image'}</span>
              </button>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
