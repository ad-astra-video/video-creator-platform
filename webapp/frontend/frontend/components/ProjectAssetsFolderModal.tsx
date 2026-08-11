import { useState } from 'react'
import { FolderOpen, X, CheckCircle2 } from 'lucide-react'
import { Button } from './ui/button'
import { supportsFileSystemAccess } from '../lib/runtime/fs-access'

interface Props {
  isOpen: boolean
  onClose: () => void
  /** Called when the user successfully picks (or is already using) a folder. */
  onChosen: (path: string) => void
}

/**
 * First-run prompt (shown right after license acceptance in the web app) asking the user
 * to choose where project assets are saved on disk, via the File System Access API
 * (showDirectoryPicker). The chosen folder handle is persisted so future visits only
 * re-grant permission. Assets stay on the user's disk — never uploaded.
 */
export function ProjectAssetsFolderModal({ isOpen, onClose, onChosen }: Props) {
  const [busy, setBusy] = useState(false)
  const [folderName, setFolderName] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)

  if (!isOpen) return null

  const supported = supportsFileSystemAccess()

  const handleChoose = async () => {
    setBusy(true)
    setError(null)
    try {
      const result = await window.electronAPI.openProjectAssetsPathChangeDialog()
      if (result.success) {
        const name = result.path.replace(/^web:\/\/project-assets\//, '') || 'selected folder'
        setFolderName(name)
        onChosen(result.path)
      } else {
        // User cancelled — keep the modal open so they can try again or skip.
        setError(null)
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="fixed inset-0 z-[70] flex items-center justify-center bg-black/70 backdrop-blur-sm p-4">
      <div className="w-full max-w-md rounded-xl border border-zinc-700 bg-zinc-900 text-zinc-100 shadow-2xl">
        <div className="flex items-center justify-between border-b border-zinc-800 px-5 py-4">
          <h3 className="flex items-center gap-2 text-base font-semibold">
            <FolderOpen className="h-5 w-5 text-primary" />
            Where should project assets be saved?
          </h3>
          <button
            onClick={onClose}
            className="rounded-md p-1 text-zinc-400 hover:text-white hover:bg-zinc-800"
            aria-label="Close"
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        <div className="px-5 py-4">
          <p className="text-sm text-zinc-300">
            Video clips, images and project files are stored in a folder you choose on your computer.
            Everything stays local to that folder — nothing is uploaded.
          </p>

          {!supported && (
            <p className="mt-3 rounded-md border border-amber-700/50 bg-amber-950/40 px-3 py-2 text-sm text-amber-200">
              Your browser doesn't support choosing a folder directly. Use <b>Chrome, Edge or Opera</b> to pick a
              folder, or skip for now and set it later in Settings.
            </p>
          )}

          {error && <p className="mt-3 text-sm text-red-400">{error}</p>}

          {folderName ? (
            <div className="mt-4 flex items-center gap-2 rounded-md border border-emerald-700/50 bg-emerald-950/40 px-3 py-2 text-sm text-emerald-200">
              <CheckCircle2 className="h-4 w-4" />
              <span>
                Project assets will be saved to <b>{folderName}</b>
              </span>
            </div>
          ) : (
            <div className="mt-4 rounded-md border border-zinc-700 bg-zinc-800/60 px-3 py-2 text-sm text-zinc-400">
              No folder selected yet.
            </div>
          )}
        </div>

        <div className="flex justify-end gap-2 border-t border-zinc-800 px-5 py-4">
          <Button variant="ghost" onClick={onClose}>
            {folderName ? 'Done' : 'Skip for now'}
          </Button>
          <Button onClick={handleChoose} disabled={busy || !supported}>
            {busy ? 'Opening…' : 'Choose Folder'}
          </Button>
        </div>
      </div>
    </div>
  )
}
