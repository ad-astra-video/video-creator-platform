import { useState, useCallback, useImperativeHandle, forwardRef, useEffect, useRef } from 'react'
import { createPortal } from 'react-dom'
import { webAssetUrl } from '../lib/file-url'
import { getBlob, isWebPath } from '../lib/runtime/web-store'
import { resolveRunner, layerImageViaRunner, editImageViaRunner, styleFrameViaRunner, suggestLayersViaRunner, sam3SelectViaRunner } from '../lib/direct-transport'
import type { LayerRunProgress } from '../lib/direct-transport'
import type { LayerPreview } from '../lib/direct-transport'
import { Layers, Loader2, X, Image as ImageIcon, ChevronDown, Wand2 } from 'lucide-react'
import { logger } from '../lib/logger'

/**
 * ImageEditPanel — layered / region-selective image editor.
 *
 * Shows the image and lets the user decompose it into semantic RGBA layers
 * (Qwen-Image-Layered via /video-creator/v1/layer, alpha channels requested) and
 * select which layers/regions to change. The actual edit is driven externally by
 * the prompt bar through the imperative handle: ref.runEdit(prompt) runs a MASKED
 * selective edit (Qwen-Image-Edit with mask_images) when layers are selected, or a
 * whole-frame edit otherwise. FLUX.2 klein is also supported as a whole-frame
 * style pass (it cannot take a mask).
 *
 * No inline prompt / model dropdowns here — the edit models are Qwen-Image-Edit and
 * FLUX.2 klein, and the prompt comes from the surrounding UI (RestylePanel's restyle
 * step, or GenSpace's main prompt bar).
 */

export type ImageEditEngine = 'qwen-edit' | 'klein' | 'zimage' | 'hidream'

export interface ImageEditCompleteMeta {
  prompt: string
  engine: ImageEditEngine
  /** Seed that produced the result (for reproducible takes). */
  seed?: number
  /** True when a layer mask constrained the edit to selected regions. */
  masked: boolean
}

export interface ImageEditPanelProps {
  /** web:// key (or resolvable asset path) of the image being edited. */
  imageKey: string | null
  /** Default edit model (Qwen-Image-Edit | FLUX.2 klein). */
  defaultEngine?: ImageEditEngine
  /** Called when an edit produces a new image. */
  onEditComplete?: (newImageKey: string, meta: ImageEditCompleteMeta) => void
  onActiveChange?: (info: { canEdit: boolean; editing: boolean; masked: boolean }) => void
  /** Optional external mask (data-URL PNG) to overlay on the image alongside the
   *  internal layer/SAM selection mask (e.g. RestylePanel's keep-subject SAM3 mask). */
  overlayMask?: string | null
}

export interface ImageEditPanelHandle {
  /** Run an edit of the layer-selected regions (or whole frame) with the given prompt. */
  runEdit: (prompt: string, opts?: { engine?: ImageEditEngine; seed?: number; strength?: number; paddingMaskCrop?: number; enhance?: boolean; quality?: 'fast' | 'balanced' | 'high'; onProgress?: (p: { step: number; totalSteps: number }) => void }) => Promise<boolean>
  /** True when one or more layers are selected, so the next edit will be masked. */
  hasMaskedSelection: () => boolean
  /** Clear layers / selection / mask. */
  clear: () => void
}

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

/** Invert a base64 white-on-black mask PNG (white <-> black) so the edit targets
 * everything EXCEPT the originally-selected item. Returns the inverted PNG b64. */
async function invertBase64Mask(b64: string): Promise<string> {
  const data = await decodeMaskToImageData(b64)
  const { width, height } = data
  const out = new Uint8ClampedArray(width * height * 4)
  for (let i = 0; i < width * height; i++) {
    let mv = data.data[i * 4]
    mv = 255 - mv
    out[i * 4] = mv
    out[i * 4 + 1] = mv
    out[i * 4 + 2] = mv
    out[i * 4 + 3] = 255
  }
  const canvas = document.createElement('canvas')
  canvas.width = width
  canvas.height = height
  const ctx = canvas.getContext('2d')!
  ctx.putImageData(new ImageData(out, width, height), 0, 0)
  return canvas.toDataURL('image/png').split(',')[1]
}

function decodeMaskToImageData(b64: string): Promise<ImageData> {
  return new Promise((resolve, reject) => {
    const img = new Image()
    img.onload = () => {
      const c = document.createElement('canvas')
      c.width = img.naturalWidth
      c.height = img.naturalHeight
      const ctx = c.getContext('2d', { willReadFrequently: true })
      if (!ctx) { reject(new Error('no 2d context')); return }
      ctx.drawImage(img, 0, 0)
      resolve(ctx.getImageData(0, 0, c.width, c.height))
    }
    img.onerror = () => reject(new Error('Failed to decode a layer mask'))
    img.src = `data:image/png;base64,${b64}`
  })
}

/**
 * Combine the alpha masks of the selected layers into a single edit mask.
 * Each layer's alpha_b64 has its RGB == alpha (white = edit region). We take the
 * max over selected layers, optionally invert (edit everything EXCEPT those
 * regions), and emit an opaque white-on-black PNG base64 suited to Qwen mask_images.
 */
async function buildMask(alphaB64List: string[], invert: boolean): Promise<string | null> {
  if (alphaB64List.length === 0) return null
  const datas = await Promise.all(alphaB64List.map(decodeMaskToImageData))
  const width = datas[0].width
  const height = datas[0].height
  const out = new Uint8ClampedArray(width * height * 4)
  const N = datas.length
  for (let i = 0; i < width * height; i++) {
    let mv = 0
    for (let d = 0; d < N; d++) {
      const r = datas[d].data[i * 4] // RGB holds the mask value
      if (r > mv) mv = r
    }
    if (invert) mv = 255 - mv
    out[i * 4] = mv
    out[i * 4 + 1] = mv
    out[i * 4 + 2] = mv
    out[i * 4 + 3] = 255
  }
  const canvas = document.createElement('canvas')
  canvas.width = width
  canvas.height = height
  const ctx = canvas.getContext('2d')!
  ctx.putImageData(new ImageData(out, width, height), 0, 0)
  return canvas.toDataURL('image/png').split(',')[1]
}

// Layer decomposition quality presets -> Qwen-Image-Layered denoise steps.
const LAYER_QUALITY_STEPS: Record<'fast' | 'balanced' | 'detailed', number> = {
  fast: 25,
  balanced: 30,
  detailed: 50,
}
export type LayerQuality = keyof typeof LAYER_QUALITY_STEPS

// Display label for each layer-quality preset.
const LAYER_QUALITY_LABEL: Record<LayerQuality, string> = {
  fast: 'Fast',
  balanced: 'Balanced',
  detailed: 'Detailed',
}

export const ImageEditPanel = forwardRef<ImageEditPanelHandle, ImageEditPanelProps>(function ImageEditPanel(
  {
    imageKey,
    defaultEngine = 'qwen-edit',
    onEditComplete,
    onActiveChange,
    overlayMask,
  },
  ref,
) {
  const [layers, setLayers] = useState<LayerPreview[]>([])
  const [selected, setSelected] = useState<number[]>([])
  const [invertMask, setInvertMask] = useState(false)
  const [layerCount, setLayerCount] = useState(4)
  const [layerQuality, setLayerQuality] = useState<LayerQuality>('balanced')
  const [layerProgress, setLayerProgress] = useState<LayerRunProgress | null>(null)
  const [qualityOpen, setQualityOpen] = useState(false)
  const qualityBtnRef = useRef<HTMLButtonElement | null>(null)
  const [qualityMenuPos, setQualityMenuPos] = useState<{ top: number; left: number } | null>(null)
  const [layerSuggestion, setLayerSuggestion] = useState<number | null>(null)
  const [decomposing, setDecomposing] = useState(false)
  const [layerError, setLayerError] = useState<string | null>(null)
  const [editing, setEditing] = useState(false)
  const [editError, setEditError] = useState<string | null>(null)
  const [maskPreview, setMaskPreview] = useState<string | null>(null)
  // SAM3 item selection (text-prompt object mask) -> feeds runEdit as a mask.
  const [samPrompt, setSamPrompt] = useState('')
  const [samMaskB64, setSamMaskB64] = useState<string | null>(null)
  const [samSelecting, setSamSelecting] = useState(false)
  const [samError, setSamError] = useState<string | null>(null)

  const selectedLayers = useCallback(
    () => layers.filter(l => selected.includes(l.index) && l.alphaB64),
    [layers, selected],
  )

  const computeMask = useCallback(async (sel: number[], inv: boolean): Promise<string | null> => {
    const alphas = layers.filter(l => sel.includes(l.index) && l.alphaB64).map(l => l.alphaB64!)
    if (alphas.length === 0) return null
    return buildMask(alphas, inv)
  }, [layers])

  const refreshMaskPreview = useCallback(async (sel: number[], inv: boolean) => {
    try {
      const m = await computeMask(sel, inv)
      setMaskPreview(m ? `data:image/png;base64,${m}` : null)
    } catch (e) {
      logger.warn(`mask preview failed: ${e}`)
      setMaskPreview(null)
    }
  }, [computeMask])

  // Send the image through Qwen-Image-Layered and reveal the per-region layer previews.
  const showLayers = useCallback(async () => {
    if (!imageKey || !isWebPath(imageKey)) {
      setLayerError('Decompose needs a web:// image (drop or extract one first)')
      return
    }
    const blob = getBlob(imageKey)
    if (!blob) { setLayerError('Image bytes are unavailable to decompose'); return }
    setDecomposing(true)
    setLayerError(null)
    setLayerProgress(null)
    try {
      const runner = await resolveRunner(['layer'])
      if (!runner) { setLayerError('No runner with the layer capability is available right now.'); return }
      const steps = LAYER_QUALITY_STEPS[layerQuality]
      const res = await layerImageViaRunner(runner, await blobToBase64(blob), layerCount, {
        previewOnly: false,
        numSteps: steps,
        onProgress: (p) => setLayerProgress(p),
      })
      setLayers(res.layers)
      setSelected([])
      setMaskPreview(null)
      setInvertMask(false)
      logger.info(`Decomposed image into ${res.layers.length} layers (${steps} steps) via ${runner.runner_id}`)
    } catch (e) {
      logger.error(`Layer decomposition failed: ${e}`)
      setLayerError(e instanceof Error ? e.message : 'Layer decomposition failed')
    } finally {
      setDecomposing(false)
    }
  }, [imageKey, layerCount, layerQuality])

  const toggleLayer = useCallback((index: number) => {
    setSelected(prev => {
      const next = prev.includes(index) ? prev.filter(i => i !== index) : [...prev, index]
      void refreshMaskPreview(next, invertMask)
      return next
    })
  }, [invertMask, refreshMaskPreview])

  const toggleInvert = useCallback((v: boolean) => {
    setInvertMask(v)
    void refreshMaskPreview(selected, v)
  }, [selected, refreshMaskPreview])

  // Select an item in the image via SAM3 (text prompt -> object mask). The mask is
  // stored so the next edit is constrained to (or inverted away from) that item.
  const selectItem = useCallback(async () => {
    if (!imageKey || !isWebPath(imageKey)) {
      setSamError('Select item needs a web:// image (drop or extract one first)')
      return
    }
    const blob = getBlob(imageKey)
    if (!blob) { setSamError('Image bytes are unavailable to select from'); return }
    const prompt = samPrompt.trim()
    if (!prompt) { setSamError('Enter what to select (e.g. \"the red chair\")'); return }
    setSamSelecting(true)
    setSamError(null)
    try {
      const runner = await resolveRunner(['sam3'])
      if (!runner) { setSamError('No runner with the sam3 capability is available right now.'); return }
      const res = await sam3SelectViaRunner(runner, await blobToBase64(blob), prompt)
      setSamMaskB64(res.maskB64)
      setMaskPreview(webAssetUrl(res.maskKey))
      setLayers([])      // item selection supersedes any layer decomposition
      setSelected([])
      setInvertMask(false)
      logger.info(`SAM3 selected item with prompt "${prompt}" on ${runner.runner_id}`)
    } catch (e) {
      logger.error(`SAM3 selection failed: ${e}`)
      setSamError(e instanceof Error ? e.message : 'SAM3 selection failed')
    } finally {
      setSamSelecting(false)
    }
  }, [imageKey, samPrompt])

  const clear = useCallback(() => {
    setLayers([])
    setSelected([])
    setMaskPreview(null)
    setInvertMask(false)
    setSamMaskB64(null)
    setSamPrompt('')
  }, [])

  const hasMaskedSelection = useCallback(() => selectedLayers().length > 0, [selectedLayers])

  const runEdit = useCallback(async (
    p: string,
    opts?: { engine?: ImageEditEngine; seed?: number; strength?: number; paddingMaskCrop?: number; enhance?: boolean; quality?: 'fast' | 'balanced' | 'high'; onProgress?: (p: { step: number; totalSteps: number }) => void },
  ): Promise<boolean> => {
    if (!imageKey) return false
    if (!p.trim()) return false
    const blob = getBlob(imageKey)
    if (!blob) return false
    const eng = opts?.engine ?? defaultEngine
    setEditing(true)
    setEditError(null)
    try {
      // The edit mask is either a SAM3 item mask (white = item, inverted when the
      // "Invert" toggle is on so the edit touches everything EXCEPT that item) or
      // the union of selected layer alphas. SAM3 selection takes priority.
      let maskB64: string | null = null
      if (samMaskB64) {
        maskB64 = invertMask ? await invertBase64Mask(samMaskB64) : samMaskB64
      } else if (eng === 'qwen-edit' && selectedLayers().length > 0) {
        maskB64 = await computeMask(selected, invertMask)
      }
      // Qwen-Image-Edit on the pinned diffusers (0.39.0) cannot take a mask, so any
      // masked edit (SAM3 item selection, or a layer mask) is routed through Z-Image's
      // inpaint pipeline, which accepts mask_image. White mask = the pixel region to
      // regenerate (the SAM3-derived mask is white = selected item; "Invert" flips it
      // so the edit touches everything EXCEPT the item).
      const maskedEdit = !!(maskB64 || samMaskB64)
      const runner = eng === 'klein' && !maskedEdit ? await resolveRunner(['restyle']) : await resolveRunner(['edit'])
      if (!runner) { setEditError('No capable runner is available for this edit right now.'); return false }
      const b64 = await blobToBase64(blob)
      const seed = opts?.seed ?? Math.floor(Math.random() * 2 ** 31)
      let newKey: string
      if (maskedEdit) {
        // Qwen-Image-Edit Inpaint: white mask = repaint, black = pixel-preserved
        // (hard-composited back over the source). strength = how aggressively the
        // masked region is regenerated. paddingMaskCrop gives context around small
        // objects.
        const r = await editImageViaRunner(
          runner,
          b64,
          p.trim(),
          { engine: 'qwen-edit', seed, strength: opts?.strength ?? 0.55, paddingMaskCrop: opts?.paddingMaskCrop ?? 0, enhance: opts?.enhance ?? false, maskImage: maskB64!, quality: opts?.quality, onProgress: opts?.onProgress },
        )
        newKey = r.imageUrl
      } else if (eng === 'klein') {
        // FLUX.2 klein is a whole-frame single-reference style edit (no mask).
        const r = await styleFrameViaRunner(runner, b64, p.trim(), { seed, enhance: opts?.enhance ?? false, quality: opts?.quality })
        newKey = r.styledImageUrl
      } else if (eng === 'hidream') {
        // HiDream-O1-Image is a whole-frame instruction edit (no mask): it takes
        // a single reference image + an edit instruction and regenerates pixels
        // directly (no VAE). Runs through the image-worker /edit with engine=hidream.
        const r = await editImageViaRunner(
          runner,
          b64,
          p.trim(),
          { engine: 'hidream', seed, enhance: opts?.enhance ?? false, maskImage: undefined, quality: opts?.quality, onProgress: opts?.onProgress },
        )
        newKey = r.imageUrl
      } else {
        const r = await editImageViaRunner(
          runner,
          b64,
          p.trim(),
          { engine: 'qwen-edit', seed, enhance: opts?.enhance ?? false, maskImage: undefined, quality: opts?.quality, onProgress: opts?.onProgress },
        )
        newKey = r.imageUrl
      }
      onEditComplete?.(newKey, { prompt: p.trim(), engine: maskedEdit ? 'qwen-edit' : eng, seed, masked: maskedEdit })
      return true
    } catch (e) {
      logger.error(`Image edit failed: ${e}`)
      setEditError(e instanceof Error ? e.message : 'Image edit failed')
      return false
    } finally {
      setEditing(false)
    }
  }, [imageKey, defaultEngine, selectedLayers, computeMask, selected, invertMask, samMaskB64, onEditComplete])

  useImperativeHandle(ref, () => ({ runEdit, hasMaskedSelection, clear }), [runEdit, hasMaskedSelection, clear])

  // Reset layers whenever the edited image changes.
  useEffect(() => { clear() }, [imageKey, clear])

  // Auto-suggest a layer count for the newly loaded image via the multimodal Gemma
  // agent (2-8 per the Qwen-Image-Layered rubric). Non-blocking + best-effort: the
  // user can still set the count manually, and a stale/failed suggestion is ignored.
  useEffect(() => {
    let cancelled = false
    const blob = imageKey && isWebPath(imageKey) ? getBlob(imageKey) : null
    if (!blob) return
    ;(async () => {
      setLayerSuggestion(null)
      try {
        const runner = await resolveRunner(['suggest-layers'])
        if (!runner || cancelled) return
        const sug = await suggestLayersViaRunner(runner, await blobToBase64(blob))
        if (cancelled) return
        const n = sug.layers
        if (n != null && n >= 2 && n <= 8) {
          setLayerSuggestion(n)
          setLayerCount(n)
          logger.info(`Gemma suggests ${n} layers for this image`)
        }
      } catch (e) {
        logger.debug(`Layer suggestion skipped: ${e instanceof Error ? e.message : e}`)
      }
    })()
    return () => { cancelled = true }
  }, [imageKey])

  useEffect(() => {
    onActiveChange?.({ canEdit: !!imageKey, editing, masked: hasMaskedSelection() })
  }, [imageKey, editing, hasMaskedSelection, onActiveChange])

  const layerLabel = (index: number, total: number) =>
    total <= 3
      ? index === 0 ? 'Foreground' : index === total - 1 ? 'Background' : 'Midground'
      : index === 0 ? 'Foreground' : index === total - 1 ? 'Background' : `Mid ${index}`

  return (
    <div className="flex flex-col gap-2 min-h-0">
      {/* Preview with the live mask overlay + always-visible floating edit-status bar */}
      <div className="relative flex-1 min-h-0 rounded-xl border border-zinc-800 bg-black flex items-center justify-center overflow-hidden">
        {imageKey ? (
          <>
            <img src={webAssetUrl(imageKey)} alt="" className="w-full h-full object-contain" />
            {maskPreview && (
              <img
                src={maskPreview}
                alt=""
                className="absolute inset-0 w-full h-full object-contain pointer-events-none mix-blend-screen opacity-40"
              />
            )}
            {overlayMask && (
              <img
                src={overlayMask}
                alt=""
                className="absolute inset-0 w-full h-full object-contain pointer-events-none mix-blend-screen opacity-40"
              />
            )}
          </>
        ) : (
          <div className="flex flex-col items-center gap-2 text-zinc-600 px-3 text-center">
            <ImageIcon className="h-6 w-6" />
            <span className="text-[11px]">No image to edit</span>
          </div>
        )}

        {/* Floating edit-status bar — pinned to the bottom of the preview so it is
            always visible without scrolling, even while the layer controls scroll. */}
        {imageKey && (
          <div className="absolute inset-x-0 bottom-0 flex flex-col gap-1.5 px-2.5 py-2 bg-black/70 backdrop-blur-sm border-t border-zinc-700/60">
            <div className="flex items-center gap-2 px-0.5">
              <span className="text-[11px] text-zinc-200 font-medium">Edit this image</span>
              <span className="ml-auto text-[10px] text-zinc-400">
                {samMaskB64
                  ? invertMask
                    ? `Editing everything EXCEPT "${samPrompt}"`
                    : `Editing only "${samPrompt}"`
                  : hasMaskedSelection()
                    ? invertMask
                      ? 'Editing all EXCEPT the selected regions (inverted mask)'
                      : 'Editing only the selected regions (masked)'
                    : 'Editing the whole image'}
              </span>
              {editing && <Loader2 className="h-3.5 w-3.5 animate-spin text-zinc-300" />}
            </div>
          <div className="flex items-center gap-2 flex-wrap">
            <div className="relative flex items-stretch">
              <button
                type="button"
                onClick={() => void showLayers()}
                disabled={decomposing || !imageKey}
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-l-md bg-zinc-800 text-zinc-200 text-xs font-medium hover:bg-zinc-700 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
              >
                {decomposing ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Layers className="h-3.5 w-3.5" />}
                {decomposing ? 'Decomposing…' : 'Show layers'}
              </button>
              <button
                ref={qualityBtnRef}
                type="button"
                onClick={(e) => {
                  const el = e.currentTarget
                  const r = el.getBoundingClientRect()
                  setQualityMenuPos({ top: r.bottom + 4, left: r.left })
                  setQualityOpen(o => !o)
                }}
                aria-label="Layer quality preset"
                className="flex items-center px-1.5 rounded-r-md border-l border-zinc-700 bg-zinc-800 text-zinc-300 hover:bg-zinc-700 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
                disabled={decomposing || !imageKey}
              >
                <ChevronDown className={`h-3.5 w-3.5 transition-transform ${qualityOpen ? 'rotate-180' : ''}`} />
              </button>
              {qualityOpen && (
                <div
                  className="fixed inset-0 z-40 cursor-default"
                  onClick={() => setQualityOpen(false)}
                >
                  {qualityMenuPos && createPortal(
                    <div
                      className="fixed min-w-[9rem] rounded-lg border border-zinc-700 bg-zinc-900 py-1 shadow-2xl"
                      style={{ top: qualityMenuPos.top, left: qualityMenuPos.left, zIndex: 50 }}
                    >
                      {([['fast', 25], ['balanced', 30], ['detailed', 50]] as [LayerQuality, number][]).map(([q, s]) => (
                        <button
                          key={q}
                          type="button"
                          onClick={() => { setLayerQuality(q); setQualityOpen(false) }}
                          className={`w-full text-left px-3 py-1.5 text-[11px] flex items-center justify-between gap-2 hover:bg-zinc-800 ${
                            layerQuality === q ? 'text-emerald-400' : 'text-zinc-300'
                          }`}
                        >
                          <span>{LAYER_QUALITY_LABEL[q]}</span>
                          <span className="text-[10px] text-zinc-500">{s} steps</span>
                        </button>
                      ))}
                    </div>,
                    document.body,
                  )}
                </div>
              )}
            </div>
            <span className="text-[10px] text-zinc-500">Count</span>
            <button
              type="button"
              onClick={() => setLayerCount(c => Math.max(2, c - 1))}
              className="w-6 h-6 rounded bg-zinc-800 hover:bg-zinc-700 text-zinc-300 text-sm leading-none"
            >−</button>
            <span className="w-5 text-center text-xs text-zinc-200">{layerCount}</span>
            <button
              type="button"
              onClick={() => setLayerCount(c => Math.min(8, c + 1))}
              className="w-6 h-6 rounded bg-zinc-800 hover:bg-zinc-700 text-zinc-300 text-sm leading-none"
            >+</button>
            {layerSuggestion != null && (
              <span
                className="px-1.5 py-0.5 rounded bg-emerald-900/30 text-[10px] text-emerald-300"
                title="Suggested by Gemma from the image content"
              >{layerSuggestion} suggested</span>
            )}
            <div className="flex-1 min-w-[8rem] flex items-center gap-1.5">
              <input
                type="text"
                value={samPrompt}
                onChange={(e) => setSamPrompt(e.target.value)}
                onKeyDown={(e) => { if (e.key === 'Enter') void selectItem() }}
                placeholder="Select an item… (e.g. the red chair)"
                className="flex-1 min-w-0 rounded-md bg-zinc-800/70 border border-zinc-700 px-2 py-1.5 text-[11px] text-zinc-200 placeholder:text-zinc-500 focus:outline-none focus:border-emerald-500/60"
                disabled={samSelecting || !imageKey}
              />
              <button
                type="button"
                onClick={() => void selectItem()}
                disabled={samSelecting || !imageKey || !samPrompt.trim()}
                className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-md bg-emerald-700/70 text-white text-[11px] font-medium hover:bg-emerald-600 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
              >
                {samSelecting ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : <Wand2 className="h-3.5 w-3.5" />}
                {samSelecting ? 'Selecting…' : 'Select item'}
              </button>
              {samMaskB64 && (
                <button
                  type="button"
                  onClick={() => { setSamMaskB64(null); setMaskPreview(null); setSamPrompt('') }}
                  title="Clear selected item"
                  className="p-1.5 rounded bg-zinc-800 hover:bg-zinc-700 text-zinc-400 hover:text-white"
                >
                  <X className="h-3.5 w-3.5" />
                </button>
              )}
            </div>
            {layers.length > 0 && (
              <>
                <label className="flex items-center gap-1.5 cursor-pointer select-none px-2 py-1 rounded-md bg-zinc-800/60 text-[10px] text-zinc-300">
                  <input
                    type="checkbox"
                    checked={invertMask}
                    onChange={(e) => toggleInvert(e.target.checked)}
                    className="accent-emerald-500"
                  />
                  Invert
                </label>
                <button
                  type="button"
                  onClick={clear}
                  title="Clear layers"
                  className="p-1.5 rounded bg-zinc-800 hover:bg-zinc-700 text-zinc-400 hover:text-white"
                >
                  <X className="h-3.5 w-3.5" />
                </button>
              </>
            )}
          </div>
          </div>
        )}
      </div>
      {editError && <p className="text-[11px] text-red-400">{editError}</p>}

      {/* Layer controls */}
      <div className="rounded-xl border border-zinc-800 bg-zinc-900/60 p-2 flex flex-col gap-2">
        {samError && <p className="text-[11px] text-red-400">{samError}</p>}
        {layerError && <p className="text-[11px] text-red-400">{layerError}</p>}
        {decomposing && layerProgress && layerProgress.totalSteps > 0 && (
          <div className="flex items-center gap-2">
            <div className="h-1 flex-1 rounded-full bg-zinc-800 overflow-hidden">
              <div
                className="h-full bg-emerald-500 transition-all"
                style={{ width: `${Math.min(100, Math.round((layerProgress.step / layerProgress.totalSteps) * 100))}%` }}
              />
            </div>
            <span className="text-[10px] text-zinc-400 tabular-nums">
              step {layerProgress.step}/{layerProgress.totalSteps}
            </span>
          </div>
        )}
        {layers.length > 0 && (
          <div className="flex items-end gap-1.5 overflow-x-auto pb-0.5">
            {layers.map((lp, idx) => {
              const isSel = selected.includes(lp.index)
              const label = layerLabel(idx, layers.length)
              return (
                <div key={lp.index} className="flex flex-col items-center gap-1 w-16 flex-shrink-0">
                  <button
                    type="button"
                    onClick={() => toggleLayer(lp.index)}
                    className={`w-16 aspect-square rounded-md overflow-hidden border-2 transition-colors ${
                      isSel ? 'border-emerald-500' : 'border-zinc-700 hover:border-zinc-500'
                    }`}
                  >
                    <img src={webAssetUrl(lp.previewKey)} alt={label} className="w-full h-full object-cover" />
                  </button>
                  <span className={`text-[9px] ${isSel ? 'text-emerald-400' : 'text-zinc-500'}`}>{label}</span>
                </div>
              )
            })}
          </div>
        )}
      </div>
    </div>
  )
})
