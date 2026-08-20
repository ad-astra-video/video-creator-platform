/**
 * Client-side Livepeer runner discovery for the web build (browser, no Cloudflare Worker).
 *
 * Under the direct-transport design the Worker is control-plane only and is NOT in the
 * inference path, and the web build's Worker has no job rows / no knowledge of the media path.
 * So listing "which video-creator runners can I use" is done HERE, straight against the user's
 * configured Livepeer Discovery URL (which Caddy already answers CORS for on the orchestrator
 * box — the browser already talks to that orchestrator for the paying ticket rail).
 *
 * This is a faithful browser port of the Worker's OrchestratorClient.discoverRunners /
 * parseDiscovery / normalizeRunner (src/orchestrator.ts) + providers.ts toProviderDto, so the
 * returned runner shape matches what resolveRunner() and RunnersSection expect. The desktop
 * build keeps using ApiClient.getProviders() (its backend does the same discovery).
 */
import type { RunnerDto } from './direct-transport'

export function isWebPlatform(): boolean {
  try {
    return (window as unknown as { electronAPI?: { platform?: string } }).electronAPI?.platform === 'web'
  } catch {
    return false
  }
}

const TASK_LABELS: Record<string, string> = {
  t2v: 'Text-to-Video',
  image: 'Image',
  extend: 'Extend',
  retake: 'Retake',
  restyle: 'Restyle',
  'ic-lora': 'IC-LoRA',
  'ic-lora-generate': 'IC-LoRA',
  sam3: 'Segment',
  prompt: 'Prompt Enhance',
  'prompt-enhance': 'Prompt Enhance',
  'suggest-gap-prompt': 'Script Gap Fill',
  i2v: 'Image-to-Video',
  edit: 'Masked Edit',
  'extract-conditioning': 'Extract Conditioning',
  chat: 'Chat',
}

/** Price shape used by the runners list UI (superset of the selection-only RunnerDto.price_info). */
export type PriceInfo = NonNullable<RunnerDto['price_info']> & {
  usdPerSec?: number
  pricePerUnit?: number
  pixelsPerUnit?: number
}

/** A discovered runner: same fields resolveRunner needs, plus the display price extras and
 *  the runner-advertised video model specs (resolution/fps/duration) from discovery metadata. */
export interface DiscoveredRunner extends RunnerDto {
  price_info?: PriceInfo | null
  /** Structured video model specs advertised by the runner (from metadata.model_specs). */
  modelSpecs?: unknown[]
}

/** Runner-discovery inputs the web build needs at call time (fed from AppSettings context). */
interface RunnerDiscoveryConfig {
  /** The user's saved Livepeer Discovery URL ('' = none configured). */
  discoveryUrl: string
  /** The user's preferred runner id (persisted in settings), '' = no preference. */
  selectedRunnerId: string
}

let runnerDiscoveryConfig: RunnerDiscoveryConfig = { discoveryUrl: '', selectedRunnerId: '' }

/** Called by AppSettingsProvider whenever it loads/reloads settings so the lib stays in sync. */
export function setRunnerDiscoveryConfig(patch: Partial<RunnerDiscoveryConfig>): void {
  runnerDiscoveryConfig = { ...runnerDiscoveryConfig, ...patch }
}

export function getRunnerDiscoveryConfig(): RunnerDiscoveryConfig {
  return runnerDiscoveryConfig
}

/** The browser-local list of excluded runner URLs (matches RunnersSection's `vc-excluded-providers`). */
const EXCLUDED_STORAGE_KEY = 'vc-excluded-providers'

export function loadExcludedRunnerUrls(): string[] {
  try {
    const raw = localStorage.getItem(EXCLUDED_STORAGE_KEY)
    const parsed = raw ? JSON.parse(raw) : []
    return Array.isArray(parsed) ? parsed.filter((x): x is string => typeof x === 'string') : []
  } catch {
    return []
  }
}

/** Parse heartbeat/discovery `metadata`, which may be an object or a JSON string. */
function parseMeta(v: unknown): Record<string, unknown> {
  if (!v) return {}
  if (typeof v === 'object') return v as Record<string, unknown>
  try {
    const p = JSON.parse(String(v))
    return p && typeof p === 'object' ? (p as Record<string, unknown>) : {}
  } catch {
    return {}
  }
}

/** Which discovery URLs to try for a base, mirroring OrchestratorClient.discoveryCandidates. */
function discoveryCandidates(base: string): string[] {
  const withApp = (u: string): string =>
    u.includes('app=') ? u : u + (u.includes('?') ? '&app=video-creator' : '?app=video-creator')
  return [base, withApp(base), withApp(`${base}/discovery`), withApp(`${base}/api/discovery`)]
}

/** Normalize a discovery payload (go-livepeer `[{address, runners:[...]}]` + flat shapes) to DiscoveredRunner[]. */
function parseDiscovery(data: unknown): DiscoveredRunner[] {
  if (!Array.isArray(data)) return []
  const out: DiscoveredRunner[] = []
  for (const entry of data as Record<string, unknown>[]) {
    if (!entry) continue
    if (Array.isArray(entry.runners)) {
      for (const r of entry.runners as Record<string, unknown>[]) out.push(normalizeRunner(r))
    } else if (entry.runner_id !== undefined || entry.url !== undefined || entry.runner_url !== undefined) {
      out.push(normalizeRunner(entry))
    }
  }
  return out
}

function normalizeRunner(raw: Record<string, unknown>): DiscoveredRunner {
  const url = String(raw.url ?? raw.runner_url ?? '')
  let id = raw.runner_id ? String(raw.runner_id) : ''
  if (!id && url) {
    const seg = url.replace(/\/+$/, '').split('/')
    id = seg.length >= 2 ? seg[seg.length - 2] : seg[seg.length - 1] || url
  }
  if (!id) id = 'runner'

  const gpuRaw = raw.gpu && typeof raw.gpu === 'object' ? (raw.gpu as Record<string, unknown>) : {}
  const gpu: { name?: string; vram_mb?: number } = {}
  if (gpuRaw.name !== undefined) gpu.name = String(gpuRaw.name)
  if (typeof gpuRaw.vram_mb === 'number') gpu.vram_mb = gpuRaw.vram_mb

  const meta = parseMeta(raw.metadata)
  let caps: string[] = []
  if (Array.isArray(raw.capabilities)) caps = (raw.capabilities as unknown[]).map(String)
  if (caps.length === 0 && Array.isArray(meta.capabilities)) caps = (meta.capabilities as unknown[]).map(String)

  // Prices come from the discovery payload ONLY — never synthesized (see providers.ts buildPrice).
  const priceInfo: PriceInfo = {}
  if (typeof raw.priceUsdMicrosPerSec === 'number' && Number.isFinite(raw.priceUsdMicrosPerSec)) {
    priceInfo.usdPerSec = raw.priceUsdMicrosPerSec / 1_000_000
  }
  const rawPi = raw.price_info && typeof raw.price_info === 'object' ? (raw.price_info as Record<string, unknown>) : {}
  const glSource =
    (raw.priceInfo && typeof raw.priceInfo === 'object' ? (raw.priceInfo as Record<string, unknown>) : undefined) ??
    (rawPi && typeof rawPi === 'object' ? rawPi : undefined)
  if (typeof rawPi.price === 'number') {
    priceInfo.price = rawPi.price
    priceInfo.currency = typeof rawPi.currency === 'string' ? rawPi.currency : ''
    priceInfo.unit = typeof rawPi.unit === 'string' ? rawPi.unit : ''
  }
  const ppu = glSource ? glSource.pricePerUnit : raw.pricePerUnit
  const pix = glSource ? glSource.pixelsPerUnit : raw.pixelsPerUnit
  if (ppu !== undefined) priceInfo.pricePerUnit = Number(ppu)
  if (pix !== undefined) priceInfo.pixelsPerUnit = Number(pix)

  const status = raw.status === 'busy' ? 'busy' : raw.status === 'offline' ? 'offline' : 'ready'
  const models = Array.isArray(meta.models) ? (meta.models as unknown[]).map(String) : undefined
  const modelSpecs = Array.isArray(meta.model_specs) ? (meta.model_specs as unknown[]) : undefined

  // Advertised concurrency capacity (e.g. capacity=3 from GPU count). Kept when the runner
  // sends them so the UI can show real capacity instead of a VRAM guess.
  const num = (v: unknown): number | undefined => (typeof v === 'number' && Number.isFinite(v) ? v : undefined)
  const capacity = num(raw.capacity)
  const capacityUsed = num(raw.capacity_used)
  const capacityAvailable = num(raw.capacity_available)

  return {
    runner_id: id,
    url,
    status,
    selected: false, // filled in by resolveRunner from the persisted preference
    excluded: false, // filled by resolveRunner + the localStorage excluded set
    ...(gpu.name !== undefined || gpu.vram_mb !== undefined ? { gpu } : {}),
    ...(Object.keys(priceInfo).length ? { price_info: priceInfo } : {}),
    capabilities: caps.map((t) => ({ id: t, label: TASK_LABELS[t] ?? t })),
    ...(models && models.length ? { models } : {}),
    ...(modelSpecs !== undefined ? { modelSpecs } : {}),
    ...(capacity !== undefined ? { capacity } : {}),
    ...(capacityUsed !== undefined ? { capacityUsed } : {}),
    ...(capacityAvailable !== undefined ? { capacityAvailable } : {}),
  }
}

/**
 * GET the Discovery URL and return the normalized runners it advertises (all capabilities).
 * Throws if every candidate was unreachable (caller surfaces "could not reach Discovery URL");
 * returns [] if a candidate answered but advertised nothing.
 */
export async function discoverRunners(discoveryUrl: string): Promise<DiscoveredRunner[]> {
  const base = discoveryUrl.replace(/\/+$/, '')
  if (!base) return []
  let lastErr: unknown = null
  let sawOk = false
  for (const url of discoveryCandidates(base)) {
    try {
      const res = await fetch(url, { method: 'GET', headers: { accept: 'application/json' } })
      if (!res.ok) {
        lastErr = new Error(`discovery ${res.status} at ${url}: ${await res.text()}`)
        continue
      }
      sawOk = true
      const parsed = parseDiscovery(await res.json())
      if (parsed.length > 0) return parsed
    } catch (e) {
      lastErr = e
    }
  }
  if (sawOk) return []
  if (lastErr) throw lastErr instanceof Error ? lastErr : new Error(String(lastErr))
  return []
}
