/**
 * Static default catalog for models / LoRAs / IC-LoRAs served by the Worker
 * (`keep (Worker/catalog)` routes in api-contract.md). The model goes through
 * a `getCatalog()` function + `catalogService` so it can be swapped for a
 * data-driven source (D1 table or remote) later without touching the routes.
 *
 * The catalog describes *what the runners can run* (model ids + version tags).
 * Weights live on the GPU runners — the Worker never holds model files.
 */

export type ModelType = "t2v" | "image-gen" | "text-encoder" | "ic-lora" | "lora";

export interface CatalogModel {
  id: string;
  name: string;
  type: ModelType;
  /** Match a task type (dispatch routes use this to pick a model). */
  task?: "generate" | "generate-image" | "enhance-prompt" | "restyle" | "ic-lora" | "extend";
  /** Available version tags for this model (the `ltx-versions` surface). */
  versions: string[];
  /** True when this is the current recommended default for its task. */
  recommended?: boolean;
  /** Whether the model is available without a manual download (fetched at run). */
  available?: boolean;
  /** Optional HuggingFace repo the runner pulls weights from. */
  source?: string;
}

export interface CatalogLoraVariant {
  id: string;
  label: string;
  filename: string;
  size_bytes: number;
}
/** Full LoraCatalogItem / IcLoraCatalogItem-shape catalog entry (what the frontend's
 *  `catalogItemToEntry` reads: download.repo_id + download.variants + name/desc/strength). */
export interface CatalogLora {
  allows_empty_prompt?: boolean;
  /** Only on IC-LoRAs (IcLoraCatalogItem). */
  allows_reference_image?: boolean;
  author?: { name: string; url?: string | null } | null;
  description: string;
  download: { repo_id: string; variants: CatalogLoraVariant[] };
  id: string;
  license?: { name: string; url?: string | null } | null;
  media?: { thumbnail?: string | null; demo_video?: string | null } | null;
  name: string;
  recommended_strength?: number | null;
  requires_hf_login: boolean;
  tags?: string[];
}
export { };

export interface Catalog {
  models: CatalogModel[];
  loras: CatalogLora[];
  icLoras: CatalogLora[];
}

/** Default catalog of LTX-2 model ids + current LoRA list. */
const DEFAULT_CATALOG: Catalog = {
  models: [
    {
      id: "LTX-2",
      name: "LTX-2 (text-to-video)",
      type: "t2v",
      task: "generate",
      versions: ["v0.9.1", "v0.9.0", "v0.8.0"],
      recommended: true,
      available: true,
      source: "Lightricks/LTX-2",
    },
    {
      id: "LTX-2C",
      name: "LTX-2C (context / multi-shot)",
      type: "t2v",
      task: "extend",
      versions: ["v0.9.1", "v0.9.0"],
      available: true,
      source: "Lightricks/LTX-2C",
    },
    {
      id: "LTX-2-IC",
      name: "LTX-2 Image-to-Video (IC-LoRA)",
      type: "t2v",
      task: "ic-lora",
      versions: ["v0.9.1", "v0.9.0"],
      available: true,
      source: "Lightricks/LTX-2-IC",
    },
    {
      id: "FLUX.1-dev",
      name: "FLUX.1-dev (image generation)",
      type: "image-gen",
      task: "generate-image",
      versions: ["fp8", "bf16"],
      available: true,
      source: "black-forest-labs/FLUX.1-dev",
    },
    {
      id: "LTX-2-text-encoder",
      name: "LTX-2 text encoder",
      type: "text-encoder",
      versions: ["v1"],
      recommended: true,
      available: true,
      source: "Lightricks/LTX-2-text-encoder",
    },
    {
      id: "LTX-2-IC-embedder",
      name: "LTX-2 IC-LoRA embedder",
      type: "ic-lora",
      versions: ["v1"],
      recommended: true,
      available: true,
    },
  ],
  loras: [
    { id: "cinesuit", name: "CineSuit", description: "Cinematic color + film grain style", allows_empty_prompt: true, requires_hf_login: false, tags: ["style"], recommended_strength: 0.8, download: { repo_id: "Lightricks/ltx-models", variants: [{ id: "cinesuit", label: "CineSuit", filename: "cinesuit.safetensors", size_bytes: 260_000_000 }] } },
    { id: "bokeh", name: "Bokeh", description: "Shallow depth-of-field portrait look", allows_empty_prompt: true, requires_hf_login: false, tags: ["style"], recommended_strength: 0.6, download: { repo_id: "Lightricks/ltx-models", variants: [{ id: "bokeh", label: "Bokeh", filename: "bokeh.safetensors", size_bytes: 260_000_000 }] } },
    { id: "anime", name: "Anime", description: "Anime illustration style", allows_empty_prompt: true, requires_hf_login: false, tags: ["style"], recommended_strength: 0.7, download: { repo_id: "Lightricks/ltx-models", variants: [{ id: "anime", label: "Anime", filename: "anime.safetensors", size_bytes: 260_000_000 }] } },
    { id: "macro", name: "Macro", description: "Macro / close-up detail boost", allows_empty_prompt: true, requires_hf_login: false, tags: ["style"], recommended_strength: 0.75, download: { repo_id: "Lightricks/ltx-models", variants: [{ id: "macro", label: "Macro", filename: "macro.safetensors", size_bytes: 260_000_000 }] } },
  ],
  icLoras: [
    { id: "subject-preserve", name: "Subject Preserve", description: "Identity-consistent subject restyle (IC-LoRA)", allows_empty_prompt: false, allows_reference_image: true, requires_hf_login: false, tags: ["identity"], recommended_strength: 1.0, download: { repo_id: "Lightricks/ltx-models", variants: [{ id: "subject-preserve", label: "Subject Preserve", filename: "subject-preserve.safetensors", size_bytes: 320_000_000 }] } },
    { id: "style-transfer", name: "Style Transfer", description: "Reference style transfer to the subject", allows_empty_prompt: false, allows_reference_image: true, requires_hf_login: false, tags: ["style"], recommended_strength: 1.0, download: { repo_id: "Lightricks/ltx-models", variants: [{ id: "style-transfer", label: "Style Transfer", filename: "style-transfer.safetensors", size_bytes: 320_000_000 }] } },
  ],
};

/** Load the catalog; currently returns the default, later a data-driven source. */
export function getCatalog(): Catalog {
  return DEFAULT_CATALOG;
}

/** Models filtered by a type (t2v / image-gen / ...). */
export function modelsOfType(type: ModelType): CatalogModel[] {
  return getCatalog().models.filter((m) => m.type === type);
}

/** The recommended model for a task, or null. */
export function recommendedForTask(task: Exclude<ModelType, "lora">): CatalogModel | undefined {
  return getCatalog().models.find((m) => m.task === task && m.recommended);
}

export interface RecommendResult {
  name: string;
  id: string;
  recommended: boolean;
  versions: string[];
}

/**
 * The `catalogService` seam — all Worker catalog routes funnel through this so the
 * backing store can become data-driven without changing route handlers.
 */
export function catalogService() {
  const catalog = getCatalog();
  return {
    getCatalog: () => catalog,
    models: () => catalog.models,
    loras: () => catalog.loras,
    icLoras: () => catalog.icLoras,
    ltxVersions: () => catalog.models.filter((m) => m.type === "t2v").flatMap((m) => m.versions),
    modelSpecs: () => catalog.models,
    recommend: (task: "ltx" | "ltx-ic-lora" | "img-gen" | "text-encoder"): RecommendResult | null => {
      const t = task === "ltx" ? "generate" : task === "ltx-ic-lora" ? "ic-lora" : task === "img-gen" ? "generate-image" : "text-encoder";
      const m = catalog.models.find((mo) => mo.task === t && mo.recommended);
      if (!m) return null;
      return { name: m.name, id: m.id, recommended: true, versions: m.versions };
    },
  };
}


/** Contract shape for GET /api/generate/models-specs (GenerateVideoModelsSpecsResponse). */
export interface VideoGenerationModelSpecItem {
  pipeline: "fast" | "pro";
  spec: {
    display_name: string;
    supported_resolutions_durations: Record<string, { fps_to_durations: Record<string, number[]> }>;
    a2v_supported_resolutions_durations?: Record<string, { fps_to_durations: Record<string, number[]> }> | null;
  };
}

const _fullDurations = [6, 8, 10, 12, 14, 16, 18, 20];
const _shortDurations = [6, 8, 10];
const _fps = (d24: number[], d25: number[]) => ({
  "24": d24,
  "25": d25,
  "48": _shortDurations,
  "50": _shortDurations,
});
const _res = (r24: number[], r25: number[]) => ({
  "1080p": { fps_to_durations: _fps(r24, r25) },
  "1440p": { fps_to_durations: _fps(_shortDurations, _shortDurations) },
  "2160p": { fps_to_durations: _fps(_shortDurations, _shortDurations) },
});

/**
 * Real LTX-2 video-generation capability specs (mirrored from the desktop backend's
 * api_model_specs.py) so the browser timeline editor can build its resolution/fps/duration
 * pickers. There are no local models in the web app — inference runs on remote runners.
 */
export function getVideoGenerationModelSpecs(): { api_models: VideoGenerationModelSpecItem[]; local_models: VideoGenerationModelSpecItem[] } {
  const fastSpec: VideoGenerationModelSpecItem["spec"] = {
    display_name: "LTX-2.3 Fast (API)",
    supported_resolutions_durations: _res(_fullDurations, _fullDurations),
    a2v_supported_resolutions_durations: _res(_fullDurations, _fullDurations),
  };
  const proSpec: VideoGenerationModelSpecItem["spec"] = {
    display_name: "LTX-2.3 Pro (API)",
    supported_resolutions_durations: _res(_shortDurations, _shortDurations),
    a2v_supported_resolutions_durations: _res(_shortDurations, _shortDurations),
  };
  return {
    local_models: [],
    api_models: [
      { pipeline: "fast", spec: fastSpec },
      { pipeline: "pro", spec: proSpec },
    ],
  };
}
