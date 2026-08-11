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

export interface CatalogLora {
  id: string;
  name: string;
  description: string;
  /** Whether it ships pre-fetched on runners or must be downloaded on demand. */
  offline: boolean;
  recommended?: boolean;
  tags: string[];
}

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
    { id: "cinesuit", name: "CineSuit", description: "Cinematic color + film grain style", offline: true, recommended: true, tags: ["style"] },
    { id: "bokeh", name: "Bokeh", description: "Shallow depth-of-field portrait look", offline: true, tags: ["style"] },
    { id: "anime", name: "Anime", description: "Anime illustration style", offline: false, tags: ["style"] },
    { id: "macro", name: "Macro", description: "Macro / close-up detail boost", offline: false, tags: ["style"] },
  ],
  icLoras: [
    { id: "subject-preserve", name: "Subject Preserve", description: "Identity-consistent subject restyle (IC-LoRA)", offline: true, recommended: true, tags: ["identity"] },
    { id: "style-transfer", name: "Style Transfer", description: "Reference style transfer to the subject", offline: true, tags: ["style"] },
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
