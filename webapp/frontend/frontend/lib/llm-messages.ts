// Client-side prompt builder — the single source of truth for every LLM prompt.
//
// All LLM messages are authored HERE (in the browser) and, when an OpenRouter key is
// set, sent to OpenRouter directly. When no key is set the SAME prebuilt `messages`
// are executed by the runner's gemma-worker (which becomes a pure executor). The
// runner's own `_build_enhance_messages` / LAYER_SUGGEST_RUBRIC are mirrors of these
// constants.
//
// ⚠️ keep in sync with runner/gemma/server.py and runner/ltx/enhance_forward.py

export interface CustomPrompts {
  enhancerT2V?: string
  enhancerI2V?: string
  enhancerExtend?: string
  enhancerRetake?: string
  layerSuggestRubric?: string
  gapFillRubric?: string
}

export type ChatMessage = { role: string; content: unknown }

const VERBOSE_FORMAT_INSTRUCTION = (
  "Respond with ONLY the rewritten prompt, as a SINGLE long, deliberately " +
  "VERBOSE paragraph of natural language. Be exhaustively detailed: write " +
  "several dense sentences of rich, concrete visual description instead of a " +
  "terse one-liner — the prompt must be LONG. Never truncate, summarize, or " +
  "shorten it, and never favor brevity over detail; every sentence should add " +
  "specific, granular visual information. No titles, headings, prefaces, " +
  "explanations, quotes, or markdown formatting (no **bold**, no bullet " +
  "points, no numbered list) — just the prompt text itself."
)

export const DEFAULT_T2V_SYSTEM_PROMPT = (
  "You are a prompt engineer for a text-to-video generation model. Rewrite the " +
  "user's prompt into a single, vivid, concrete and VERY LONG visual " +
  "description, preserving the user's original subject and intent without " +
  "inventing an unrelated scene. Describe the scene in great detail, being " +
  "deliberately verbose and covering each of these dimensions as fully as " +
  "possible: " +
  "(1) SUBJECT — identity, age, build, distinctive appearance, clothing, colors, " +
  "facial expression, body language; " +
  "(2) ACTION & MOTION — exactly what the subject does, how, at what pace or " +
  "intensity, and any secondary movement in the frame; " +
  "(3) SETTING — the full environment: location, background, foreground props, " +
  "weather, time of day, and how it is decorated; " +
  "(4) LIGHTING & COLOR — light source and direction, shadows, highlights, mood, " +
  "and the overall color palette; " +
  "(5) CAMERA — framing, angle, distance, lens feel, and any camera movement; " +
  "(6) ATMOSPHERE — texture, material detail, cinematography style, genre, and " +
  "overall mood or tone. " +
  "Err on the side of MORE detail: the more granular and specific your " +
  "description, the better the generated video matches the intent. " +
  VERBOSE_FORMAT_INSTRUCTION
)

export const DEFAULT_I2V_SYSTEM_PROMPT = (
  "You are a prompt engineer for a text-to-video generation model. The user has also " +
  "provided a reference IMAGE that must drive the result: carefully inspect that image and " +
  "rewrite their prompt into a single, vivid, concrete and VERY LONG description of the " +
  "result video, grounded in every detail the image actually shows. Be deliberately " +
  "verbose and cover, as fully as possible: " +
  "(1) the image's subject — identity, appearance, pose, clothing and colors; " +
  "(2) the surrounding scene and composition; " +
  "(3) the action and MOTION to animate on top of the still, including pace and intensity; " +
  "(4) lighting, color palette and style, matching the reference; " +
  "(5) camera framing and any camera movement; " +
  "(6) atmosphere, texture and overall mood. " +
  "Faithfully preserve the image's subject and its visual appearance, surrounding scene, " +
  "composition, and style while describing the desired motion in detail. Do not invent " +
  "a different subject, character, or setting that contradicts the reference image. " +
  "Err on the side of MORE detail: exhaustively describe the scene in great detail. " +
  VERBOSE_FORMAT_INSTRUCTION
)

export const DEFAULT_EXTEND_SYSTEM_PROMPT = (
  "You are a prompt engineer for a video EXTENSION (continuation) model. The user is " +
  "extending an existing clip, and has provided a set of FRAMES sampled from the source " +
  "video's context window — the stretch of footage right at the boundary the extension " +
  "attaches to (the last second for an 'end' extension, the first second for a 'start' " +
  "extension). Carefully inspect those frames and rewrite the user's short direction into " +
  "a single, vivid, VERY LONG continuation prompt fully grounded in the source. Be " +
  "deliberately verbose, describing in great detail and at length: the same subject(s) — " +
  "identity, appearance, clothing and pose; the same setting, composition, framing, " +
  "lighting, color palette and overall style shown in the frames; the same motion/action " +
  "already in progress — while honoring the motion or scene change the user's direction " +
  "requests, and the natural next beat of the footage. Do not invent a different subject, " +
  "character, location, or a jarring visual style that contradicts the context frames; the " +
  "result must read as the natural next moment of the supplied footage. " +
  "Lavish concrete visual detail on every element so the continuation is seamless. " +
  VERBOSE_FORMAT_INSTRUCTION
)

export const DEFAULT_RETAKE_SYSTEM_PROMPT = (
  "You are a prompt engineer for a video RETAKE (re-render) model. The user is " +
  "re-rendering a selected segment of an existing clip and has provided FRAMES sampled " +
  "from that selected segment. Carefully inspect those frames and rewrite the user's short " +
  "instruction into a single, vivid, VERY LONG prompt that re-renders the segment to match " +
  "the source closely. Be deliberately verbose and richly detailed: restate at length the " +
  "same subject(s) — identity, appearance, clothing and pose — the same setting, " +
  "composition, framing, lighting, color palette and style, and the same general activity. " +
  "Change ONLY the one specific thing the user's direction asks to change (for example, if " +
  "a jet is moving erratically, have it straighten out its flight path) and leave every " +
  "other visual element exactly as the frames show; do not alter anything the direction " +
  "does not explicitly touch. The result must remain visually continuous with the supplied " +
  "frames. Describe everything in great, granular detail. " +
  VERBOSE_FORMAT_INSTRUCTION
)

export const LAYER_SUGGEST_RUBRIC = (
  "You are a meticulous image-decomposition analyst. Analyze the image and " +
  "select the appropriate Qwen-Image-Layered layer count.\n\n" +
  "Use this rubric:\n" +
  "- 2 = simple image with one main subject and simple background\n" +
  "- 3 = main subject + background + one distinct secondary element\n" +
  "- 4 = several distinct objects or regions\n" +
  "- 5 = moderately complex scene with multiple overlapping objects\n" +
  "- 6 = complex scene with many independently editable objects\n" +
  "- 7 = very complex scene with many distinct overlapping elements\n" +
  "- 8 = extremely complex scene where separating many objects is useful\n\n" +
  "Think step by step before answering:\n" +
  "1. Identify every semantically distinct element or region in the image.\n" +
  "2. Consider background, foreground subjects, and any independent objects.\n" +
  "3. Estimate whether each element is cleanly separable for editing.\n" +
  "4. Choose the SMALLEST number that adequately represents the image.\n" +
  "5. Show your reasoning in  thinking... response tags, then answer with ONLY " +
  "the final integer 2-8 on the very last line, inside a single pair of " +
  "angle brackets, e.g. <5>.\n" +
  "Put no other text after the bracketed number."
)

export const GAP_FILL_RUBRIC = (
  "You are a video editor's assistant filling an EMPTY GAP in a timeline. A clip is missing " +
  "between two existing segments and you must write the prompt for the fill clip.\n\n" +
  "Write ONE vivid, concrete and fairly long prompt describing the short clip that should " +
  "fill the gap. Make it visually continuous with the surrounding footage: match the same " +
  "subject(s), setting, lighting, color palette, style and general motion so the result " +
  "reads as the natural missing beat, and weave in the requested gap duration (~{duration}s) " +
  "as a sense of pacing. When the before/after summary prompts or sampled frames are " +
  "provided, ground the fill in what they actually show and do not invent a jarringly " +
  "different subject, location or style.\n\n" +
  "Respond with ONLY the fill-clip prompt text — a single paragraph, no titles, headings, " +
  "prefaces, quotes or markdown formatting."
)

/** Sniff a base64-encoded image's MIME type from its header bytes (mirror of the runner). */
export function imageMime(base64Image: string): string {
  try {
    const head = atob(base64Image.slice(0, 64)).slice(0, 4)
    if (head.charCodeAt(0) === 0xff && head.charCodeAt(1) === 0xd8) return 'image/jpeg'
  } catch {
    /* fall through */
  }
  return 'image/png'
}

export const toDataUrl = (base64Image: string) =>
  `data:${imageMime(base64Image)};base64,${base64Image}`

/**
 * OpenAI-style messages for prompt enhancement. Mirrors the runner's
 * `_build_enhance_messages`: context_frames -> EXTEND (or RETAKE) system prompt with a
 * multimodal user message; a single image -> I2V system prompt + one image_url part;
 * neither -> T2V system prompt with a plain-text user message.
 */
export function buildEnhanceMessages(
  input: {
    prompt: string
    systemPrompt?: string
    imageBase64?: string
    contextFrames?: string[]
    task?: 'extend' | 'retake'
    direction?: string
  },
  overrides?: CustomPrompts,
): ChatMessage[] {
  const prompt = (input.prompt || '').trim()
  if (input.contextFrames && input.contextFrames.length) {
    const userContent: unknown[] = []
    for (const frame of input.contextFrames) {
      if (!frame) continue
      userContent.push({ type: 'image_url', image_url: { url: toDataUrl(frame) } })
    }
    let system: string
    let note = ''
    if (input.task === 'retake') {
      system = input.systemPrompt || overrides?.enhancerRetake || DEFAULT_RETAKE_SYSTEM_PROMPT
      note = ' Task: re-render this selected segment.'
    } else {
      system = input.systemPrompt || overrides?.enhancerExtend || DEFAULT_EXTEND_SYSTEM_PROMPT
      const direction = (input.direction || '').trim()
      note = direction ? ` Extend direction: ${direction}.` : ''
    }
    userContent.push({ type: 'text', text: `User Raw Input Prompt: ${prompt}.${note}` })
    return [{ role: 'system', content: system }, { role: 'user', content: userContent }]
  }
  if (input.imageBase64) {
    const system = input.systemPrompt || overrides?.enhancerI2V || DEFAULT_I2V_SYSTEM_PROMPT
    return [
      { role: 'system', content: system },
      {
        role: 'user',
        content: [
          { type: 'image_url', image_url: { url: toDataUrl(input.imageBase64) } },
          { type: 'text', text: `User Raw Input Prompt: ${prompt}.` },
        ],
      },
    ]
  }
  const system = input.systemPrompt || overrides?.enhancerT2V || DEFAULT_T2V_SYSTEM_PROMPT
  return [
    { role: 'system', content: system },
    { role: 'user', content: `user prompt: ${prompt}` },
  ]
}

/** Layer-count suggestion: rubric + image + "how many layers?" (mirror of the runner). */
export function buildLayerSuggestMessages(
  imageBase64: string,
  overrides?: CustomPrompts,
): ChatMessage[] {
  return [
    { role: 'system', content: overrides?.layerSuggestRubric || LAYER_SUGGEST_RUBRIC },
    {
      role: 'user',
      content: [
        { type: 'image_url', image_url: { url: toDataUrl(imageBase64) } },
        { type: 'text', text: 'How many layers should this image be decomposed into?' },
      ],
    },
  ]
}

/** Gap-fill: rubric + surrounding context (prompts and/or frames) -> a fill-clip prompt. */
export function buildGapSuggestMessages(
  input: {
    gapDuration: number
    mode: string
    beforePrompt?: string
    afterPrompt?: string
    beforeFrame?: string
    afterFrame?: string
    inputImage?: string
  },
  overrides?: CustomPrompts,
): ChatMessage[] {
  const rubric = (overrides?.gapFillRubric || GAP_FILL_RUBRIC).replace(
    '{duration}',
    String(Math.max(0, Math.round(input.gapDuration))),
  )
  const userContent: unknown[] = []
  if (input.inputImage) userContent.push({ type: 'image_url', image_url: { url: toDataUrl(input.inputImage) } })
  if (input.beforeFrame) userContent.push({ type: 'image_url', image_url: { url: toDataUrl(input.beforeFrame) } })
  if (input.afterFrame) userContent.push({ type: 'image_url', image_url: { url: toDataUrl(input.afterFrame) } })
  const note = ` Fill mode: ${input.mode}.${input.beforePrompt ? ` Before summary: ${input.beforePrompt}.` : ''}${input.afterPrompt ? ` After summary: ${input.afterPrompt}.` : ''}`
  userContent.push({ type: 'text', text: `Write the prompt for the ${Math.max(0, Math.round(input.gapDuration))}s timeline gap.${note}` })
  return [{ role: 'system', content: rubric }, { role: 'user', content: userContent }]
}
