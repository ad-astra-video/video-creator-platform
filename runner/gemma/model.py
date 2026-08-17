"""Llama.cpp-backed Gemma 4 LLM wrapper for the gemma-worker.

Owns the single ``llama_cpp.Llama`` instance. llama.cpp releases the GIL during
model eval, but a single ``Llama`` shares ONE context, so actual inference is
serialized with a generation lock while the async server admits up to
GEMMA_MAX_PARALLEL concurrent prompt requests (each runs in its own worker
thread, queued on the lock). This satisfies "limit parallel executions of
prompts to N" (N concurrent request slots) while remaining GPU-safe.
"""

from __future__ import annotations

import gc
import logging
import threading
from typing import Any

logger = logging.getLogger("video_creator.runner.gemma.model")


class GemmaLLM:
    """Wraps a single llama_cpp.Llama instance + its generation lock."""

    def __init__(
        self,
        model_path: str,
        mmproj: str | None = None,
        n_gpu_layers: int = -1,
        main_gpu: int = 0,
        n_ctx: int = 8192,
        flash_attn: bool = True,
        kv_cache_q8: bool = True,
    ) -> None:
        self.model_path = model_path
        self.mmproj = mmproj
        self.n_gpu_layers = n_gpu_layers
        self.main_gpu = main_gpu
        self.n_ctx = n_ctx
        # Flash-attention is REQUIRED for large context on this Gemma arch
        # (without it llama.cpp pads the V-cache to head_dim 2048 and OOMs early)
        # and is always on. Q8_0 KV-cache quantization (GGML_TYPE_Q8_0=8) is what
        # makes the full 128K model context fit a 32 GB card (~31.8 GB vs OOM).
        self.flash_attn = flash_attn
        self.type_k = 8 if kv_cache_q8 else 1  # 8=Q8_0, 1=F16
        self.type_v = 8 if kv_cache_q8 else 1
        self._llm: Any | None = None
        self._load_lock = threading.Lock()
        self._gen_lock = threading.Lock()

    @property
    def is_ready(self) -> bool:
        return self._llm is not None

    def load(self) -> None:
        """Build the Llama instance (mmap-backed; fast when weights are warm)."""
        with self._load_lock:
            if self._llm is not None:
                return
            import llama_cpp  # heavy native import — load lazily

            kwargs: dict[str, Any] = dict(
                model_path=self.model_path,
                n_gpu_layers=self.n_gpu_layers,
                main_gpu=self.main_gpu,
                n_ctx=self.n_ctx,
                flash_attn=self.flash_attn,
                type_k=self.type_k,
                type_v=self.type_v,
            )
            if self.mmproj:
                kwargs["mmproj"] = self.mmproj
                # Wire the multimodal chat handler so image content is actually encoded.
                # WITHOUT this, Llama falls back to `chat_template.default`, which emits the
                # <|image|> placeholder but never runs the CLIP/vision encoder — so every
                # image-bearing request degrades to a text-only prompt and the model replies
                # "please provide the image". Gemma4ChatHandler drives the mtmd/clip path
                # (verified: prompt tokens jump to image-count and the model sees the pixels).
                from llama_cpp.llama_chat_format import Gemma4ChatHandler

                kwargs["chat_handler"] = Gemma4ChatHandler(
                    clip_model_path=self.mmproj, use_gpu=True
                )
            self._llm = llama_cpp.Llama(**kwargs)
            logger.info(
                "Gemma LLM loaded: %s (n_gpu_layers=%s main_gpu=%s)",
                self.model_path, self.n_gpu_layers, self.main_gpu,
            )

    def evict(self) -> None:
        """Drop the model so GPU layers release for an incoming render worker."""
        with self._load_lock:
            self._llm = None
        gc.collect()
        logger.info("Gemma LLM evicted")

    def chat(self, messages: list[dict[str, Any]], *, max_tokens: int = 512,
             temperature: float = 0.7, seed: int | None = None) -> str:
        """Run a chat completion synchronously (call via ``asyncio.to_thread``)."""
        if self._llm is None:
            raise RuntimeError("Gemma LLM not loaded (call /load first)")
        with self._gen_lock:
            kwargs: dict[str, Any] = dict(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=False,
            )
            if seed is not None:
                kwargs["seed"] = seed
            try:
                out = self._llm.create_chat_completion(**kwargs)
            except Exception as exc:  # surface upstream errors, not a generic 500
                raise RuntimeError(f"llama.cpp generation failed: {exc}") from exc
        try:
            content = out["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Unexpected LLM response shape: {exc}") from exc
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("LLM returned empty content")
        return content.strip()
