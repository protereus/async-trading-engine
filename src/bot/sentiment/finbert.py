"""FinBERTClassifier — local on-CPU per-headline sentiment triage.

The deep-research report recommended FinBERT (ProsusAI/finbert) as a free
local pre-filter ahead of the cloud LLM tier.  ~110M-param BERT-base
fine-tuned on Financial PhraseBank; published accuracy ~86–97% depending on
inter-annotator agreement level.  Inference is fast enough on CPU (estimated
200–400 ms per 200-token input on 4 vCPU) that headline-level triage is
feasible even alongside Kronos.

Role in
  - NewsAgent calls FinBERTClassifier on each headline before any LLM dispatch
  - Headlines with confident polarity (|score| ≥ ``high_confidence_threshold``)
    are surfaced directly — no Cerebras call
  - Ambiguous headlines (|score| below the threshold) are batched into the
    aggregated Cerebras prompt for richer cross-headline reasoning

Cost: ONNX-quantised model resident memory ~120 MB; first load ~3–5 s while
the model converts from PyTorch.  Subsequent inferences are ~200–400 ms per
200-token headline.  Gated by ``BotConfig.finbert_enabled`` so the bot can
run without ``optimum`` / ``transformers`` installed.

This module **does not** import transformers/optimum at module load — they
are pulled in lazily by ``FinBERTClassifier.load()``.  Skipping the load
keeps test imports fast and the runtime image small for users who don't
opt into the local pre-filter.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

logger = logging.getLogger(__name__)

# ProsusAI/finbert label ordering observed via tokenizer.config:
#   0 = positive, 1 = negative, 2 = neutral
# We map to a signed score in [-1, +1]: score = P(positive) - P(negative).
_MODEL_ID = "ProsusAI/finbert"


class FinBERTClassifier:
    """Per-headline financial-sentiment classifier with lazy ONNX backend.

    Construction is cheap (no model load).  Call ``load()`` once before the
    first ``classify`` invocation; subsequent calls reuse the cached model.
    """

    def __init__(
        self,
        high_confidence_threshold: float = 0.4,
        model_id: str = _MODEL_ID,
    ) -> None:
        if not 0.0 < high_confidence_threshold <= 1.0:
            raise ValueError("high_confidence_threshold must be in (0, 1]")
        self._threshold = high_confidence_threshold
        self._model_id = model_id
        self._model: Any = None
        self._tokenizer: Any = None
        self._loaded = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Pull transformers/optimum, download FinBERT, prepare for inference.

        Raises ``ImportError`` with a clear message if optional deps are absent.
        """
        if self._loaded:
            return
        try:
            from optimum.onnxruntime import ORTModelForSequenceClassification  # noqa: PLC0415
            from transformers import AutoTokenizer  # noqa: PLC0415
        except ImportError as exc:
            raise ImportError(
                "FinBERT requires `optimum[onnxruntime]` + `transformers`. "
                "Install via `uv sync --extra finbert`."
            ) from exc

        logger.info("FinBERT: loading %s (ONNX, CPU)", self._model_id)
        self._tokenizer = AutoTokenizer.from_pretrained(self._model_id)
        self._model = ORTModelForSequenceClassification.from_pretrained(self._model_id, export=True)
        self._loaded = True
        logger.info("FinBERT: model loaded")

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def classify(self, text: str) -> dict[str, float | str]:
        """Classify one short string.  Returns ``{score, confidence, label}``.

        - ``score``: signed in [-1, +1] (positive minus negative probability)
        - ``confidence``: max-probability of the predicted class in [0, 1]
        - ``label``: one of "positive" / "negative" / "neutral"

        The classifier is deterministic at temperature 0 (logits → softmax)
        so identical inputs always produce the same output.
        """
        if not self._loaded:
            raise RuntimeError("FinBERTClassifier.load() must be called first")
        import torch

        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=256,  # headlines + brief context fit comfortably
        )
        with torch.no_grad():
            outputs = self._model(**inputs)
        logits = outputs.logits[0]
        probs = torch.nn.functional.softmax(logits, dim=-1)
        # ProsusAI label ordering: 0=positive, 1=negative, 2=neutral
        p_pos, p_neg, p_neu = (float(p) for p in probs)
        score = p_pos - p_neg
        max_idx = int(probs.argmax().item())
        label = ("positive", "negative", "neutral")[max_idx]
        confidence = float(probs[max_idx])
        return {
            "score": round(score, 4),
            "confidence": round(confidence, 4),
            "label": label,
        }

    # ------------------------------------------------------------------
    # Bulk + ambiguity gating
    # ------------------------------------------------------------------

    def classify_batch(self, texts: list[str]) -> list[dict[str, float | str]]:
        """Classify a list of texts.  Returns one dict per input."""
        return [self.classify(t) for t in texts]

    def is_ambiguous(self, classified: dict[str, float | str]) -> bool:
        """Whether the classified result falls below the high-confidence threshold.

        Ambiguous headlines should escalate to the cloud LLM tier for richer
        cross-document reasoning; confident ones can be used directly.
        """
        score = classified.get("score", 0.0)
        if not isinstance(score, int | float):
            return True
        return abs(score) < self._threshold

    # ------------------------------------------------------------------
    # Async wrapper — heavy work runs in a thread pool to avoid blocking the loop
    # ------------------------------------------------------------------

    async def classify_async(self, text: str) -> dict[str, float | str]:
        return await asyncio.to_thread(self.classify, text)

    async def classify_batch_async(self, texts: list[str]) -> list[dict[str, float | str]]:
        return await asyncio.to_thread(self.classify_batch, texts)
