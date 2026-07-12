"""Tests for FinBERTClassifier.

The actual ProsusAI/finbert model is not downloaded in tests — that would
take ~400 MB and 5+ seconds.  Tests instead mock the model + tokenizer to
exercise the classifier's parsing + ambiguity-threshold logic.
"""

from __future__ import annotations

import sys
from typing import Any
from unittest.mock import MagicMock

import pytest

from bot.sentiment.finbert import FinBERTClassifier

# ---------------------------------------------------------------------------
# Mocks for torch + transformers + optimum
# ---------------------------------------------------------------------------


class _FakeTensor:
    """Minimal torch-tensor stand-in: holds a list of floats, supports argmax/iter."""

    def __init__(self, values: list[float]) -> None:
        self._values = values

    def __iter__(self) -> Any:
        return iter(self._values)

    def __getitem__(self, idx: int) -> float:
        return self._values[idx]

    def argmax(self) -> Any:
        max_idx = max(range(len(self._values)), key=lambda i: self._values[i])
        return _FakeArgmax(max_idx)


class _FakeArgmax:
    def __init__(self, idx: int) -> None:
        self._idx = idx

    def item(self) -> int:
        return self._idx


class _FakeLogits:
    """Mimics outputs.logits[0] returning a tensor."""

    def __init__(self, values: list[float]) -> None:
        self._values = values

    def __getitem__(self, idx: int) -> _FakeTensor:
        return _FakeTensor(self._values)


class _FakeOutputs:
    def __init__(self, logits_values: list[float]) -> None:
        self.logits = _FakeLogits(logits_values)


def _install_fake_torch_and_transformers(
    monkeypatch: pytest.MonkeyPatch,
    logits: list[float],
    softmax_probs: list[float],
) -> None:
    """Install minimal torch / transformers / optimum stand-ins in sys.modules.

    ``softmax_probs`` is the expected output of softmax(logits) — passed
    directly so the test author controls the post-softmax distribution.
    """
    # torch
    torch_mod = MagicMock()
    torch_mod.no_grad.return_value.__enter__ = MagicMock(return_value=None)
    torch_mod.no_grad.return_value.__exit__ = MagicMock(return_value=False)
    torch_mod.nn.functional.softmax = MagicMock(return_value=_FakeTensor(softmax_probs))
    monkeypatch.setitem(sys.modules, "torch", torch_mod)

    # transformers
    transformers_mod = MagicMock()

    def _tokenize(text: str, **kwargs: Any) -> dict[str, Any]:
        return {"input_ids": [1, 2, 3]}

    tokenizer_mock = MagicMock(side_effect=_tokenize)
    tokenizer_mock.return_value = {"input_ids": [1, 2, 3]}
    transformers_mod.AutoTokenizer.from_pretrained = MagicMock(return_value=tokenizer_mock)
    monkeypatch.setitem(sys.modules, "transformers", transformers_mod)

    # optimum.onnxruntime
    optimum_mod = MagicMock()

    def _make_model() -> Any:
        model = MagicMock()
        model.return_value = _FakeOutputs(logits)
        return model

    optimum_mod.ORTModelForSequenceClassification.from_pretrained = MagicMock(
        return_value=_make_model()
    )
    monkeypatch.setitem(sys.modules, "optimum", optimum_mod)
    monkeypatch.setitem(sys.modules, "optimum.onnxruntime", optimum_mod)


# ---------------------------------------------------------------------------
# Constructor / threshold validation
# ---------------------------------------------------------------------------


def test_constructor_rejects_invalid_threshold() -> None:
    with pytest.raises(ValueError):
        FinBERTClassifier(high_confidence_threshold=0.0)
    with pytest.raises(ValueError):
        FinBERTClassifier(high_confidence_threshold=-0.1)
    with pytest.raises(ValueError):
        FinBERTClassifier(high_confidence_threshold=1.5)


def test_constructor_accepts_valid_threshold() -> None:
    fb = FinBERTClassifier(high_confidence_threshold=0.4)
    assert fb._threshold == 0.4


def test_classify_before_load_raises() -> None:
    fb = FinBERTClassifier()
    with pytest.raises(RuntimeError):
        fb.classify("Bitcoin rallies on Fed pause")


# ---------------------------------------------------------------------------
# Ambiguity gate (pure logic — no mocks needed)
# ---------------------------------------------------------------------------


def test_is_ambiguous_below_threshold() -> None:
    fb = FinBERTClassifier(high_confidence_threshold=0.4)
    assert fb.is_ambiguous({"score": 0.2, "confidence": 0.5, "label": "neutral"})
    assert fb.is_ambiguous({"score": -0.3, "confidence": 0.5, "label": "negative"})


def test_is_ambiguous_above_threshold() -> None:
    fb = FinBERTClassifier(high_confidence_threshold=0.4)
    assert not fb.is_ambiguous({"score": 0.6, "confidence": 0.9, "label": "positive"})
    assert not fb.is_ambiguous({"score": -0.7, "confidence": 0.85, "label": "negative"})


def test_is_ambiguous_non_numeric_score_defaults_ambiguous() -> None:
    fb = FinBERTClassifier(high_confidence_threshold=0.4)
    assert fb.is_ambiguous({"score": "not-a-number", "label": "neutral"})


# ---------------------------------------------------------------------------
# Classify pipeline (with mocked torch + transformers + optimum)
# ---------------------------------------------------------------------------


def test_classify_returns_signed_score_positive(monkeypatch: pytest.MonkeyPatch) -> None:
    # softmax → [0.80, 0.10, 0.10] (positive / negative / neutral)
    _install_fake_torch_and_transformers(monkeypatch, [3.0, 0.5, 0.5], [0.8, 0.1, 0.1])
    fb = FinBERTClassifier()
    fb.load()
    result = fb.classify("Markets rally on Fed pause")
    assert result["label"] == "positive"
    assert result["score"] == pytest.approx(0.7, abs=1e-4)
    assert result["confidence"] == pytest.approx(0.8, abs=1e-4)


def test_classify_returns_signed_score_negative(monkeypatch: pytest.MonkeyPatch) -> None:
    # softmax → [0.05, 0.85, 0.10] → negative dominant
    _install_fake_torch_and_transformers(monkeypatch, [0.1, 3.0, 0.5], [0.05, 0.85, 0.10])
    fb = FinBERTClassifier()
    fb.load()
    result = fb.classify("Recession fears mount")
    assert result["label"] == "negative"
    assert result["score"] == pytest.approx(-0.80, abs=1e-4)
    assert result["confidence"] == pytest.approx(0.85, abs=1e-4)


def test_classify_neutral(monkeypatch: pytest.MonkeyPatch) -> None:
    # softmax → [0.20, 0.20, 0.60] → neutral
    _install_fake_torch_and_transformers(monkeypatch, [1.0, 1.0, 2.5], [0.2, 0.2, 0.6])
    fb = FinBERTClassifier()
    fb.load()
    result = fb.classify("Markets trade sideways")
    assert result["label"] == "neutral"
    assert result["score"] == pytest.approx(0.0, abs=1e-4)


def test_classify_batch_returns_one_per_input(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_torch_and_transformers(monkeypatch, [3.0, 0.5, 0.5], [0.8, 0.1, 0.1])
    fb = FinBERTClassifier()
    fb.load()
    results = fb.classify_batch(["a", "b", "c"])
    assert len(results) == 3
    assert all(r["label"] == "positive" for r in results)


# ---------------------------------------------------------------------------
# Load: error path when optional deps absent
# ---------------------------------------------------------------------------


def test_load_raises_clean_error_when_optimum_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """If optimum/transformers are not installed, ``load()`` should raise ImportError
    with an actionable message rather than crashing deeper in the stack."""
    real_import = (
        __builtins__["__import__"] if isinstance(__builtins__, dict) else __builtins__.__import__
    )

    def _fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name.startswith("optimum") or name.startswith("transformers"):
            raise ImportError(f"No module named {name!r}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", _fake_import)
    fb = FinBERTClassifier()
    with pytest.raises(ImportError, match="finbert"):
        fb.load()


# ---------------------------------------------------------------------------
# Async wrappers
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_classify_async_delegates_to_sync(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_torch_and_transformers(monkeypatch, [3.0, 0.5, 0.5], [0.8, 0.1, 0.1])
    fb = FinBERTClassifier()
    fb.load()
    result = await fb.classify_async("bullish news")
    assert result["label"] == "positive"


@pytest.mark.asyncio
async def test_classify_batch_async(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_torch_and_transformers(monkeypatch, [3.0, 0.5, 0.5], [0.8, 0.1, 0.1])
    fb = FinBERTClassifier()
    fb.load()
    out = await fb.classify_batch_async(["x", "y"])
    assert len(out) == 2
