from __future__ import annotations

import math
import re
from dataclasses import dataclass

import numpy as np

try:
    from transformers import pipeline
except Exception:  # pragma: no cover - optional dependency
    pipeline = None


POSITIVE_TERMS = {
    "beat", "beats", "strong", "surge", "surges", "growth", "accelerates",
    "raised", "raise", "upside", "expands", "profit", "profits",
    "record", "wins", "win", "outperform", "outperforms", "bullish",
    "approval", "approved", "momentum", "improves", "improved",
}

NEGATIVE_TERMS = {
    "miss", "misses", "weak", "plunge", "plunges", "decline", "declines",
    "cuts", "cut", "downside", "compresses", "loss", "losses",
    "warning", "warns", "lawsuit", "probe", "recall", "bearish",
    "downgrade", "downgrades", "slows", "slowing", "deteriorates",
}


@dataclass(slots=True)
class SentimentScore:
    score: float
    confidence: float
    label: str
    model_name: str


class LexiconSentimentModel:
    def __init__(self, model_name: str = "lexicon-finance-v1"):
        self.model_name = model_name

    def score_text(self, text: str) -> SentimentScore:
        tokens = re.findall(r"[A-Za-z']+", (text or "").lower())
        if not tokens:
            return SentimentScore(0.0, 0.0, "neutral", self.model_name)

        pos = sum(tok in POSITIVE_TERMS for tok in tokens)
        neg = sum(tok in NEGATIVE_TERMS for tok in tokens)
        raw = pos - neg
        denom = max(pos + neg, 1)
        score = float(np.clip(raw / denom, -1.0, 1.0))
        confidence = float(np.clip(math.sqrt(denom) / 4.0, 0.1, 0.95))
        label = "positive" if score > 0.1 else "negative" if score < -0.1 else "neutral"
        return SentimentScore(score=score, confidence=confidence, label=label, model_name=self.model_name)


class HuggingFaceSentimentModel:
    def __init__(self, model_name: str = "ProsusAI/finbert"):
        self.model_name = model_name
        self._pipeline = None

    def _ensure_pipeline(self):
        if self._pipeline is None:
            if pipeline is None:  # pragma: no cover - depends on optional package
                raise RuntimeError("transformers is not installed")
            self._pipeline = pipeline("text-classification", model=self.model_name)
        return self._pipeline

    def score_text(self, text: str) -> SentimentScore:
        if not text:
            return SentimentScore(0.0, 0.0, "neutral", self.model_name)
        result = self._ensure_pipeline()(text[:2048])[0]  # pragma: no cover - optional path
        label = str(result.get("label", "neutral")).lower()
        confidence = float(np.clip(result.get("score", 0.0), 0.0, 1.0))
        if "pos" in label:
            score = confidence
            label = "positive"
        elif "neg" in label:
            score = -confidence
            label = "negative"
        else:
            score = 0.0
            label = "neutral"
        return SentimentScore(score=float(np.clip(score, -1.0, 1.0)), confidence=confidence, label=label, model_name=self.model_name)


def build_sentiment_model(prefer_hf: bool = True, model_name: str = "ProsusAI/finbert"):
    if prefer_hf and pipeline is not None:
        try:
            return HuggingFaceSentimentModel(model_name=model_name)
        except Exception:
            pass
    return LexiconSentimentModel()
