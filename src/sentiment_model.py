"""Sentiment analysis model wrapper using Hugging Face Transformers."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import torch
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

from .preprocessing import clean_text

logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
NEUTRAL_THRESHOLD = 0.55
MAX_TOKEN_LENGTH = 512


@dataclass
class SentimentResult:


    label: str
    confidence: float
    probabilities: dict[str, float]
    raw_label: str
    raw_confidence: float

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "confidence": self.confidence,
            "probabilities": self.probabilities,
            "raw_label": self.raw_label,
            "raw_confidence": self.raw_confidence,
        }


class SentimentAnalyzer:


    _instance: Optional["SentimentAnalyzer"] = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        neutral_threshold: float = NEUTRAL_THRESHOLD,
    ):
        if hasattr(self, "_initialized"):
            return
        self.model_name = model_name
        self.neutral_threshold = neutral_threshold
        self.tokenizer = None
        self.model = None
        self.id2label: dict[int, str] = {}
        self._initialized = True

    def load(self) -> None:

        if self.model is not None and self.tokenizer is not None:
            return

        logger.info("Loading model: %s", self.model_name)
        try:
            config = AutoConfig.from_pretrained(self.model_name)
            self.id2label = {
                int(k): v.upper() for k, v in config.id2label.items()
            }

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name
            )
            self.model.eval()
            logger.info("Model loaded successfully. Labels: %s", self.id2label)
        except Exception as exc:
            self.tokenizer = None
            self.model = None
            raise RuntimeError(
                f"Failed to load model '{self.model_name}': {exc}"
            ) from exc

    def is_loaded(self) -> bool:

        return self.model is not None and self.tokenizer is not None

    @torch.no_grad()
    def predict(self, text: str) -> SentimentResult:

        if not self.is_loaded():
            raise RuntimeError("Model is not loaded. Call .load() first.")

        cleaned = clean_text(text)
        if not cleaned:
            raise ValueError("Text is empty after preprocessing.")

        inputs = self.tokenizer(
            cleaned,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_TOKEN_LENGTH,
            padding=True,
        )

        outputs = self.model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)[0]

        raw_confidence = float(probs.max().item())
        raw_label_idx = int(probs.argmax().item())
        raw_label = self.id2label.get(raw_label_idx, f"LABEL_{raw_label_idx}")

        probabilities = {
            self.id2label.get(i, f"LABEL_{i}"): float(p.item())
            for i, p in enumerate(probs)
        }

        if raw_confidence < self.neutral_threshold:
            final_label = "NEUTRAL"
        else:
            final_label = raw_label

        final_confidence = raw_confidence if final_label != "NEUTRAL" else (
            1.0 - raw_confidence
        )

        return SentimentResult(
            label=final_label,
            confidence=final_confidence,
            probabilities=probabilities,
            raw_label=raw_label,
            raw_confidence=raw_confidence,
        )

    def predict_batch(self, texts: list[str]) -> list[SentimentResult]:

        results: list[SentimentResult] = []
        for text in texts:
            try:
                results.append(self.predict(text))
            except (ValueError, RuntimeError) as exc:
                logger.warning("Prediction failed for a batch item: %s", exc)
                results.append(
                    SentimentResult(
                        label="NEUTRAL",
                        confidence=0.0,
                        probabilities={},
                        raw_label="ERROR",
                        raw_confidence=0.0,
                    )
                )
        return results


def get_analyzer(
    model_name: str = DEFAULT_MODEL_NAME,
) -> SentimentAnalyzer:

    return SentimentAnalyzer(model_name=model_name)