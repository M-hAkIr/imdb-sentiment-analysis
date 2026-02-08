import logging
import time
from dataclasses import dataclass

import torch
from transformers import pipeline

from src.config import (
    DEFAULT_BATCH_SIZE,
    LABEL_MAPPING,
    MAX_TOKEN_LENGTH,
    MIN_TEXT_LENGTH,
    Sentiment,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class PredictionResult:
    text: str
    label: str
    confidence: float
    model_name: str
    inference_time: float


def normalize_label(label: str) -> Sentiment:
    """Map model output to standard sentiment."""
    result = LABEL_MAPPING.get(label.upper())
    if result is None:
        raise ValueError(f"Unknown sentiment label: '{label}'")
    return result


class SentimentAnalyzer:
    """Wraps a HuggingFace sentiment pipeline with batch support."""

    def __init__(self, model_id: str, device: str | None = None,
                 batch_size: int = DEFAULT_BATCH_SIZE,
                 max_length: int = MAX_TOKEN_LENGTH):
        self.model_id = model_id
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        logger.info("Loading %s on %s", model_id, self.device)

        try:
            self.pipeline = pipeline(
                "sentiment-analysis",
                model=model_id,
                device=self.device,
                truncation=True,
                max_length=max_length,
            )
        except Exception as e:
            logger.error("Model loading failed: %s", e)
            raise RuntimeError(f"Failed to load model: {model_id}") from e

    def predict(self, text: str) -> PredictionResult:
        """Run sentiment analysis on a single text."""
        if not text or len(text.strip()) < MIN_TEXT_LENGTH:
            raise ValueError(f"Text too short (min {MIN_TEXT_LENGTH} chars)")

        start = time.perf_counter()
        result = self.pipeline(text)[0]
        elapsed = time.perf_counter() - start

        return PredictionResult(
            text=text,
            label=result['label'],
            confidence=result['score'],
            model_name=self.model_id,
            inference_time=elapsed,
        )

    def predict_batch(self, texts: list[str],
                      batch_size: int | None = None) -> list[PredictionResult]:
        """Run sentiment analysis on multiple texts using batched inference."""
        if not texts:
            raise ValueError("Empty text list")

        for i, text in enumerate(texts):
            if not text or len(text.strip()) < MIN_TEXT_LENGTH:
                raise ValueError(
                    f"Text at index {i} too short (min {MIN_TEXT_LENGTH} chars)"
                )

        batch_size = batch_size if batch_size is not None else self.batch_size
        logger.info("Processing %d texts (batch_size=%d)", len(texts), batch_size)

        try:
            start = time.perf_counter()
            results = self.pipeline(texts, batch_size=batch_size)
            total_time = time.perf_counter() - start
            avg_time = total_time / len(texts)

            logger.info(
                "Done: %d texts in %.2fs (%.1f texts/sec)",
                len(texts), total_time, len(texts) / total_time,
            )

            return [
                PredictionResult(
                    text=text,
                    label=result['label'],
                    confidence=result['score'],
                    model_name=self.model_id,
                    inference_time=avg_time,
                )
                for text, result in zip(texts, results)
            ]
        except Exception as e:
            logger.error("Batch prediction failed: %s", e)
            raise RuntimeError("Batch prediction error") from e


def create_analyzer(model_name: str, **kwargs) -> SentimentAnalyzer:
    from src.config import MODELS
    model_id = MODELS.get(model_name.lower(), model_name)
    return SentimentAnalyzer(model_id, **kwargs)
