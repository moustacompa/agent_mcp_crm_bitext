"""Hugging Face Transformer intent classifier wrapper.

This module makes a saved BERT-like sequence classification model expose the
same small API as the scikit-learn classifier used by the agent:
`predict()` and `predict_proba()`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .config import TRANSFORMER_CLASSIFIER_DIR


class TransformerIntentClassifier:
    """Adapter around a saved Hugging Face sequence classifier."""

    def __init__(self, model_dir: str | Path = TRANSFORMER_CLASSIFIER_DIR, max_length: int = 128) -> None:
        self.model_dir = Path(model_dir)
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_dir)
        self.model.eval()
        self.classes_ = np.array([self.model.config.id2label[index] for index in range(self.model.config.num_labels)])

    def predict_proba(self, messages: list[str] | tuple[str, ...]) -> np.ndarray:
        encoded = self.tokenizer(
            list(messages),
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        with torch.no_grad():
            logits = self.model(**encoded).logits
            probabilities = torch.softmax(logits, dim=-1)
        return probabilities.cpu().numpy()

    def predict(self, messages: list[str] | tuple[str, ...]) -> np.ndarray:
        probabilities = self.predict_proba(messages)
        indexes = probabilities.argmax(axis=1)
        return self.classes_[indexes]


def load_transformer_classifier(
    model_dir: str | Path = TRANSFORMER_CLASSIFIER_DIR,
    max_length: int = 128,
) -> TransformerIntentClassifier:
    return TransformerIntentClassifier(model_dir=model_dir, max_length=max_length)
