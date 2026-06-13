"""TF-IDF + LogisticRegression intent classifier."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline

from .config import CLASSIFIER_PATH, TEST_CSV, TRAIN_CSV, VAL_CSV


def _text_column(df: pd.DataFrame) -> pd.Series:
    if "instruction_clean" in df.columns:
        return df["instruction_clean"].fillna(df["instruction"])
    return df["instruction"].fillna("")


def build_classifier() -> Pipeline:
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=10000, sublinear_tf=True)),
            ("clf", LogisticRegression(max_iter=500, C=5.0, random_state=42)),
        ]
    )


def train_classifier(
    train_csv: str | Path = TRAIN_CSV,
    val_csv: str | Path = VAL_CSV,
    output_path: str | Path = CLASSIFIER_PATH,
) -> dict[str, Any]:
    train_data = pd.read_csv(train_csv)
    val_data = pd.read_csv(val_csv)

    model = build_classifier()
    model.fit(_text_column(train_data), train_data["intent"])

    val_predictions = model.predict(_text_column(val_data))
    metrics = {
        "validation_accuracy": accuracy_score(val_data["intent"], val_predictions),
        "classification_report": classification_report(val_data["intent"], val_predictions),
    }

    with Path(output_path).open("wb") as output:
        pickle.dump(model, output)

    return metrics


def load_classifier(path: str | Path = CLASSIFIER_PATH) -> Pipeline:
    with Path(path).open("rb") as model_file:
        return pickle.load(model_file)


def evaluate_classifier(
    classifier: Pipeline,
    test_csv: str | Path = TEST_CSV,
) -> dict[str, Any]:
    test_data = pd.read_csv(test_csv)
    x_test = _text_column(test_data)
    y_test = test_data["intent"]
    y_pred = classifier.predict(x_test)
    y_proba = classifier.predict_proba(x_test).max(axis=1)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "mean_confidence": float(y_proba.mean()),
        "high_confidence_ratio": float((y_proba > 0.8).mean()),
        "classification_report": classification_report(y_test, y_pred),
        "y_test": y_test,
        "y_pred": y_pred,
    }


def predict_intent(classifier: Pipeline, message: str) -> tuple[str, float]:
    intent = classifier.predict([message])[0]
    confidence = float(classifier.predict_proba([message]).max())
    return intent, confidence
