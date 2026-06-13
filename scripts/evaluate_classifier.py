from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

import bootstrap  # noqa: F401

from src.classifier import evaluate_classifier, load_classifier


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the intent classifier.")
    parser.add_argument("--plot", action="store_true", help="Save confusion_matrix.png for top intents.")
    parser.add_argument("--classifier-backend", choices=["tfidf", "transformer"], default="tfidf")
    args = parser.parse_args()

    if args.classifier_backend == "transformer":
        from src.transformer_classifier import load_transformer_classifier

        classifier = load_transformer_classifier()
    else:
        classifier = load_classifier()
    metrics = evaluate_classifier(classifier)
    print(f"Accuracy globale: {metrics['accuracy']:.4f}")
    print(f"Confiance moyenne: {metrics['mean_confidence']:.4f}")
    print(f"Ratio confiance > 80%: {metrics['high_confidence_ratio']:.1%}")
    print()
    print(metrics["classification_report"])

    if args.plot:
        y_test = metrics["y_test"]
        y_pred = pd.Series(metrics["y_pred"], index=y_test.index)
        top_intents = y_test.value_counts().head(10).index.tolist()
        mask = y_test.isin(top_intents)
        cm = confusion_matrix(y_test[mask], y_pred[mask], labels=top_intents)
        cm_norm = cm.astype("float") / cm.sum(axis=1)[:, None]

        fig, ax = plt.subplots(figsize=(12, 9))
        sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues", xticklabels=top_intents, yticklabels=top_intents, ax=ax)
        ax.set_title("Matrice de confusion normalisee - Top 10 intents")
        ax.set_ylabel("Intent reel")
        ax.set_xlabel("Intent predit")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig("confusion_matrix.png", dpi=150, bbox_inches="tight")
        print("Matrice sauvegardee: confusion_matrix.png")


if __name__ == "__main__":
    main()
