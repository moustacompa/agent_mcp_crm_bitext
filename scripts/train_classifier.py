from __future__ import annotations

from src.crm_agent.classifier import train_classifier
from src.crm_agent.config import CLASSIFIER_PATH


def main() -> None:
    metrics = train_classifier()
    print(f"Accuracy validation: {metrics['validation_accuracy']:.4f}")
    print()
    print(metrics["classification_report"])
    print(f"Classificateur sauvegarde: {CLASSIFIER_PATH}")


if __name__ == "__main__":
    main()
