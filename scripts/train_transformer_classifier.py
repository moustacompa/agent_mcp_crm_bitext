from __future__ import annotations

import argparse
import inspect
import time
from pathlib import Path

import bootstrap  # noqa: F401
import pandas as pd
from datasets import Dataset
from sklearn.metrics import accuracy_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

from src.config import TRAIN_CSV, TRANSFORMER_CLASSIFIER_DIR, VAL_CSV


def text_column(df: pd.DataFrame) -> pd.Series:
    """Use the user instruction only for fast intent classification."""

    if "instruction_clean" in df.columns:
        return df["instruction_clean"].fillna(df["instruction"])
    return df["instruction"].fillna("")


def freeze_transformer_base(model) -> None:
    """Freeze the encoder and train only the lightweight classification head."""

    base_model = getattr(model, "base_model", None)
    if base_model is None:
        return
    for parameter in base_model.parameters():
        parameter.requires_grad = False


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a lightweight BERT-like intent classifier.")
    parser.add_argument(
        "--model-name",
        default="google/bert_uncased_L-2_H-128_A-2",
        help="Small BERT model. Use distilbert-base-uncased for better quality but slower CPU training.",
    )
    parser.add_argument("--output-dir", default=str(TRANSFORMER_CLASSIFIER_DIR))
    parser.add_argument("--epochs", type=float, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=96)
    parser.add_argument(
        "--sample-size",
        type=int,
        default=0,
        help="Optional number of training rows to keep for very fast CPU experiments. 0 keeps all rows.",
    )
    parser.add_argument(
        "--fine-tune-base",
        action="store_true",
        help="Also train the Transformer encoder. Slower on CPU, but can improve quality.",
    )
    args = parser.parse_args()

    started_at = time.perf_counter()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(TRAIN_CSV)
    val_df = pd.read_csv(VAL_CSV)
    if args.sample_size and args.sample_size < len(train_df):
        train_df = train_df.sample(n=args.sample_size, random_state=42).reset_index(drop=True)

    train_df["text"] = text_column(train_df)
    val_df["text"] = text_column(val_df)

    labels = sorted(train_df["intent"].unique().tolist())
    label2id = {label: index for index, label in enumerate(labels)}
    id2label = {index: label for label, index in label2id.items()}
    train_df["label"] = train_df["intent"].map(label2id)
    val_df["label"] = val_df["intent"].map(label2id)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=args.max_length,
        )

    train_ds = Dataset.from_pandas(train_df[["text", "label"]]).map(tokenize, batched=True)
    val_ds = Dataset.from_pandas(val_df[["text", "label"]]).map(tokenize, batched=True)
    train_ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    val_ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
    )
    if not args.fine_tune_base:
        freeze_transformer_base(model)

    def compute_metrics(eval_pred):
        predictions, labels_array = eval_pred
        preds = predictions.argmax(axis=-1)
        return {"accuracy": accuracy_score(labels_array, preds)}

    training_kwargs = {
        "output_dir": str(output_dir),
        "save_strategy": "epoch",
        "save_total_limit": 1,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "num_train_epochs": args.epochs,
        "learning_rate": 5e-4 if not args.fine_tune_base else 2e-5,
        "weight_decay": 0.01,
        "logging_dir": "data/logs",
        "logging_steps": 20,
        "report_to": "none",
    }
    strategy_key = "eval_strategy" if "eval_strategy" in inspect.signature(TrainingArguments).parameters else "evaluation_strategy"
    training_kwargs[strategy_key] = "epoch"
    training_args = TrainingArguments(**training_kwargs)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    metrics = trainer.evaluate()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    elapsed = time.perf_counter() - started_at
    print(f"Modele Transformer sauvegarde dans: {output_dir}")
    print(f"Accuracy validation: {metrics.get('eval_accuracy', 0):.4f}")
    print(f"Temps total: {elapsed:.1f} secondes")
    if not args.fine_tune_base:
        print("Mode rapide actif: encodeur gele, seule la tete de classification est entrainee.")


if __name__ == "__main__":
    main()
