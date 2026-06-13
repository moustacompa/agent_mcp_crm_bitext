from __future__ import annotations

import argparse

import pandas as pd
from datasets import Dataset
from sklearn.metrics import accuracy_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments

from src.crm_agent.config import TRAIN_CSV, VAL_CSV


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a DistilBERT intent classifier.")
    parser.add_argument("--model-name", default="distilbert-base-uncased")
    parser.add_argument("--output-dir", default="distilbert_intent_model")
    parser.add_argument("--epochs", type=int, default=2)
    args = parser.parse_args()

    train_df = pd.read_csv(TRAIN_CSV)
    val_df = pd.read_csv(VAL_CSV)
    train_df["text"] = train_df["instruction_clean"].fillna("") + " " + train_df["response_clean"].fillna("")
    val_df["text"] = val_df["instruction_clean"].fillna("") + " " + val_df["response_clean"].fillna("")

    labels = sorted(train_df["intent"].unique().tolist())
    label2id = {label: index for index, label in enumerate(labels)}
    id2label = {index: label for label, index in label2id.items()}
    train_df["label"] = train_df["intent"].map(label2id)
    val_df["label"] = val_df["intent"].map(label2id)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length")

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

    def compute_metrics(eval_pred):
        predictions, labels_array = eval_pred
        preds = predictions.argmax(axis=-1)
        return {"accuracy": accuracy_score(labels_array, preds)}

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    print(f"Modele sauvegarde dans: {args.output_dir}")


if __name__ == "__main__":
    main()
