"""Data loading, cleaning and formatting utilities for the Bitext CRM dataset."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterable

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import DATASET_PATH, TEST_CSV, TRAIN_CSV, TRAIN_JSONL, VAL_CSV, VAL_JSONL


TEMPLATE_REPLACEMENTS = {
    'Order Number': '#ORD-12345',
    'Customer Support Hours': '9h-18h du lundi au vendredi',
    'Customer Support Phone Number': '+1-800-555-0100',
    'Online Company Portal Info': 'notre portail en ligne (www.example.com)',
    'Online Order Interaction': 'Mes Commandes',
    'Website URL': 'www.example.com',
    'Live Chat': 'chat en direct',
    'Return Policy': 'politique de retour sous 30 jours',
    'Refund Policy': 'politique de remboursement sous 5-10 jours ouvrés',
    'Product Name': 'Produit XYZ',
    'Account Type': 'compte standard',
    'Operator': 'notre équipe support',
}


def load_bitext_dataset(csv_path: str | Path = DATASET_PATH) -> pd.DataFrame:
    """Load the local Bitext CSV file."""

    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset introuvable: {path}. Place le CSV Bitext a cet emplacement "
            "ou passe --csv-path au script de preparation."
        )
    return pd.read_csv(path)


def extract_templates(text: object) -> list[str]:
    """Return template names found in a text, e.g. {{Order Number}}."""

    return re.findall(r"\{\{([^}]+)\}\}", str(text))


def count_templates(texts: Iterable[object]) -> Counter:
    templates: list[str] = []
    for text in texts:
        templates.extend(extract_templates(text))
    return Counter(templates)


def replace_templates(text: object) -> str:
    """Replace known Bitext placeholders with demo values."""

    value = str(text)
    for template, replacement in TEMPLATE_REPLACEMENTS.items():
        value = value.replace(f"{{{{{template}}}}}", replacement)
    return re.sub(r"\{\{[^}]+\}\}", "[INFO]", value)


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Add cleaned instruction/response columns used by the models."""

    cleaned = df.copy()
    cleaned["instruction_clean"] = cleaned["instruction"].apply(replace_templates)
    cleaned["response_clean"] = cleaned["response"].apply(replace_templates)
    return cleaned


def split_dataset(
    df: pd.DataFrame,
    test_size: float = 0.2,
    val_size_within_temp: float = 0.5,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create train/validation/test splits stratified by intent."""

    train_df, temp_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df["intent"],
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=val_size_within_temp,
        random_state=random_state,
        stratify=temp_df["intent"],
    )
    return train_df, val_df, test_df


def format_for_finetuning(row: pd.Series) -> dict[str, str]:
    """Format one row as an instruction/answer training sample."""

    text = (
        "<|system|>\n"
        "Tu es un agent CRM specialise dans le support client. Identifie "
        "l'intention de l'utilisateur et reponds de maniere professionnelle "
        "et empathique.<|end|>\n"
        "<|user|>\n"
        f"{row['instruction_clean']}<|end|>\n"
        "<|assistant|>\n"
        f"{row['response_clean']}<|end|>"
    )
    return {"text": text, "intent": row["intent"], "category": row["category"]}


def write_jsonl(records: Iterable[dict], output_path: str | Path) -> None:
    with Path(output_path).open("w", encoding="utf-8") as output:
        for item in records:
            output.write(json.dumps(item, ensure_ascii=False) + "\n")


def prepare_dataset(
    csv_path: str | Path = DATASET_PATH,
    train_csv: str | Path = TRAIN_CSV,
    val_csv: str | Path = VAL_CSV,
    test_csv: str | Path = TEST_CSV,
    train_jsonl: str | Path = TRAIN_JSONL,
    val_jsonl: str | Path = VAL_JSONL,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load, clean, split and save all local training artifacts."""

    df = clean_dataset(load_bitext_dataset(csv_path))
    train_df, val_df, test_df = split_dataset(df)

    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)
    test_df.to_csv(test_csv, index=False)

    write_jsonl((format_for_finetuning(row) for _, row in train_df.iterrows()), train_jsonl)
    write_jsonl((format_for_finetuning(row) for _, row in val_df.iterrows()), val_jsonl)

    return train_df, val_df, test_df
