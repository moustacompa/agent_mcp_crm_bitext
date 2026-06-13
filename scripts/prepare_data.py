from __future__ import annotations

import argparse

import bootstrap  # noqa: F401

from src.data import prepare_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Bitext CRM train/val/test files.")
    parser.add_argument("--csv-path", default=None, help="Path to the Bitext CSV dataset.")
    args = parser.parse_args()

    kwargs = {"csv_path": args.csv_path} if args.csv_path else {}
    train_df, val_df, test_df = prepare_dataset(**kwargs)

    print(f"Train: {len(train_df)} exemples")
    print(f"Validation: {len(val_df)} exemples")
    print(f"Test: {len(test_df)} exemples")
    print("Fichiers crees dans data/: bitext_train.csv, bitext_val.csv, bitext_test.csv")
    print("Fichiers JSONL crees dans data/: bitext_train_formatted.jsonl, bitext_val_formatted.jsonl")


if __name__ == "__main__":
    main()
