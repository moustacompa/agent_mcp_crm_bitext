from __future__ import annotations

from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd

import bootstrap  # noqa: F401

from src.data import count_templates, load_bitext_dataset


def main() -> None:
    df = load_bitext_dataset()
    print(f"Shape: {df.shape}")
    print(f"Colonnes: {list(df.columns)}")
    print("\n=== Infos generales ===")
    print(df.info())
    print("\n=== Valeurs manquantes ===")
    print(df.isnull().sum())
    print("\n=== Doublons ===")
    print(df.duplicated().sum())

    print("\n=== Categories ===")
    print(df["category"].value_counts())
    print("\n=== Intents ===")
    print(df["intent"].value_counts())

    flag_counts = df["flags"].value_counts()
    print("\n=== Top flags ===")
    print(flag_counts.head(20))

    all_flags: list[str] = []
    for flags_str in df["flags"].dropna():
        all_flags.extend(list(str(flags_str)))
    print("\n=== Flags individuels ===")
    print(Counter(all_flags))

    templates = count_templates(df["instruction"].tolist() + df["response"].tolist())
    print("\n=== Templates trouves ===")
    for template, count in templates.most_common(20):
        print(f"{{{{{template}}}}}: {count}")

    df["instruction_len"] = df["instruction"].apply(lambda value: len(str(value).split()))
    df["response_len"] = df["response"].apply(lambda value: len(str(value).split()))

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    df["category"].value_counts().plot(kind="bar", ax=axes[0], color="steelblue", title="Distribution des categories")
    df["intent"].value_counts().head(15).plot(kind="barh", ax=axes[1], color="coral", title="Top 15 intents")
    plt.tight_layout()
    plt.savefig("distribution_dataset.png", dpi=150, bbox_inches="tight")
    print("Graphique sauvegarde: distribution_dataset.png")

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    axes[0].hist(df["instruction_len"], bins=30, color="steelblue", edgecolor="white")
    axes[0].set_title("Distribution longueur des instructions")
    axes[1].hist(df["response_len"], bins=30, color="coral", edgecolor="white")
    axes[1].set_title("Distribution longueur des reponses")
    plt.tight_layout()
    plt.savefig("longueurs_texte.png", dpi=150, bbox_inches="tight")
    print("Graphique sauvegarde: longueurs_texte.png")


if __name__ == "__main__":
    main()
