"""Unified dataset loader for the three datasets (reviews, news, unlp).

Both attacks previously had their own loader (`textfooler/dataloader.py:read_corpus`
returned token lists; `bert_attack/main.py:get_data_cls` returned strings). The
unified loader returns a list of ``(text: str, label: int)`` tuples — attacks
can tokenize on their own when needed (TextFooler does, BERT-Attack does too).

The fixed sub-sample seed (1914) and 10 000-row cap are preserved from the
original code so existing experiments remain reproducible.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


NEWS_LABELS = ["бізнес", "новини", "політика", "спорт", "технології"]
NEWS_LABEL2ID = {label: i for i, label in enumerate(NEWS_LABELS)}


def load_dataset(
    name: str,
    path: str | Path,
    *,
    text_col: str | None = None,
    label_col: str | None = None,
    max_rows: int = 10_000,
    sample_seed: int = 1914,
) -> list[tuple[str, int]]:
    """Load a CSV dataset and normalize labels to zero-indexed ints."""
    df = pd.read_csv(path)
    if len(df) > max_rows:
        df = df.sample(max_rows, random_state=sample_seed)

    if name == "reviews":
        tcol = text_col or "text"
        lcol = label_col or "label"
        return [(str(row[tcol]), int(row[lcol]) - 1) for _, row in df.iterrows()]

    if name == "news":
        tcol = text_col or "title"
        lcol = label_col or "target"
        df[lcol] = df[lcol].map(NEWS_LABEL2ID)
        out: list[tuple[str, int]] = []
        for _, row in df.iterrows():
            label = row[lcol]
            if pd.isna(label):
                continue
            out.append((str(row[tcol]), int(label)))
        return out

    if name == "unlp":
        tcol = text_col or "text"
        lcol = label_col or "label"
        return [(str(row[tcol]), int(row[lcol])) for _, row in df.iterrows()]

    raise ValueError(f"unknown dataset name: {name!r}")
