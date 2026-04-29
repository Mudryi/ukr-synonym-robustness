"""Strict label parsing for LLM completions.

Returns ``None`` for unparseable outputs — never silently picks a default
class. Callers count parse-failure rate separately so we can spot prompt
issues instead of inflating accuracy with random labels.
"""

from __future__ import annotations

import re

from .prompts import PROMPTS


_REVIEWS_DIGIT = re.compile(r"\b([1-5])\b")
_UNLP_TOKEN = re.compile(r"\b(так|ні)\b")


def parse_label(raw: str, dataset: str) -> int | None:
    if raw is None:
        return None
    s = raw.strip().lower()
    if not s:
        return None

    vocab = PROMPTS[dataset]["label_vocab"]

    # Fast path: exact match (after strip+lower) — most well-instructed outputs.
    if s in vocab:
        return vocab[s]

    if dataset == "reviews":
        m = _REVIEWS_DIGIT.search(s)
        return int(m.group(1)) - 1 if m else None

    if dataset == "news":
        # longest-substring match against the 5 category names
        hits = [(name, s.find(name)) for name in vocab if name in s]
        if not hits:
            return None
        # pick the leftmost match (LLM usually produces the label first)
        hits.sort(key=lambda x: x[1])
        return vocab[hits[0][0]]

    if dataset == "unlp":
        m = _UNLP_TOKEN.search(s)
        return vocab[m.group(1)] if m else None

    return None
