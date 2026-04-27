"""Tokenization helpers shared by both attacks.

`tokenize_ukrainian` keeps whitespace and punctuation as separate tokens so the
attack can later reconstruct the exact original text via ``''.join(tokens)``.
Apostrophe-joined Ukrainian forms (e.g. "п'ять") are merged into one word
token. `tokenize_with_whitespace` is a simpler regex variant kept for the
BERT-attack code path which tokenises by ``re.findall``.
"""

from __future__ import annotations

import re

_APOSTROPHE_PATTERN = r"(\s+|[^\w\s']+|[\w']+)"
_WS_PATTERN = r"(\s+|[^\w\s]+|\w+)"

_LATIN_RE = re.compile(r"[A-Za-z]")
_RUS_EXCLUSIVE_RE = re.compile(r"[ЁёЫыЭэЪъ]")


def tokenize_ukrainian(text: str) -> list[str]:
    """Tokenize Ukrainian text preserving whitespace and punctuation as tokens.

    Apostrophe-joined word fragments (e.g. ``п'ять``) are merged into one token.
    """
    tokens = re.findall(_APOSTROPHE_PATTERN, text)

    merged: list[str] = []
    i = 0
    while i < len(tokens):
        if (
            i + 2 < len(tokens)
            and tokens[i].isalpha()
            and tokens[i + 1] == "'"
            and tokens[i + 2].isalpha()
        ):
            merged.append(tokens[i] + tokens[i + 1] + tokens[i + 2])
            i += 3
        else:
            merged.append(tokens[i])
            i += 1
    return merged


def tokenize_with_whitespace(text: str) -> list[str]:
    """Coarser tokenizer that the BERT-attack importance-scoring path expects."""
    return re.findall(_WS_PATTERN, text)


def is_word_token(tok: str) -> bool:
    """True iff the token is a pure alphabetic word (no whitespace, no punct)."""
    return tok.strip().isalpha()


def has_foreign_letters(token: str) -> bool:
    """Latin or Russian-exclusive Cyrillic letters disqualify a substitute."""
    return bool(_LATIN_RE.search(token) or _RUS_EXCLUSIVE_RE.search(token))


def filter_not_words(word: str, target_word: str | None = None, *, stopwords: set[str]) -> bool:
    """Return True if ``word`` should be skipped as a substitution candidate."""
    if word.lower() in stopwords:
        return True
    if not any(ch.isalpha() for ch in word):
        return True
    if len(word) < 3:
        return True
    if "</s>" in word or "<s>" in word:
        return True
    if target_word is not None and word == target_word:
        return True
    return False
