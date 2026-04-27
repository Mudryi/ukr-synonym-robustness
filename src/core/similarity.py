"""Semantic-similarity scorers used to filter substitution candidates.

* :class:`SBERT` returns sentence-pair similarity (used by TextFooler).
* :class:`FastTextSim` returns word-pair cosine similarity (used by BERT-Attack
  to keep the substitute semantically close to the target word).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np


class SBERT:
    """Wrapper around sentence-transformers cosine similarity."""

    def __init__(self, model_name: str = "sentence-transformers/paraphrase-xlm-r-multilingual-v1"):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)

    def semantic_sim(self, sents1, sents2):
        e1 = self.model.encode(sents1)
        e2 = self.model.encode(sents2)
        e1 = e1 / np.linalg.norm(e1, axis=1, keepdims=True)
        e2 = e2 / np.linalg.norm(e2, axis=1, keepdims=True)
        cos = np.clip(np.sum(e1 * e2, axis=1), -1.0, 1.0)
        sim = 1.0 - np.arccos(cos)
        return [sim]


class FastTextSim:
    """Subword-based word similarity using fastText (Ukrainian CBOW model)."""

    def __init__(self, ft_path: str | Path | None = None, env_var: str = "FASTTEXT_UK_PATH"):
        import fasttext

        path = self._resolve_path(ft_path, env_var)
        self.model = fasttext.load_model(str(path))

    @staticmethod
    def _resolve_path(ft_path, env_var):
        if ft_path:
            p = Path(ft_path)
        elif os.getenv(env_var):
            p = Path(os.environ[env_var])
        else:
            # repo_root / resources / fasttext_uk / cbow.uk.300.bin
            repo_root = Path(__file__).resolve().parents[2]
            p = repo_root / "resources" / "fasttext_uk" / "cbow.uk.300.bin"
        if not p.exists():
            raise FileNotFoundError(
                f"fastText model not found at {p}. "
                f"Run scripts/download_fasttext_uk.py or set {env_var}."
            )
        return p

    def _vec(self, word):
        try:
            return self.model[word]
        except KeyError:
            return None

    def is_semantic_near(self, u: str, v: str, threshold: float = 0.35) -> bool:
        """Return True if the words are similar enough; True on missing vectors (fallback)."""
        vu, vv = self._vec(u), self._vec(v)
        if vu is None or vv is None:
            return True
        nu, nv = np.linalg.norm(vu), np.linalg.norm(vv)
        if nu == 0 or nv == 0:
            return True
        cos = float(np.dot(vu, vv) / (nu * nv))
        return cos >= threshold
