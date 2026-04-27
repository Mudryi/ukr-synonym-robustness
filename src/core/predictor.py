"""Unified target-classifier wrapper.

Both attacks need to query the target classifier with batches of strings and
get class probabilities back. Each attack used to define its own ``predictor``
closure with subtly different signatures (one accepted token lists, the other
strings). The unified :class:`Predictor` accepts both: lists of tokens are
joined into a string before tokenization.
"""

from __future__ import annotations

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class Predictor:
    def __init__(
        self,
        target_model: str,
        target_checkpoint: str,
        nclasses: int,
        device: str | torch.device | None = None,
        max_length: int = 512,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(target_model)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            target_checkpoint, num_labels=nclasses
        ).to(self.device)
        self.model.eval()
        self.max_length = max_length

    @staticmethod
    def _stringify(texts):
        """Accept list[str], list[list[str]], or str. Always return list[str]."""
        if isinstance(texts, str):
            return [texts]
        if len(texts) == 0:
            return []
        if isinstance(texts[0], list):
            return ["".join(toks) for toks in texts]
        return list(texts)

    @torch.no_grad()
    def __call__(self, texts) -> torch.Tensor:
        texts = self._stringify(texts)
        inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=self.max_length,
        ).to(self.device)
        logits = self.model(**inputs).logits
        return torch.softmax(logits, dim=-1)
