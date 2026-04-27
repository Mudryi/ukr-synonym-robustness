"""TextFooler-UA attack as a :class:`BaseAttack` implementation.

The algorithm is unchanged from the original ``src/attacks/textfooler/main.py``:

1. tokenize and find perturbable positions (alphabetic, non-stopword);
2. score per-position importance via leave-one-out predictions;
3. for each high-importance position, look up synonyms in the dictionary,
   inflect them to match the target word's grammar, and probe the classifier;
4. accept the substitute that maximises (semantic_sim) when it flips the label
   (or that minimises orig-label probability if no candidate flips).

What changed:
* hard-coded paths, model construction, dataset I/O and result dumping all
  moved out of this module — the attack is now usable on a single ``(text, label)``
  pair via :meth:`run`.
* output is a single :class:`AttackResult` per call, not a tuple.
"""

from __future__ import annotations

import numpy as np
import torch

from ..core.importance import importance_scores_from_probs
from ..core.morphology import get_normal_form, replace_word
from ..core.pos_filter import get_pos, pos_filter
from ..core.tokenization import is_word_token, tokenize_ukrainian
from .base import AttackResult, BaseAttack, Status


class TextFoolerAttack(BaseAttack):
    def __init__(
        self,
        predictor,
        sbert,
        synonym_dict: dict[str, list[str]],
        stopwords: set[str],
        *,
        sim_threshold: float = 0.7,
        synonym_num: int = 50,
        import_score_threshold: float = -1.0,
        compute_final_sim: bool = True,
    ):
        self.predictor = predictor
        self.sbert = sbert
        self.synonym_dict = synonym_dict
        self.stopwords = stopwords
        self.sim_threshold = sim_threshold
        self.synonym_num = synonym_num
        self.import_score_threshold = import_score_threshold
        self.compute_final_sim = compute_final_sim

    @property
    def name(self) -> str:
        return "textfooler"

    # ---------- helpers ----------

    def _get_synonyms(self, word: str) -> list[str]:
        if word in self.synonym_dict:
            syns = self.synonym_dict[word]
        else:
            normal = get_normal_form(word)
            if normal in self.synonym_dict:
                syns = self.synonym_dict[normal]
            else:
                return []
        normal = get_normal_form(word)
        return [
            s for s in set(syns)
            if s.lower() != word.lower() and s.lower() != normal.lower()
        ]

    @staticmethod
    def _word_token_count(tokens: list[str]) -> int:
        return sum(1 for t in tokens if t.isalpha())

    @staticmethod
    def _result_skipped(idx, text, true_label, orig_label) -> AttackResult:
        return AttackResult(
            id=idx,
            attack="textfooler",
            orig_text=text,
            adv_text=text,
            true_label=true_label,
            orig_label=orig_label,
            adv_label=orig_label,
            status=Status.SKIPPED_ORIG_WRONG.value,
            num_changes=0,
            num_queries=0,
            change_rate=0.0,
            replacements=[],
            semantic_sim=None,
        )

    # ---------- public API ----------

    def run(self, text: str, true_label: int, idx: int) -> AttackResult:
        text_ls = tokenize_ukrainian(text)
        word_token_count = self._word_token_count(text_ls)

        orig_probs = self.predictor([text_ls]).squeeze()
        orig_label = int(torch.argmax(orig_probs))
        orig_prob = orig_probs.max()
        num_queries = 1

        if true_label != orig_label:
            return self._result_skipped(idx, text, true_label, orig_label)

        perturbable = [
            i for i, tok in enumerate(text_ls)
            if is_word_token(tok) and tok.lower() not in self.stopwords
        ]
        if not perturbable:
            return AttackResult(
                id=idx, attack=self.name, orig_text=text, adv_text=text,
                true_label=true_label, orig_label=orig_label, adv_label=orig_label,
                status=Status.FAILED.value, num_changes=0, num_queries=num_queries,
                change_rate=0.0, replacements=[], semantic_sim=None,
            )

        leave_1_texts = [text_ls[:i] + ["<oov>"] + text_ls[i + 1 :] for i in perturbable]
        leave_1_probs = self.predictor(leave_1_texts)
        num_queries += len(leave_1_texts)

        scores = importance_scores_from_probs(
            orig_prob, orig_label, orig_probs, leave_1_probs
        )

        words_perturb = [
            (pos, text_ls[pos]) for pos, score in
            sorted(zip(perturbable, scores), key=lambda x: x[1], reverse=True)
            if score > self.import_score_threshold
        ]

        synonyms_all: list[tuple[int, list[str]]] = []
        for position, word in words_perturb:
            syns = self._get_synonyms(word)
            syns = [s for s in syns if len(s.split(" ")) == 1][: self.synonym_num]
            if syns:
                synonyms_all.append((position, syns))

        pos_ls = get_pos(text_ls)
        text_prime = text_ls.copy()
        text_cache = text_prime.copy()
        replacements: list[dict] = []
        num_changed = 0
        success = False

        for pos, syns in synonyms_all:
            new_texts = [
                replace_word("".join(text_prime), text_prime[pos], syn) for syn in syns
            ]
            kept_syns = [s for s, t in zip(syns, new_texts) if t is not None]
            kept_texts = [t for t in new_texts if t is not None]
            if not kept_texts:
                continue

            new_probs = self.predictor(kept_texts)
            num_queries += len(kept_texts)

            sims = self.sbert.semantic_sim(
                ["".join(text_cache)] * len(kept_texts),
                ["".join(t) for t in kept_texts],
            )[0]

            if new_probs.dim() < 2:
                new_probs = new_probs.unsqueeze(0)

            mask = (orig_label != torch.argmax(new_probs, dim=-1)).cpu().numpy()
            mask = mask & (sims >= self.sim_threshold)

            syn_pos_ls = []
            for new_text in kept_texts:
                if len(new_text) > 10:
                    syn_pos_ls.append(
                        get_pos(new_text[max(pos - 4, 0) : pos + 5])[min(4, pos)]
                    )
                else:
                    syn_pos_ls.append(get_pos(new_text)[pos])

            pos_mask = np.array(pos_filter(pos_ls[pos], syn_pos_ls))
            mask = mask & pos_mask

            if np.sum(mask) > 0:
                idx_best = int((mask * sims).argmax())
                replaced = kept_syns[idx_best]
                replacements.append({"position": pos, "orig": text_prime[pos], "new": replaced})
                text_prime = kept_texts[idx_best]
                num_changed += 1
                success = True
                break
            else:
                penalty = torch.from_numpy(
                    (sims < self.sim_threshold) + (1 - pos_mask).astype(float)
                ).float().to(new_probs.device)
                new_label_probs = new_probs[:, orig_label] + penalty
                min_val, min_idx = torch.min(new_label_probs, dim=-1)
                if min_val < orig_prob:
                    replaced = kept_syns[int(min_idx)]
                    replacements.append(
                        {"position": pos, "orig": text_prime[pos], "new": replaced}
                    )
                    text_prime = kept_texts[int(min_idx)]
                    num_changed += 1

        adv_text = "".join(text_prime)
        adv_label = int(torch.argmax(self.predictor([text_prime])))
        num_queries += 1

        if true_label != adv_label:
            status = Status.SUCCESS.value
        else:
            status = Status.FAILED.value

        change_rate = num_changed / word_token_count if word_token_count else 0.0

        sem_sim = None
        if self.compute_final_sim and (success or num_changed > 0):
            sem_sim = float(self.sbert.semantic_sim([text], [adv_text])[0][0])

        return AttackResult(
            id=idx,
            attack=self.name,
            orig_text=text,
            adv_text=adv_text,
            true_label=true_label,
            orig_label=orig_label,
            adv_label=adv_label,
            status=status,
            num_changes=num_changed,
            num_queries=num_queries,
            change_rate=change_rate,
            replacements=replacements,
            semantic_sim=sem_sim,
        )
