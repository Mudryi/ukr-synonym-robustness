"""BERT-Attack-UA implementation as a :class:`BaseAttack`.

Two attack modes are supported, selectable at construction time:

* ``mode="classic"`` — original BERT-Attack: take top-k MLM logits at each
  target position, optionally combine sub-pieces (BPE) for multi-token words.
* ``mode="fill-mask"`` — uses the HuggingFace ``fill-mask`` pipeline; this is
  what the previous ``run_attack`` entrypoint actually called.

Both modes share the importance-scoring + filtering pipeline.
"""

from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset

from ..core.importance import importance_scores_from_probs
from ..core.morphology import compare_normal_forms
from ..core.tokenization import filter_not_words, has_foreign_letters, tokenize_with_whitespace
from .base import AttackResult, BaseAttack, Status


def _tokenize_keep_spacing(seq: str, tokenizer):
    tokens = tokenize_with_whitespace(seq.replace("\n", ""))
    sub_words: list[str] = []
    keys: list[list[int]] = []
    index = 0
    for tok in tokens:
        if tok.isspace():
            keys.append([index, index])
            continue
        pieces = tokenizer.tokenize(tok)
        sub_words.extend(pieces)
        keys.append([index, index + len(pieces)])
        index += len(pieces)
    return tokens, sub_words, keys


def _get_masked_variants(words: list[str], stopwords: set[str]):
    valid_positions = [
        i for i, w in enumerate(words) if not filter_not_words(w, stopwords=stopwords)
    ]
    masked: list[list[str]] = []
    for i in valid_positions:
        tmp = list(words)
        tmp[i] = "[UNK]"
        masked.append(tmp)
    return valid_positions, masked


class BertAttack(BaseAttack):
    def __init__(
        self,
        predictor,
        mlm_model,
        tokenizer_mlm,
        stopwords: set[str],
        ft_sim,
        unmasker=None,
        *,
        mode: str = "fill-mask",
        sbert=None,
        max_length: int = 512,
        batch_size: int = 64,
        # classic-mode hyperparams
        topk: int = 48,
        use_bpe: int = 1,
        threshold_pred_score: float = 0.3,
        # fill-mask-mode hyperparams
        num_subs: int = 128,
        threshold_score: float = 0.04,
        # shared
        cos_sim_threshold: float = 0.4,
        max_changes_frac: float = 0.4,
    ):
        if mode not in {"classic", "fill-mask"}:
            raise ValueError(f"unknown mode: {mode!r}")
        if mode == "fill-mask" and unmasker is None:
            raise ValueError("mode='fill-mask' requires an `unmasker` pipeline")
        self.predictor = predictor
        self.mlm_model = mlm_model
        self.tokenizer_mlm = tokenizer_mlm
        self.stopwords = stopwords
        self.ft_sim = ft_sim
        self.unmasker = unmasker
        self.mode = mode
        self.sbert = sbert
        self.max_length = max_length
        self.batch_size = batch_size
        self.topk = topk
        self.use_bpe = use_bpe
        self.threshold_pred_score = threshold_pred_score
        self.num_subs = num_subs
        self.threshold_score = threshold_score
        self.cos_sim_threshold = cos_sim_threshold
        self.max_changes_frac = max_changes_frac

    @property
    def name(self) -> str:
        return f"bert_attack:{self.mode}"

    # ---------- importance scoring ----------

    def _importance_scores(self, words, orig_prob, orig_label, orig_probs):
        valid_positions, masked_wordlists = _get_masked_variants(words, self.stopwords)
        if not valid_positions:
            return np.zeros(len(words), dtype=float)

        tokenizer = self.predictor.tokenizer
        device = self.predictor.device

        all_input_ids = []
        for wlist in masked_wordlists:
            text = "".join(wlist)
            enc = tokenizer.encode_plus(
                text,
                add_special_tokens=True,
                max_length=self.max_length,
                truncation=True,
                padding="max_length",
            )
            all_input_ids.append(enc["input_ids"])

        seqs = torch.tensor(all_input_ids, dtype=torch.long).to(device)
        ds = TensorDataset(seqs)
        loader = DataLoader(ds, sampler=SequentialSampler(ds), batch_size=self.batch_size)

        leave_1 = []
        for (batch,) in loader:
            leave_1.append(self.predictor.model(batch)[0])
        leave_1 = torch.cat(leave_1, dim=0)
        leave_1 = torch.softmax(leave_1, -1)

        raw = importance_scores_from_probs(orig_prob, orig_label, orig_probs, leave_1)
        scores = np.zeros(len(words), dtype=float)
        for pos, sc in zip(valid_positions, raw):
            scores[pos] = sc
        return scores

    # ---------- substitute generation: classic mode ----------

    @staticmethod
    def _get_substitutes(substitutes, tokenizer, mlm_model, use_bpe, scores, threshold):
        words: list[str] = []
        sub_len, _ = substitutes.size()

        if sub_len == 0:
            return words

        if sub_len == 1:
            for tok_id, j in zip(substitutes[0], scores[0]):
                if threshold != 0 and j < threshold:
                    break
                token = tokenizer._convert_id_to_token(int(tok_id))
                if not token.startswith("▁"):
                    continue
                words.append(token.lstrip("▁"))
            return words

        if sub_len > 4:
            return words

        if use_bpe == 1:
            return BertAttack._get_bpe_substitutes(substitutes, tokenizer, mlm_model)
        return words

    @staticmethod
    def _get_bpe_substitutes(substitutes, tokenizer, mlm_model, k_sub: int = 12):
        substitutes = substitutes[:, :k_sub]
        device = next(mlm_model.parameters()).device

        all_subs: list[list[int]] = []
        for i in range(substitutes.size(0)):
            if not all_subs:
                all_subs = [[int(c)] for c in substitutes[i]]
            else:
                lev_i = []
                for prev in all_subs:
                    for j in substitutes[i]:
                        lev_i.append(prev + [int(j)])
                all_subs = lev_i

        c_loss = nn.CrossEntropyLoss(reduction="none")
        all_t = torch.tensor(all_subs)[:64].to(device)
        n, length = all_t.size()
        preds = mlm_model(all_t)[0]
        ppl = c_loss(preds.view(n * length, -1), all_t.view(-1))
        ppl = torch.exp(torch.mean(ppl.view(n, length), dim=-1))
        _, order = torch.sort(ppl)
        ordered = [all_t[i] for i in order]

        out: list[str] = []
        for word in ordered:
            tokens = [tokenizer._convert_id_to_token(int(t)) for t in word]
            try:
                out.append(tokenizer.convert_tokens_to_string(tokens))
            except Exception:
                out.append("[UNK]")
        return list(dict.fromkeys(out))

    # ---------- shared helpers ----------

    @staticmethod
    def _result_skipped(idx, attack_name, text, true_label, orig_label):
        return AttackResult(
            id=idx, attack=attack_name, orig_text=text, adv_text=text,
            true_label=true_label, orig_label=orig_label, adv_label=orig_label,
            status=Status.SKIPPED_ORIG_WRONG.value, num_changes=0, num_queries=0,
            change_rate=0.0, replacements=[], semantic_sim=None,
        )

    def _candidate_passes_filters(self, candidate: str, target_word: str) -> bool:
        if has_foreign_letters(candidate):
            return False
        if filter_not_words(candidate, target_word, stopwords=self.stopwords):
            return False
        if not self.ft_sim.is_semantic_near(target_word, candidate, self.cos_sim_threshold):
            return False
        if compare_normal_forms(target_word, candidate):
            return False
        return True

    @staticmethod
    def _word_count(words: list[str]) -> int:
        return sum(1 for w in words if any(c.isalpha() for c in w))

    def _finalize(
        self, idx, text, true_label, orig_label, adv_label, status, replacements,
        adv_text, num_queries, words,
    ):
        word_count = self._word_count(words)
        change_rate = (len(replacements) / word_count) if word_count else 0.0
        sem_sim = None
        if self.sbert is not None and replacements:
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
            num_changes=len(replacements),
            num_queries=num_queries,
            change_rate=change_rate,
            replacements=replacements,
            semantic_sim=sem_sim,
        )

    # ---------- public API ----------

    def run(self, text: str, true_label: int, idx: int) -> AttackResult:
        if self.mode == "classic":
            return self._run_classic(text, true_label, idx)
        return self._run_fill_mask(text, true_label, idx)

    # ---------- classic ----------

    def _run_classic(self, text: str, true_label: int, idx: int) -> AttackResult:
        tokenizer_tgt = self.predictor.tokenizer
        device = self.predictor.device
        words = tokenize_with_whitespace(text)

        enc = tokenizer_tgt.encode_plus(text, add_special_tokens=True, max_length=self.max_length)
        input_ids = torch.tensor(enc["input_ids"])
        attn = torch.tensor([1] * len(input_ids))
        orig_probs = self.predictor.model(
            input_ids.unsqueeze(0).to(device),
            attn.unsqueeze(0).to(device),
        )[0].squeeze()
        orig_probs = torch.softmax(orig_probs, -1)
        orig_label = int(torch.argmax(orig_probs))
        current_prob = orig_probs.max()

        if orig_label != true_label:
            return self._result_skipped(idx, self.name, text, true_label, orig_label)

        scores = self._importance_scores(words, current_prob, orig_label, orig_probs)
        num_queries = int(len(words) / 2)

        sorted_positions = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        final_words = copy.deepcopy(words)

        _, sub_words_mlm, keys_mlm = _tokenize_keep_spacing(text, self.tokenizer_mlm)
        sub_words_mlm = ["<s>"] + sub_words_mlm[: self.max_length - 2] + ["</s>"]
        ids_ = torch.tensor([self.tokenizer_mlm.convert_tokens_to_ids(sub_words_mlm)])
        mlm_out = self.mlm_model(ids_.to(device))[0].squeeze()
        word_pred_scores_all, word_predictions = torch.topk(mlm_out, self.topk, -1)
        word_predictions = word_predictions[1 : len(sub_words_mlm) + 1, :]
        word_pred_scores_all = word_pred_scores_all[1 : len(sub_words_mlm) + 1, :]

        replacements: list[dict] = []
        max_changes = int(self.max_changes_frac * (len(words) / 2))

        for pos, _score in sorted_positions:
            if len(replacements) > max_changes:
                return self._finalize(
                    idx, text, true_label, orig_label, orig_label,
                    Status.BUDGET_EXCEEDED.value, replacements,
                    "".join(final_words), num_queries, words,
                )

            tgt_word = words[pos]
            if filter_not_words(tgt_word, stopwords=self.stopwords):
                continue
            if keys_mlm[pos][0] > self.max_length - 2:
                continue

            substitutes = word_predictions[keys_mlm[pos][0] : keys_mlm[pos][1]]
            sub_scores = word_pred_scores_all[keys_mlm[pos][0] : keys_mlm[pos][1]]
            cand = self._get_substitutes(
                substitutes, self.tokenizer_mlm, self.mlm_model,
                self.use_bpe, sub_scores, self.threshold_pred_score,
            )

            most_gap = 0.0
            best = None
            for s in cand:
                if s is None:
                    continue
                if not self._candidate_passes_filters(s, tgt_word):
                    continue

                tmp = list(final_words)
                tmp[pos] = s
                tmp_text = "".join(tmp)

                enc = tokenizer_tgt.encode_plus(
                    tmp_text, add_special_tokens=True, max_length=self.max_length
                )
                ids = torch.tensor(enc["input_ids"]).unsqueeze(0).to(device)
                tmp_probs = torch.softmax(self.predictor.model(ids)[0].squeeze(), -1)
                num_queries += 1
                tmp_label = int(torch.argmax(tmp_probs))

                if tmp_label != orig_label:
                    final_words[pos] = s
                    replacements.append({"position": pos, "orig": tgt_word, "new": s})
                    return self._finalize(
                        idx, text, true_label, orig_label, tmp_label,
                        Status.SUCCESS.value, replacements,
                        tmp_text, num_queries, words,
                    )

                gap = current_prob - tmp_probs[orig_label]
                if gap > most_gap:
                    most_gap = float(gap)
                    best = s

            if most_gap > 0 and best is not None:
                replacements.append({"position": pos, "orig": tgt_word, "new": best})
                current_prob = current_prob - most_gap
                final_words[pos] = best

        adv_text = "".join(final_words)
        return self._finalize(
            idx, text, true_label, orig_label, orig_label,
            Status.FAILED.value, replacements,
            adv_text, num_queries, words,
        )

    # ---------- fill-mask ----------

    def _run_fill_mask(self, text: str, true_label: int, idx: int) -> AttackResult:
        tokenizer_tgt = self.predictor.tokenizer
        device = self.predictor.device

        enc = tokenizer_tgt.encode_plus(text, add_special_tokens=True, max_length=self.max_length)
        input_ids = torch.tensor(enc["input_ids"])
        attn = torch.tensor(enc["attention_mask"])
        orig_probs = self.predictor.model(
            input_ids.unsqueeze(0).to(device), attn.unsqueeze(0).to(device)
        )[0].squeeze()
        orig_probs = torch.softmax(orig_probs, -1)
        orig_label = int(torch.argmax(orig_probs))
        true_prob = orig_probs.max()

        if orig_label != true_label:
            return self._result_skipped(idx, self.name, text, true_label, orig_label)

        mlm_enc = self.tokenizer_mlm.encode_plus(
            text, return_tensors=None, return_attention_mask=False, return_token_type_ids=False
        )
        if len(mlm_enc["input_ids"]) > 511:
            return AttackResult(
                id=idx, attack=self.name, orig_text=text, adv_text=text,
                true_label=true_label, orig_label=orig_label, adv_label=orig_label,
                status=Status.FAILED.value, num_changes=0, num_queries=0,
                change_rate=0.0, replacements=[], semantic_sim=None,
            )

        words = tokenize_with_whitespace(text)
        final_words = copy.deepcopy(words)

        scores = self._importance_scores(words, true_prob, orig_label, orig_probs)
        num_queries = int(len(words) / 2)

        sorted_positions = np.argsort(scores)[::-1]
        replacements: list[dict] = []
        max_changes = int(self.max_changes_frac * (len(words) / 2))

        for i in sorted_positions:
            i = int(i)
            if len(replacements) > max_changes:
                return self._finalize(
                    idx, text, true_label, orig_label, orig_label,
                    Status.BUDGET_EXCEEDED.value, replacements,
                    "".join(final_words), num_queries, words,
                )
            if filter_not_words(words[i], stopwords=self.stopwords):
                continue

            masked = ["<mask>" if k == i else w for k, w in enumerate(words)]
            mask_results = self.unmasker("".join(masked), top_k=self.num_subs)
            filtered = [r for r in mask_results if r["score"] >= self.threshold_score]
            unmasked_words = [j["token_str"] for j in filtered]

            most_gap = 0.0
            best = None
            for new_word in unmasked_words:
                if not self._candidate_passes_filters(new_word, words[i]):
                    continue

                replaced_text = "".join(
                    new_word if k == i else w for k, w in enumerate(final_words)
                )
                enc = tokenizer_tgt.encode_plus(
                    replaced_text, add_special_tokens=True, max_length=self.max_length
                )
                ids = torch.tensor(enc["input_ids"])
                am = torch.tensor(enc["attention_mask"])
                adv_probs = self.predictor.model(
                    ids.unsqueeze(0).to(device), am.unsqueeze(0).to(device)
                )[0].squeeze()
                adv_probs = torch.softmax(adv_probs, -1)
                num_queries += 1
                adv_label = int(torch.argmax(adv_probs))

                if adv_label != true_label:
                    final_words[i] = new_word
                    replacements.append({"position": i, "orig": words[i], "new": new_word})
                    return self._finalize(
                        idx, text, true_label, orig_label, adv_label,
                        Status.SUCCESS.value, replacements,
                        replaced_text, num_queries, words,
                    )

                gap = true_prob - adv_probs[true_label]
                if gap > most_gap:
                    most_gap = float(gap)
                    best = new_word

            if most_gap > 0 and best is not None:
                replacements.append({"position": i, "orig": words[i], "new": best})
                true_prob = true_prob - most_gap
                final_words[i] = best

        adv_text = "".join(final_words)
        return self._finalize(
            idx, text, true_label, orig_label, orig_label,
            Status.FAILED.value, replacements,
            adv_text, num_queries, words,
        )
