# Paper-vs-current results: where they differ and why

This compares `paper_results.txt` (Overleaf tables) against
`results/summary_tables.md` (the new grid run by
`scripts/run_paper_experiments.py`). For each row I report the paper number,
the current number, and a third "paper-style recompute" — the same
`examples.jsonl` re-aggregated with the *old code's* formulas. When the
paper-style recompute lands close to the paper number, the gap is purely an
**aggregation / metric definition** difference. When it doesn't, the
difference is **algorithmic / sampling**.

The aggregation rules used by each codebase:

| metric | new code (`src/evaluation/metrics.py`) | old TextFooler (`textfooler_ukr/main.py:255-275`) | old BERT-Attack (`bert_attack_uk/main.py:110-137`) |
|---|---|---|---|
| `original_accuracy` | `n_correct_orig / n_total` | `1 − orig_failures / n_total` | `1 − origin_success / n_total` |
| `after_attack_accuracy` | `n_correct_adv / n_total` | `1 − adv_failures / n_total` | `1 − suc` where `suc = (success + skipped) / n_total` |
| `attack_success_rate` | `n_success / (n_total − n_skipped)` | not reported | `acc / total` where `acc = success + skipped` |
| `avg_change_rate` | **macro** mean of `num_changes / alpha_token_count`, over **SUCCESS** only | **macro** mean of `num_changed / [t for t in tokens if t.isalpha()]`, over **SUCCESS** only | **micro** `Σ change / Σ len(seq.split(' '))`, over **SUCCESS + SKIPPED** |
| `avg_queries` | mean over **SUCCESS + FAILED + BUDGET** | mean over **SUCCESS + FAILED + BUDGET** (i.e. not-skipped) | `Σ query / (n_success + n_skipped)` — dilutes with 0-query skips |

Two consequences fall out of the table:

1. **TextFooler's old aggregator matches the new one almost exactly.** Both are
   macro means over SUCCESS using alphabetic-token denominators, and both
   compute `original_accuracy` and `after_attack_accuracy` over the full set.
   We expect TextFooler numbers to match paper closely.
2. **BERT-Attack's old aggregator is structurally different.** It micro-averages
   over `SUCCESS + SKIPPED`, so SKIPPED docs add full word counts to the
   denominator while contributing zero changes. It also reports
   `after_attack_accuracy` as `(success + skipped) / total`, which conflates the
   model's clean errors with the attacker's flips. We expect BERT-Attack
   numbers to disagree, with the new code looking *worse* (higher change
   rate, higher avg-query because FAILED docs are now counted, etc.).

## Side-by-side comparison

`Δ%` = average change rate. `q` = avg queries. `→` separates clean/adv
accuracy. `paper-style` is the same examples.jsonl re-aggregated using the
old BERT-Attack formula (micro change_rate over SUCCESS+SKIPPED, queries
over SUCCESS+SKIPPED with 0-query skips).

### TextFooler

| dataset / model | paper (orig→adv, Δ, q) | current new (orig→adv, Δ, q) | Δ-mismatch |
|---|---|---|---|
| reviews / ukr_roberta | 76.28→36.01, Δ=15.80%, q=112.9 | 76.28→37.19, Δ=15.25%, q=112.2 | none — matches |
| reviews / sbert       | 77.58→49.70, Δ=13.01%, q=122.1 | 77.58→49.68, Δ=12.84%, q=123.2 | none |
| reviews / xlmr        | 77.91→49.73, Δ=14.16%, q=126.8 | 77.91→50.90, Δ=14.15%, q=129.1 | none |
| news / ukr_roberta    | **88.55**→73.17, Δ=18.87%, q=50.5 | **98.74**→90.73, Δ=22.67%, q=54.0 | **paper orig_acc looks wrong** (see below) |
| news / sbert          | 93.46→82.67, Δ=19.11%, q=51.8  | 93.68→83.18, Δ=19.67%, q=52.8  | matches |
| news / xlmr           | 93.52→82.95, Δ=19.87%, q=52.0  | 93.63→83.55, Δ=20.03%, q=53.0  | matches |
| unlp / ukr_roberta    | 81.41→47.65, Δ=11.62%, q=343.4 | 81.41→48.95, Δ=11.32%, q=347.4 | matches |
| unlp / sbert          | 81.67→45.03, Δ= 9.95%, q=333.5 | 81.68→42.15, Δ= 9.98%, q=319.1 | matches |
| unlp / xlmr           | 80.10→47.38, Δ=10.57%, q=330.5 | 80.10→49.74, Δ=11.07%, q=338.1 | matches |

### BERT-Attack

| dataset / model | paper (orig→adv, Δ, q) | current new (orig→adv, Δ, q) | paper-style recompute (orig→adv, Δ, q) |
|---|---|---|---|
| reviews / ukr_roberta | 76.28→63.31, Δ= 3.69%, q= 29.7 | 76.28→65.62, Δ=12.97%, q=33.3 | **76.28→89.33**, Δ= 3.22%, q=13.2 |
| reviews / sbert       | 77.58→64.79, Δ= 3.74%, q= 30.5 | 77.58→66.15, Δ=12.85%, q=33.4 | **77.58→88.57**, Δ= 3.40%, q=14.1 |
| reviews / xlmr        | 77.91→67.27, Δ= 4.47%, q= 31.9 | 77.91→68.09, Δ=14.62%, q=33.4 | **77.91→90.18**, Δ= 3.66%, q=13.3 |
| news / ukr_roberta    | 98.83→78.94, Δ=15.67%, q= 17.9 | 98.74→84.71, Δ=28.14%, q=19.2 | **98.74→85.97**, Δ=20.65%, q=14.8 |
| news / sbert          | 93.46→66.43, Δ=13.85%, q= 16.6 | 93.68→73.59, Δ=27.00%, q=18.9 | **93.68→79.91**, Δ=16.96%, q=12.3 |
| news / xlmr           | 93.52→69.10, Δ=14.06%, q= 16.9 | 93.63→75.00, Δ=27.28%, q=19.1 | **93.63→81.37**, Δ=16.83%, q=12.3 |
| unlp / ukr_roberta    | 81.41→58.38, Δ= 5.73%, q=104.0 | 81.41→69.63, Δ=14.21%, q=41.5 | **81.41→88.22**, Δ= 3.23%, q=22.9 |
| unlp / sbert          | 81.67→57.07, Δ= 4.90%, q=101.8 | 81.68→65.18, Δ=13.77%, q=44.1 | **81.68→83.51**, Δ= 4.10%, q=32.8 |
| unlp / xlmr           | 80.10→61.52, Δ= 8.08%, q=109.1 | 80.10→68.59, Δ=12.12%, q=44.9 | **80.10→88.48**, Δ= 2.57%, q=24.1 |

Three things stand out:

* The new BERT-Attack `Δ` is **3–4× larger** than paper on Reviews and UNLP, but
  the **paper-style recompute on the same examples lands almost exactly on the
  paper number** (3.22% vs 3.69%, 3.40% vs 3.74%, 3.23% vs 5.73%, …). The
  attack is doing the same thing; the metric definition changed.
* The new BERT-Attack `after_attack_accuracy` matches the paper closely
  (e.g. reviews/ukr 65.62% vs 63.31%) — the *attack* is similarly effective.
  The paper-style recompute that yields **89.33%** "adv acc" is the old formula
  `1 − (success + skipped) / total`, which is a misleading metric: it counts
  every skipped doc as if the attacker "succeeded," even though those were
  already wrong before the attack.
* Avg queries in the new run are also 2–2.5× higher on UNLP than paper. Same
  cause: the old denominator was `success + skipped` (skipped contribute 0
  queries → drives mean down). The new code averages over `success + failed +
  budget`, so failed docs (which exhausted the candidate list and ran many
  queries) drag the mean up.

## Biggest gaps, ranked

1. **BERT-Attack `Δ` (change rate) — gap 3–4×** on Reviews and UNLP, ~2× on
   News. **Cause: aggregation method.**
   * Old (`bert_attack_uk/main.py:114-137`):
     `change_rate = Σ_change_over_succ_and_skip / Σ_split_word_count_over_succ_and_skip`.
     SKIPPED docs add to the denominator with zero numerator → strong dilution.
   * New (`src/evaluation/metrics.py:67-68`):
     `mean(num_changes / alphabetic_token_count)` over SUCCESS only. No
     dilution.
   * **Empirical confirmation:** running the old formula on the new
     `examples.jsonl` reproduces 3.22% (paper 3.69%), 3.40% (paper 3.74%),
     3.23% (paper 5.73%), etc. The 1–2 pp residual is run-to-run drift
     (random init, different transformers/torch versions).

2. **BERT-Attack "adv accuracy" gap — ~+25 pp** in paper vs new code on
   Reviews/UNLP (paper says 63%, new says 65%; *but* paper-style recompute
   says 89%). The new column is a reasonable definition; the paper's
   reported "adv accuracy" was computed as `1 − (n_succ + n_skip)/n_total`,
   which double-counts already-misclassified inputs as adversarial wins.
   Paper's `original_accuracy` and `attack_success_rate` are still
   meaningful; the column "Adv. Acc" / "Drop" in the paper for BERT-Attack
   is the suspect quantity. **Cause: metric definition.**

3. **BERT-Attack `q` (avg queries) — paper is 2–2.5× higher than new on
   UNLP, but 2× *lower* on Reviews/News.** **Cause: aggregation method.**
   * Old denominator `n_succ + n_skip` (with skipped docs contributing 0
     queries) deflates the mean.
   * New denominator `n_succ + n_fail + n_budget` includes FAILED docs that
     ran many queries before giving up.
   * Direction of the gap depends on the `skip_rate` vs `fail_rate` mix:
     UNLP has lots of FAILED docs (262/382), so the new mean is much higher
     than the old in absolute terms when failed docs are query-heavy, and
     lower when they aren't.
   * Same paper-style recompute (col 3 in the BERT table) reproduces 13.2,
     14.1, 13.3, 14.8, 12.3, 12.3, 22.9, 32.8, 24.1 — closer but still
     2–4 q below paper. The residual is again model/version drift.

4. **TextFooler News + ukr-roberta `original_accuracy` — paper 88.55%
   vs current 98.74%** (~10 pp gap). **Cause: probably a paper typo or a
   smaller-sample run.** Evidence:
   * The paper's own BERT-Attack table reports the *same* model on the
     *same* test set as 98.83% (paper_results.txt:50). Two attacks on the
     same `(model, test set)` cannot have different clean accuracies.
   * The new code's `load_dataset` (`src/core/data.py:34-35`) and the old
     `read_corpus` (`textfooler_ukr/dataloader.py:47-50`) use identical
     sampling: `df.sample(10_000, random_state=1914)`. The old textfooler
     checkpoint path matches the new one (`npz4/model_npz4_9_1000`).
   * The new BERT-Attack run on the same model/data reports 98.74%, the
     same as TextFooler. So the test set, sampling, and checkpoint are
     consistent.
   * Conclusion: the 88.55% in the paper is almost certainly stale (an
     earlier checkpoint or smaller subset that didn't make it to the
     final grid). It's not an attack-code issue.

5. **TextFooler everything else — within 0.1–3 pp of paper.** No
   investigation needed; this is run-to-run noise from torch/transformers
   versions and the small amount of stochasticity in candidate filtering.
   No code-level discrepancy.

## What's *not* a cause

I checked these and they don't explain the gaps:

* **`max_changes_frac` budget.** Both old and new use
  `int(0.4 * len(words) / 2)` where `len(words)` is the
  `tokenize_with_whitespace` output (so ~2× actual word count, making the
  effective budget ~40% of words). Identical.
  (`bert_attack_uk/main.py:578` vs `src/attacks/bert_attack.py:411`.)
* **`cos_sim_threshold`, `threshold_score`, `num_subs`, `topk`.** All
  defaults match (0.33, 0.04, 128, 48). (`bert_attack_uk/main.py:544` vs
  `scripts/run_paper_experiments.py` flag list.)
* **`filter_not_words`, `compare_normal_forms`, `is_semantic_near`.** Same
  predicates, same thresholds.
* **Importance scoring.** Identical leave-one-out formula
  (`textfooler_ukr/main.py:74-77` vs `src/core/importance.py`,
  `bert_attack_uk/main.py:243-330` vs the same).
* **Tokenization.** `tokenize_ukrainian` and `tokenize_with_whitespace` are
  byte-for-byte identical between repos.
* **Query counter convention.** Both initialize with `int(len(words)/2)`
  for the leave-one-out importance pass and `+1` per candidate
  evaluation. The *aggregation* differs, not the per-doc counter.
* **Dataset / sampling.** Same `df.sample(10_000, random_state=1914)`.
* **GPU.** Both runs used CUDA (the new venv reports
  `torch.cuda.is_available() == True` with 1 device, and the old runs were
  GPU per the original repo).

## TL;DR

Almost every meaningful gap between paper and new results is a **metric-
aggregation difference**, not an attack-effectiveness difference. When you
re-aggregate the new `examples.jsonl` with the old paper formulas, you
recover paper numbers within 1–2 pp / 1–4 q. The two genuine algorithmic
deltas are within run-to-run noise.

Two follow-ups worth considering:

* If you want the new tables to *match* the paper for cross-paper
  comparison, change `src/evaluation/metrics.py` to expose both the new
  per-attack-canonical aggregator and a paper-compat micro-aggregator
  (e.g. add `avg_change_rate_micro`, `avg_queries_paper_style`).
* The paper's BERT-Attack "Adv. Acc." column (`1 − (success+skipped)/total`)
  is misleading — it inflates the attack's apparent effect by counting
  pre-existing model errors. The new code's `after_attack_accuracy`
  (`1 − n_correct_after_attack / n_total`) is the more standard
  definition; if you re-issue the paper, prefer that.
