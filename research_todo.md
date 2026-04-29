# Technical TODO — finishing the Ukrainian robustness paper

Companion to `research_plan.md`. The plan describes the *story*; this file
lists the **code, data, and runs** needed to finish it. Each item names the
exact files/modules to add or change so work can be parallelized.

## Repo state at time of writing

| Area | Status |
|---|---|
| Attack code | TextFooler + BERT-Attack (fill-mask + classic) wired through `src/cli/run_attack.py`. Stable. |
| Datasets | 3 of 4 from the plan: `reviews`, `news`, `unlp`. **Missing: government-request sentiment.** |
| Models | 3 fine-tuned classifiers per dataset (`ukr_roberta`, `sbert_mpnet`, `xlmr_base`). Checkpoints under `…/trained_models/`. |
| Main grid | 3 × 3 × 2 = 18 cells in `results/textfooler/` and `results/bert_attack/`. Numbers in `results/summary_tables.md`. Reviews TF-ASR 34–51%, BA-ASR 13–15%; News TF-ASR 8–11%, BA-ASR 14–21%; UNLP TF-ASR 38–48%, BA-ASR 14–20%. |
| Metric audit | `results/paper_vs_current_diff.md` reconciles new vs paper aggregations — gaps are metric-definition, not algorithmic. |
| Replacement analysis | `results/replacement_analysis.md` ranks (orig→new) lemma pairs across the grid. |
| LLM transfer | **Not started.** No code, no prompts, no eval. |
| WSD analysis / WSD-filtered attack | **Not started.** |
| Validity annotation | **Not started.** No subset, no schema, no judge. |
| Defenses | **Not started.** |
| Ablations | **Not started** (single attack budget / threshold per dataset). |
| Statistical testing | **Not started.** |

## Priority ordering

P0 = blocks the paper. P1 = makes the paper credible. P2 = nice-to-have ablations.

---

# P0 — close the core benchmark

## T1. Add the 4th dataset: government-request sentiment

**Why:** plan calls for 4 datasets; repo has 3. Without it, RQ2 (task
sensitivity) only has 3 data points.

- [ ] Locate / acquire the gov-request CSV (decide schema: `text` + `label`).
- [ ] Add `configs/govrequests.yaml` mirroring `configs/unlp.yaml` (set `nclasses`, `dataset-path`, attack defaults).
- [ ] Extend `src/core/data.py:load_dataset` — currently dispatches on the `dataset` arg with hard-coded branches for `reviews`/`news`/`unlp`. Add a `govrequests` branch.
- [ ] Extend `src/cli/run_attack.py:parse_args` — `--dataset` choices list (line 132).
- [ ] Extend `scripts/run_paper_experiments.py`:
  - Add `"govrequests"` to `DATASETS` (line 80).
  - Add 3 entries to `MODELS["govrequests"]` with checkpoint paths.
- [ ] Train 3 classifiers (`ukr_roberta`, `sbert_mpnet`, `xlmr_base`) on this dataset. Reuse the fine-tuning script in `…/xml-roberta-finetune-reviews/` if available; otherwise write `scripts/finetune_classifier.py`.
- [ ] Run the 6 new attack cells (3 models × 2 attacks). Re-run `--tables-only` to refresh `results/summary_tables.md`.

## T2. Resumability + batching for long news runs

**Why:** news cells take 6+ hours; without `--resume` a single crash wastes
the run. Already flagged in `NEXT_STEPS.md` items 2 and 3.

- [ ] `--resume` in `src/cli/run_attack.py`: read existing `examples.jsonl`, skip seen `id`s, re-aggregate including existing rows in `ResultWriter.finalize()`.
- [ ] `Predictor.predict_batch(texts, batch_size)` in `src/core/predictor.py`. Wire into `src/attacks/bert_attack.py` (`_run_classic` / `_run_fill_mask`) and confirm `src/attacks/textfooler.py` uses it. Expected 5–10× speedup on GPU.
- [ ] Single-determinism CI check: same `--seed` twice → byte-identical `summary.json`. Currently no test exists.

## T3. Expose paper-compat metrics alongside canonical metrics

**Why:** `results/paper_vs_current_diff.md` shows the new aggregator
disagrees with the paper on BERT-Attack `Δ` and `q` purely because of
denominator choice (SUCCESS-only macro vs SUCCESS+SKIPPED micro). For
cross-paper comparison, expose both.

- [ ] In `src/evaluation/metrics.py`, add fields to the summary dict:
  - `avg_change_rate_micro` — micro over SUCCESS+SKIPPED (paper formula).
  - `avg_queries_paper_style` — Σq / (n_succ + n_skip).
  - `attack_success_rate_paper_style` — `(success + skipped) / total`.
- [ ] Add a column to `runs.csv` and a switch in `scripts/run_paper_experiments.py:build_table` to render either version.

---

# P1 — Validity layer + WSD (the paper's main contribution per the plan)

## T4. Validity annotation harness

**Why:** without validity, reviewers will say "the attack changed meaning,
not robustness." Plan §4 calls this out as the primary critique to defuse.

- [ ] **Subset builder** `scripts/build_validity_subset.py`:
  - For each (dataset, model, attack), sample 50–100 SUCCESS + 20–50 FAILED examples (target ~400–800 pairs total).
  - Stratify by dataset and attack; balance gold labels.
  - Emit `data/validity/<attack>__<dataset>__<model>.jsonl` with `{id, orig_text, adv_text, true_label, orig_label, adv_label, replacements}`.
- [ ] **Annotation schema** (per pair):
  - `meaning_preserved`: yes / partial / no
  - `label_preserved`: yes / unclear / no
  - `fluency_ok`: yes / no
  - `substitution_sense_correct`: yes / no
  - `attack_valid`: yes / no
  - `notes` (free text)
- [ ] **LLM-judge implementation** `src/evaluation/llm_judge.py`:
  - Prompt template (Ukrainian) returning the 5 fields as JSON.
  - Use Claude / GPT-4o (one model is fine for first pass).
  - Pin model version, temperature=0, schema-constrained output.
- [ ] **Human spot-check** — 100 pair sample by the author; report inter-rater agreement vs LLM judge (κ).
- [ ] **New metric** in `src/evaluation/metrics.py`: `valid_attack_success_rate = n_success_valid / (n_total - n_skipped)`. Add to summary and runs.csv when annotations are available.
- [ ] Output: `results/validity_annotations.jsonl` + `results/validity_summary.md` table:

  | Attack | Raw ASR | % meaning preserved | % label preserved | Valid-ASR |

## T5. WSD-aware substitution analysis

**Why:** Plan §5.1. Existing `results/replacement_analysis.md` shows raw
lemma-pair counts but no sense reasoning — e.g. `чудовий → ненаглядний`
(357 events) is technically a synonym in one sense only.

- [ ] **WSD module** `src/core/wsd.py`:
  - Resolve each substitution against the original sentence context.
  - Two implementations (pick the cheaper one that performs adequately):
    - **Embedding-based:** sense-disambiguation via contextual nearest neighbor in a Ukrainian sense-tagged resource (e.g. WordNet-UA or Lapa-LLM contextual embeddings).
    - **LLM-based:** prompt an LLM with the sentence + candidate replacement, ask "does the replacement preserve the original word's sense in this context?"
- [ ] **Substitution category classifier** `scripts/classify_substitutions.py`:
  - Per substitution, assign one of: `correct_sense`, `wrong_sense`, `register_shift`, `domain_shift`, `morph_error`, `named_entity_or_number`, `label_changing`.
  - Implementation: rule-based (POS/morph from pymorphy, regex for entities/numbers) → LLM fallback for the semantic categories.
  - Read from `results/<attack>/<dataset>__<model>/examples.jsonl`, write `results/wsd/<attack>__<dataset>__<model>.jsonl`.
- [ ] **Aggregate report** `scripts/wsd_report.py` → `results/wsd_summary.md`:
  - Stacked-bar source data: error category share per (attack, dataset, model).
  - Compare TextFooler vs BERT-Attack, per-dataset, success vs failed.
  - Expected finding: large fraction of TF successes use wrong-sense synonyms (paper hypothesis).

## T6. WSD-filtered attack variant

**Why:** Plan §5.2 — strongest methodological addition.

- [ ] New attack `src/attacks/wsd_textfooler.py` that subclasses `TextFoolerAttack`:
  - Override `_get_synonym_candidates` (or equivalent hook) to call `src/core/wsd.py` per candidate and drop wrong-sense ones.
  - Configurable strictness: `--wsd-threshold`.
- [ ] Mirror for `wsd_bert_attack.py` if time allows (BERT-Attack already conditions on context, so the gain is smaller — start with TF only).
- [ ] Wire into `src/cli/run_attack.py` `--attack` choices.
- [ ] Add `wsd_textfooler` row to the paper grid via `scripts/run_paper_experiments.py:ATTACKS`.
- [ ] Run on 4 datasets × 3 models (or just 2 datasets if budget tight, but at minimum reviews + unlp, the high-ASR ones).
- [ ] Comparison table `results/wsd_attack_results.md` showing raw ASR vs Valid-ASR for `tf` vs `wsd_tf`.

---

# P1 — LLM transfer experiments

## T7. LLM eval scaffolding

**Why:** Plan §6 / RQ3. Repo has zero LLM code today.

- [ ] New package `src/llm/`:
  - `client.py` — unified `LLMClient` interface with backends:
    - `huggingface` (local) — Mamay, Lapa, Qwen, Gemma 2/4 via `transformers` or `vllm`.
    - `anthropic` / `openai` for closed-model reference rows.
  - `prompts.py` — Ukrainian zero-shot + few-shot templates per dataset (the plan has a template at line 246 of `research_plan.md`; lift it into code).
  - `parse.py` — strict label parser (regex + label-vocabulary match; fail loudly on unparseable output).
- [ ] `scripts/run_llm_eval.py`:
  - Inputs: `--subset {clean,adv,hard_adv}`, `--llm <name>`, `--mode {zero,few}`, `--dataset`.
  - Output: `results/llm/<llm>__<dataset>__<subset>__<mode>.jsonl` + summary.
  - Pin per-run config (model name, version, parameter size, quantization, backend, temperature, prompt hash, date) into the summary, per the plan (lines 220–229).

## T8. Build the 3 LLM eval subsets

**Why:** Plan §6 — Subset A (clean control), B (any-classifier-broken), C (hard-adversarial: broke all 3).

- [ ] `scripts/build_llm_subsets.py`:
  - Read `results/<attack>/<dataset>__<model>/examples.jsonl` for all (model, attack) per dataset.
  - **Subset A:** random clean samples (size: matched to B for fair comparison, e.g. 500/dataset).
  - **Subset B:** examples where `status == SUCCESS` for ≥1 (model, attack) cell.
  - **Subset C:** examples where `status == SUCCESS` for **all 3** classifiers under at least one attack.
  - Store both original and adversarial text per example. Track which model produced the adversarial version (for B/C, pick one canonical adv; document the choice).
  - Output: `data/llm_subsets/<dataset>__{A,B,C}.jsonl`.

## T9. Run LLM transfer grid

- [ ] LLMs: MamayLM-9B, Lapa-LLM (Gemma-3-12B base), Qwen2.5-14B-Instruct, Gemma-3-12B (or Gemma-4 if released by run time). Optional: Claude-Sonnet-4.6 / GPT-4o as a strong reference.
- [ ] For each LLM × dataset × subset (A/B/C) × mode (zero-shot, few-shot) — run.
- [ ] Metrics in `src/evaluation/metrics.py` — extend with:
  - `llm_clean_acc`, `llm_adv_acc`
  - `llm_transfer_asr` = fraction of B/C samples where LLM's adv prediction ≠ true label
  - `llm_consistency` = fraction where LLM gives same label on orig and adv
  - `robustness_gap` = clean − adv
- [ ] Output `results/llm_transfer.md` with the table from research_plan.md line 279:

  | Dataset | Attack | Classifier ASR | Mamay ASR | Lapa ASR | Qwen ASR | Gemma ASR |

---

# P1 — Defenses

## T10. Defense 1 — adversarial training

**Why:** Plan §7.1. Most publication-friendly defense.

- [ ] Subset script `scripts/build_adv_train_set.py` (already proposed in `NEXT_STEPS.md` item 5):
  - Walk `results/*/*/examples.jsonl`, keep `status == SUCCESS` and `semantic_sim ≥ 0.8`.
  - Optionally restrict to WSD-validated examples (after T5).
  - Emit per-dataset CSV in the same schema as the source.
- [ ] Fine-tuning script `scripts/finetune_with_adv.py`:
  - Mix clean train + α·adversarial (sweep α ∈ {0, 0.25, 0.5, 1.0}).
  - Save under `…/trained_models/<run_id>_adv/`.
- [ ] Re-attack the adv-trained checkpoints **using a held-out subset** (do not test on the same adversarial examples used in training — plan §7.1 explicitly warns about this).
- [ ] Output: `results/defense_advtrain.md` with the table from plan line 304:

  | Training setup | Clean F1 | TF-robust F1 | BA-robust F1 | Valid-ASR |

## T11. Defense 2 — consistency voting (optional, drop if T10 done well)

**Why:** Plan §7.2. Works without retraining.

- [ ] `src/defenses/consistency_voting.py`:
  - Generate 3–5 paraphrases per input via Lapa or another paraphraser.
  - Classify each variant; majority vote / probability average.
  - Disagreement above threshold → label "uncertain".
- [ ] Eval on the same adversarial subset used elsewhere; output `results/defense_voting.md`.

---

# P2 — Ablations

Pick 3–4 from this list per Plan §8.

## T12. Attack budget sweep

- [ ] Run TF + BA at `max-changes-frac` ∈ {0.05, 0.10, 0.20, 0.40} on **one model per dataset** (probably the median-ASR model to control compute). 4 budgets × 4 datasets × 2 attacks = 32 runs.
- [ ] `scripts/budget_ablation.py` — wrapper around `scripts/run_paper_experiments.py` that injects `--max-changes-frac`. Output: `results/ablation_budget.md` (line plot data: ASR vs budget).

## T13. Semantic-similarity threshold sweep

- [ ] TF: `--sim-threshold` ∈ {0.5, 0.7, 0.85, 0.9}. BA: `--cos-sim-threshold` ∈ {0.20, 0.33, 0.50, 0.70}.
- [ ] Show ASR ↓ and Valid-ASR ↑ as threshold tightens (plan §8 ablation B).

## T14. POS-based vulnerability

- [ ] `scripts/pos_analysis.py`:
  - Read all `examples.jsonl`; tag each `(orig, new)` with pymorphy2 POS.
  - Aggregate ASR contribution per POS (NOUN / VERB / ADJF / ADVB).
- [ ] Output: `results/ablation_pos.md`.

## T15. Word-frequency analysis

- [ ] Add Ukrainian word-frequency table (e.g. from a CC corpus or fastText vocab counts) under `resources/`.
- [ ] `scripts/freq_analysis.py` — compute mean log-freq of (orig, new) per success vs failure, per dataset. Test the hypothesis "rare-word substitutions cause stronger flips" (plan §8 ablation D).

## T16. Confidence drop on near-failures

- [ ] Already have orig/adv probability in attack code, but it isn't logged. Patch `src/attacks/{textfooler,bert_attack}.py` to record `orig_conf` and `adv_conf` in each `AttackResult`.
- [ ] Aggregate per dataset/model — show that even when the label doesn't flip, confidence drops markedly.

---

# P1 — Statistical rigor

## T17. Bootstrap CIs and significance tests

**Why:** Plan §9.

- [ ] `src/evaluation/stats.py`:
  - `bootstrap_ci(values, statistic, n=10_000, alpha=0.05)` — used for ASR, robust accuracy, LLM transfer.
  - `mcnemar_paired(orig_correct, adv_correct)` — for clean-vs-adversarial paired predictions.
  - `paired_bootstrap(model_a_outcomes, model_b_outcomes)` — for cross-model comparisons (e.g. classifier vs LLM ASR).
- [ ] Patch `src/evaluation/writer.py:finalize` to attach 95% CIs to summary metrics.
- [ ] Patch `scripts/run_paper_experiments.py:build_table` to render `metric (lo, hi)` cells.

---

# P2 — Reproducibility & infra

These exist in `NEXT_STEPS.md` already; the relevant ones for the paper:

- [ ] HuggingFace dataset hosting (NEXT_STEPS §4) — only after datasets are public.
- [ ] Cross-run leaderboard (NEXT_STEPS §6) — useful for the paper's table generation.
- [ ] Unit tests on morphology / pos_filter / synonym_dict (NEXT_STEPS §7) — paper-defensive, low-priority.

---

# Final paper deliverables checklist (per `research_plan.md` §10–§11)

After P0 + P1 are done:

- [ ] **Table 1** — Datasets (task, classes, sizes, domain).
- [ ] **Table 2** — Clean F1 per (model × dataset) — already extractable from `summary.json`s.
- [ ] **Table 3** — Attack results with CIs (replaces `results/summary_tables.md`).
- [ ] **Table 4** — Valid attack analysis (raw ASR / meaning-preserved / label-preserved / Valid-ASR). From T4.
- [ ] **Table 5** — LLM transfer per (LLM × dataset × attack). From T9.
- [ ] **Table 6** — Defense results (clean F1, adv F1, robust acc, Valid-ASR reduction). From T10.
- [ ] **Figure 1** — Pipeline diagram.
- [ ] **Figure 2** — Robustness drop bar chart per dataset.
- [ ] **Figure 3** — Stacked WSD error categories. From T5.

---

# Suggested execution order (one-shot plan, 4–6 weeks of compute)

1. **Week 1:** T1 (4th dataset + classifiers) in parallel with T2 (resume + batching). Baseline T17 stats wiring.
2. **Week 2:** Re-run main grid at 4×3×2 = 24 cells with the new metrics (T3) and CIs (T17).
3. **Week 2–3:** T4 validity annotation (run LLM judge over ~600 pairs) + T5 WSD analysis. These can run during week-2 attacks.
4. **Week 3:** T8 build LLM subsets, T7 LLM scaffolding, T9 LLM transfer grid (this is the slowest cell — start as soon as subsets exist).
5. **Week 4:** T6 WSD-filtered attack runs.
6. **Week 4–5:** T10 adversarial-training defense (fine-tunes are slow, do in background).
7. **Week 5:** Pick 3 ablations from T12–T16. Skip the rest.
8. **Week 6:** Tables, figures, paper writing.

If compute is tight, drop in this order: T16 → T15 → T13 → T11 → T6.
