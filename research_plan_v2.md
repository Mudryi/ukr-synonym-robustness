# Research plan v2 — Ukrainian synonym-robustness paper

Replaces the framing in `research_plan.md`. The story is no longer
"benchmark first, transfer second." It is:

> Modern Ukrainian LLMs are not robust to off-the-shelf synonym attacks,
> those attacks are themselves *broken* (wrong-sense substitutions inflate
> ASR), a WSD-aware variant gives a more trustworthy robustness estimate,
> and human evaluation confirms the gap.

The fine-tuned classifiers from the previous paper are kept only as
**adversarial-example generators** — they're the cheapest way to produce
attack candidates we can then aim at LLMs.

---

# High-level plan (5 layers)

| Layer | What | Paper section |
|---|---|---|
| L1 | Run TextFooler + BERT-Attack on 3 datasets × 3 classifiers (already done) — use these as adversarial-example sources | §3 Setup |
| L2 | Evaluate 3–4 modern LLMs (MamayLM, Lapa, Qwen, Gemini) on clean + adversarial subsets — show LLMs are not robust | §4 LLM robustness |
| L3 | Show existing attacks are flawed: a large fraction of "successful" attacks change meaning. Build a WSD-aware attack variant. | §5 WSD-aware attacks |
| L4 | Human evaluation comparing original vs WSD-aware attacks for validity, fluency, label preservation | §6 Human evaluation |
| L5 | (Stretch) One defense: adversarial training or consistency voting | §7 Defenses |

The 4th dataset (gov-requests) is **stretch** — only add if L1–L4 are
solid. Three datasets (reviews, news, unlp) already cover sentiment,
topic, and propaganda — a defensible matrix.

---

# Preliminary results we already have (MamayLM-Gemma-3-12B-IT-v1.0, 4-bit)

Source: `results/llm/runs.csv`, 15 cells run.

## Clean zero-shot accuracy vs fine-tuned classifiers

| Dataset | MamayLM clean acc | Best classifier acc | Gap |
|---|---:|---:|---:|
| reviews | **0.450** | 0.779 (xlmr) | −33 pts |
| news    | **0.754** | 0.987 (ukr-roberta) | −23 pts |
| unlp    | **0.688** | 0.817 (sbert) | −13 pts |

**Read:** MamayLM is a much weaker zero-shot classifier than the
fine-tuned baselines. This is expected, but it makes "transfer ASR" the
right metric — not absolute accuracy.

## Transfer attack success rate (% of *classifier-broken* examples that LLM also gets wrong)

| Dataset | Attack | sbert_mpnet | ukr_roberta | xlmr_base |
|---|---|---:|---:|---:|
| reviews | TextFooler  | **47.8%** | 37.1% | **48.2%** |
| reviews | BERT-Attack | 38.4% | 33.6% | *missing* |
| news    | TextFooler  | 8.2% | 9.4% | 11.0% |
| news    | BERT-Attack | **25.6%** | **24.8%** | **27.7%** |
| unlp    | TextFooler  | 28.3% | 37.5% | **47.3%** |
| unlp    | BERT-Attack | *missing* | *missing* | *missing* |

## Robustness gap (clean acc − adv acc) and consistency

| Dataset | Attack | Gap range | Consistency range |
|---|---|---:|---:|
| reviews | TF | 10.8 – 15.6 pts | 0.79 – 0.84 |
| reviews | BA | 10.5 – 11.3 pts | 0.83 – 0.84 |
| news    | TF | 0.7 – 1.3 pts | 0.94 – 0.95 |
| news    | BA | 5.2 – 5.5 pts | 0.88 – 0.89 |
| unlp    | TF | 1.8 – 7.3 pts | 0.81 – 0.83 |

## Two findings to lead with in the paper

1. **MamayLM is genuinely shaken on reviews/unlp** — transfer ASR 28–48%, gap 11–16 pts on reviews. This is exactly the LLM-not-robust story we want.
2. **News is much harder to break** — transfer ASR 8–11% (TF) but jumps to 25–28% with BERT-Attack. BA generates more aggressive substitutions (Δ≈0.27, sim≈0.44) than TF (Δ≈0.20, sim≈0.77). This already foreshadows the WSD argument: BA's "successes" are likely meaning-changing, not robust failures.

This is enough to motivate every later step.

## Gaps in current LLM data

- BERT-Attack × unlp: all 3 cells missing.
- BERT-Attack × reviews × xlmr_base: missing.
- All 15 cells use 4-bit quantization (acceptable for a preliminary table; document it; rerun reviews/unlp at higher precision before final submission).
- Only one LLM (MamayLM). No Lapa, Qwen, Gemini yet.

---

# Step-by-step plan

## Step 0 — Close the LLM grid for MamayLM (1 day, blocking)

**Why:** the preliminary table above has holes. Filling them gives us
3 datasets × 2 attacks × 3 classifiers = 18 cells per LLM, which is the
clean unit for every subsequent analysis.

- [ ] Run BERT-Attack × unlp × {sbert, ukr_roberta, xlmr} → 3 cells
- [ ] Run BERT-Attack × reviews × xlmr_base → 1 cell
- [ ] Spot-check the `wall_time_sec=0.07` row for `mamay__textfooler__reviews__ukr_roberta` — looks like a duplicate-write bug, verify the predictions.jsonl is full
- [ ] Refresh `results/llm/runs.csv` and a new `results/llm_summary.md` with the preliminary table above

**Deliverable:** complete 18-cell MamayLM table (paper Table 5a).

---

## Step 1 — Add 3 more LLMs (1–2 weeks, GPU-bound)

**Why:** L2 of the high-level plan. Need at least 3 LLM rows to claim
"LLMs are not robust" rather than "MamayLM specifically is not robust."

### Models to add (pin versions today)

| Model | HF id (proposed) | Notes |
|---|---|---|
| Lapa-LLM | `lapa-llm/Lapa-Gemma-3-12B-...` | Ukrainian-focused, Gemma-3-12B base |
| Qwen2.5-14B-Instruct | `Qwen/Qwen2.5-14B-Instruct` | Strong general open model |
| Gemini-2.5-Flash | `google/gemini-2.5-flash` (API) | Closed-weight reference; cheap; documents the "API-grade" upper bound |

If Gemini API is out of scope, swap for `google/gemma-3-12b-it`
(open-weight Gemma-3 base) so we have an apples-to-apples comparison
with Lapa.

### Tasks

- [ ] Extend `src/llm/` (or wherever the MamayLM runner lives) to a generic backend interface with backends: `hf_4bit`, `hf_fp16`, `vllm`, `gemini_api`.
- [ ] Pin per-run config in `summary.json`: `model_id`, `model_revision`, `quantization`, `backend`, `temperature=0`, `max_new_tokens`, `prompt_hash`, `prompt_version`, `date`.
- [ ] Reuse the same prompt hash across LLMs (already enforced by `prompt_hash` field). Use Ukrainian zero-shot for the main grid; few-shot only for the ablation (Step 6).
- [ ] Run each LLM on the 18 attack cells.

**Deliverable:** `results/llm_transfer.md` with the table:

| Dataset | Attack | Classifier ASR | MamayLM ASR | Lapa ASR | Qwen ASR | Gemini ASR |

→ paper Table 5.

**Decision point:** if any LLM has clean accuracy < 30% on a dataset,
exclude that cell from transfer ASR (the LLM was never solving the task
in the first place — transfer numbers become noise).

---

## Step 2 — Build the LLM eval subsets (½ day)

**Why:** running every LLM on every full attack output is wasteful.
A stratified subset is enough for the paper.

- [ ] `scripts/build_llm_subsets.py`:
  - **Subset A — Random clean control** (500/dataset). Measures baseline LLM ability per dataset.
  - **Subset B — Successful adversarial** (≤1000/dataset, balanced across 3 classifiers × 2 attacks). Used for transfer ASR.
  - **Subset C — Hard adversarial**: examples where SUCCESS for all 3 classifiers under at least one attack. Tests whether LLMs are genuinely more robust than classifiers.
- [ ] For each example in B/C, store: original text, adversarial text, source (model+attack), gold label, classifier orig/adv prediction, classifier confidence.
- [ ] Output: `data/llm_subsets/<dataset>__{A,B,C}.jsonl`.

**Deliverable:** 3 × 3 = 9 jsonl files. Frozen for the paper; never resampled.

---

## Step 3 — Show existing attacks are broken (1 week)

This is the core methodological pivot. We argue: transfer ASR over-states LLM vulnerability because many attack "successes" are meaning-changing.

### 3a. Automated WSD-style validity scoring

- [ ] `src/core/wsd.py`:
  - Input: `(orig_text, adv_text, replacement_pairs)`.
  - Output per replacement: `{sense_preserved: yes/no, confidence: float}`.
  - Implementation A (baseline): contextual embedding cosine — embed original word in original context vs replacement word in adversarial context using a Ukrainian sentence encoder; threshold on cosine.
  - Implementation B (stronger): LLM-as-judge prompt — "Does this replacement preserve the original word's sense in this sentence?" returning JSON.
  - Pick whichever performs adequately on a 100-pair calibration set; document agreement with author judgment.

- [ ] `scripts/classify_substitutions.py`:
  - For each substitution in `results/<attack>/<dataset>__<model>/examples.jsonl`, label as one of: `correct_sense`, `wrong_sense`, `register_shift`, `domain_shift`, `morph_error`, `named_entity_or_number`, `label_changing`.
  - Rule-based first pass (POS via pymorphy, regex for entities/numbers), LLM fallback for semantic categories.

- [ ] `scripts/wsd_report.py` → `results/wsd_summary.md`:
  - Stacked-bar source data: error category share per (attack, dataset, classifier).
  - Per-attack breakdown: TF vs BA — which attack is "cheating" more?
  - Expected finding: BERT-Attack has a higher `wrong_sense` + `label_changing` share, especially on news (semantic similarity 0.44 already hints at this).

**Deliverable:** Figure 3 — stacked WSD error categories.

### 3b. WSD-filtered attack variant

- [ ] `src/attacks/wsd_textfooler.py` subclassing `TextFoolerAttack`:
  - Override candidate filtering to drop wrong-sense substitutions before scoring.
  - Configurable `--wsd-threshold`.
- [ ] Optional: `wsd_bert_attack.py`. Lower priority — BA already conditions on context.
- [ ] Add `wsd_textfooler` row to `scripts/run_paper_experiments.py:ATTACKS`.
- [ ] Run on 3 datasets × 3 classifiers (9 cells).
- [ ] Re-run all LLMs on the WSD-filtered adversarial outputs (Step 1 grid extended).

**Deliverable:** comparison table `results/wsd_attack_results.md`:

| Attack | Dataset | Raw ASR (clf) | Raw ASR (LLM) | Valid-ASR (clf) | Valid-ASR (LLM) |

Expected story: WSD-filtered TF has lower raw ASR but Valid-ASR is *higher* — a more honest robustness estimate.

---

## Step 4 — Human evaluation (1–2 weeks elapsed; ~10 hours of annotation)

**Why:** this is what reviewers will ask for. Without it, every claim
about "wrong-sense substitutions" is asserted, not measured.

### 4a. Build the annotation set

- [ ] `scripts/build_validity_subset.py`:
  - Per (dataset, attack ∈ {TF, BA, WSD-TF}, classifier), sample 30 SUCCESS + 10 FAILED pairs.
  - Stratify by gold label.
  - Total: 3 datasets × 3 attacks × 3 classifiers × 40 = **1080 pairs** in principle, but compress to **~600 pairs** by sampling 1 classifier per (dataset, attack) cell.
  - Output: `data/validity/<attack>__<dataset>.jsonl` with `{id, orig_text, adv_text, true_label, orig_pred, adv_pred, replacements}`.

### 4b. Annotation schema (per pair)

| Field | Values |
|---|---|
| `meaning_preserved` | yes / partial / no |
| `label_preserved` | yes / unclear / no |
| `fluency_ok` | yes / no |
| `substitution_sense_correct` | yes / no |
| `attack_valid` | yes / no — derived: `meaning_preserved=yes AND label_preserved=yes` |

### 4c. Annotators

- [ ] Author + 1 native Ukrainian speaker (paid hourly or favor-trade). 600 pairs, ~20 sec/pair → ~3 hours per annotator.
- [ ] Compute Cohen's κ on the 600-pair overlap. Target κ ≥ 0.6 on `attack_valid`.
- [ ] Build a secondary LLM-judge (GPT-4o or Claude Sonnet) that runs the same schema. Report LLM-vs-human agreement; if κ ≥ 0.6, scale to the full attack output for `valid_attack_success_rate`.

### 4d. New metric

`valid_attack_success_rate = #(success AND attack_valid) / n_originally_correct`

- [ ] Add to `src/evaluation/metrics.py` and `runs.csv`.
- [ ] Re-render summary tables with both raw ASR and Valid-ASR side by side.

**Deliverable:** Table 4 — Valid attack analysis:

| Attack | Dataset | Raw ASR | % meaning preserved | % label preserved | Valid-ASR | Δ(Raw − Valid) |

The Δ column is the headline number: how inflated is each attack?

---

## Step 5 — LLM robustness testing framework (½ week, packaging work)

**Why:** L4 of the high-level plan. The paper claims a *framework*, not
just a benchmark. Make it real and reproducible.

- [ ] Refactor `src/llm/` into a clean package:
  - `client.py` — unified `LLMClient` (HF, vLLM, Anthropic, OpenAI, Gemini API).
  - `prompts.py` — versioned prompt templates per task; `prompt_hash` derived deterministically.
  - `parse.py` — strict label parser with explicit fail-loud behavior.
  - `runner.py` — given (LLM, dataset, subset) → write `predictions.jsonl` + `summary.json`.
- [ ] CLI: `python -m src.llm.runner --llm <id> --dataset <d> --subset <A|B|C> --mode <zero|few>`.
- [ ] Top-level: `scripts/run_llm_robustness.py` walks a YAML config and runs the full grid.
- [ ] Add a leaderboard generator: `scripts/llm_leaderboard.py` → `results/llm_leaderboard.md`. One row per (LLM, dataset), columns for clean acc, transfer ASR (TF / BA / WSD-TF), Valid-ASR.

**Deliverable:** the paper repo can be cloned by another researcher and
they can run `python -m src.llm.runner --llm any-hf-id ...` to add a new
LLM to the leaderboard. This is what "framework" means.

---

## Step 6 — Defenses (stretch, 1 week if time)

Pick **one**, not both.

### Option A — Adversarial training (preferred)

- [ ] `scripts/build_adv_train_set.py` — keep `status==SUCCESS AND semantic_sim ≥ 0.8 AND attack_valid==yes` (uses Step 4 annotations).
- [ ] `scripts/finetune_with_adv.py` — sweep mixing α ∈ {0, 0.25, 0.5}.
- [ ] Re-attack with held-out adversarial set (do not test on training adversarials).

### Option B — Consistency voting (LLM-side only, no retraining)

- [ ] `src/defenses/consistency_voting.py` — generate 3 paraphrases per input, classify all 4 versions, majority vote.
- [ ] Eval on Subset B of LLM outputs.

**Deliverable:** Table 6 — defense results:

| Defense | LLM | Clean acc | Adv acc | Transfer ASR | Valid-ASR reduction |

---

## Step 7 — Statistical rigor (always-on, ~1 day at end)

- [ ] `src/evaluation/stats.py`:
  - `bootstrap_ci(values, statistic, n=10000, alpha=0.05)`
  - `mcnemar_paired(orig_correct, adv_correct)`
  - `paired_bootstrap` for cross-LLM comparisons
- [ ] Patch `src/evaluation/writer.py:finalize` to attach 95% CIs.
- [ ] Render tables as `metric (lo, hi)`.

Apply to: classifier ASR, LLM transfer ASR, Valid-ASR, robustness gap.

---

## Step 8 — Optional: 4th dataset (gov-requests)

Only attempt if Steps 1–5 are done by week 5. See `research_todo.md` T1 for the file-level checklist. **Do not let this block the paper.**

---

# Final paper deliverables — concrete tables/figures

| # | Title | Source step |
|---|---|---|
| Table 1 | Datasets (task, classes, sizes, domain) | done |
| Table 2 | Clean accuracy: classifiers vs LLMs per dataset | Step 0 + 1 |
| Table 3 | Classifier attack results (already have, add CIs) | Step 7 |
| Table 4 | Valid attack analysis: Raw ASR vs Valid-ASR per attack | Step 4 |
| Table 5 | LLM transfer ASR: classifiers × LLMs × attacks | Step 1 |
| Table 6 | WSD-filtered attack vs original attack | Step 3b |
| Table 7 | Defense results | Step 6 (stretch) |
| Figure 1 | Pipeline diagram | end |
| Figure 2 | LLM transfer ASR bar chart per (LLM, dataset) | Step 1 |
| Figure 3 | Stacked WSD error categories per attack | Step 3a |
| Figure 4 | Human-vs-LLM-judge agreement (κ) on validity | Step 4c |

---

# Suggested execution order (6 weeks of compute, single GPU)

| Week | Goal |
|---|---|
| 1 | Step 0 (close MamayLM grid) + Step 2 (subsets) + Step 5 scaffolding |
| 2 | Step 1: run Lapa + Qwen on full grid |
| 3 | Step 1: run Gemini/Gemma-3-12B-it; Step 3a (WSD analysis on existing attacks) |
| 4 | Step 3b: WSD-filtered attack runs on 3 datasets × 3 classifiers; rerun all LLMs on these |
| 5 | Step 4: build validity subset, recruit annotator, run human eval, train LLM-judge |
| 6 | Step 7 (stats) + Step 6 (one defense) + tables/figures + writing |

If 4 weeks are available instead of 6: drop Step 6 (defenses), drop Gemini, keep MamayLM + Lapa + Qwen, keep WSD, keep human eval. The paper still has 4 contributions.

---

# Contributions claim (final)

1. **First systematic robustness evaluation of modern Ukrainian-capable LLMs** (MamayLM, Lapa, Qwen, Gemini) under TextFooler and BERT-Attack adversarial inputs.
2. **Empirical demonstration that synonym attacks are unreliable for Ukrainian**: a large fraction of "successful" attacks change word sense, register, or label.
3. **WSD-aware attack variant + Valid-Attack-Success-Rate metric** giving a more trustworthy robustness estimate.
4. **Human-validated comparison** of original vs WSD-aware attacks across 3 datasets.
5. **Open framework** for reproducing LLM robustness numbers on Ukrainian classification tasks.

(Defenses move to a 6th contribution if Step 6 ships.)

---

# What's deprioritized vs research_plan.md

| Was | Now |
|---|---|
| 4 datasets mandatory | 3 datasets, gov-requests is stretch |
| Defenses mandatory | One defense, only if time |
| Many ablations (POS, freq, budget, threshold) | Drop most; keep WSD-threshold sweep inside Step 3b |
| LLM transfer is one section among many | LLM transfer is the *primary* result |
| Validity annotation is an add-on | Validity is the methodological core |

---

# Open decisions

1. **Gemini vs Gemma-3-12B-it as the 4th LLM?** Gemini gives an "API-grade closed model" reference but adds reproducibility friction. Gemma-3-12B-it is the open base for Lapa, so direct comparison is cleaner. Lean: Gemma-3-12B-it.
2. **WSD implementation: contextual embeddings vs LLM-judge?** LLM-judge is cheaper to build and likely better, but adds a dependency. Run a 100-pair calibration in week 3 and pick the winner.
3. **Few-shot or only zero-shot for LLMs?** Zero-shot for the main grid (one less variable). Few-shot as a single-table ablation in week 3 if time.
4. **MamayLM precision**: 4-bit for cost, fp16 for credibility. Document the 4-bit results as preliminary; rerun reviews + unlp at fp16 for the final table.
