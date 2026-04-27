# Next-step recommendations

Captured during the unification refactor (April 2026). Listed in priority
order, top-to-bottom.

## 1. Reproducibility seed wiring (high)

`--seed` currently sets `random`, `numpy`, `torch`, and the dataframe
sub-sample seed in `load_dataset`. Verify that this propagates into:

- the SBERT encoder (sentence-transformers respects torch's seed);
- the HuggingFace `fill-mask` pipeline (it doesn't sample, but check `top_k`);
- pymorphy2 (deterministic, no seed needed).

Add a one-liner CI check that runs the same command twice and diffs the
resulting `summary.json` — they should be byte-identical.

## 2. Resumability on long runs (high)

`ResultWriter` already streams `examples.jsonl`, so partial output survives a
crash. Add a `--resume` flag that:

1. reads existing `examples.jsonl` and collects seen `id`s;
2. skips them when iterating the dataset;
3. re-aggregates including the existing rows on `finalize()`.

Critical for the news dataset (~10 k rows, 6+ hours per attack run).

## 3. Batched candidate prediction (high)

Per-candidate prediction is the hot loop. Both attacks do single-candidate
forwards. Add `Predictor.predict_batch(texts, batch_size)` and let:

- `BertAttack._run_classic` / `_run_fill_mask` batch all candidates per target
  position;
- `TextFoolerAttack` already batches per position — confirm it uses the
  unified path.

Expected 5–10× speedup on GPU.

## 4. HuggingFace-hosted datasets (medium)

Replace the local CSV paths in `configs/*.yaml` with HF dataset IDs once the
splits are public:

```yaml
dataset-path: hf://lang-uk/ua-news:test
```

`load_dataset` would dispatch to `datasets.load_dataset(...)` for the `hf://`
prefix.

## 5. Adversarial-training feedback loop (medium)

Add `scripts/build_adv_train_set.py`:

- read all `examples.jsonl` files under `results/`;
- keep only `status == SUCCESS` and `semantic_sim >= 0.8`;
- emit a CSV in the same schema as the source dataset, ready for
  robustness-aware fine-tuning.

## 6. Cross-run leaderboard (medium)

Add `python -m src.evaluation.compare runs.csv` that prints a markdown table:

```
attack          | dataset | model            | ASR  | queries | change-rate
textfooler      | reviews | xlm-roberta-base | 0.72 |   124.6 |       0.13
bert_attack:fm  | reviews | xlm-roberta-base | 0.81 |    98.2 |       0.10
...
```

`runs.csv` already has every column needed.

## 7. Tests beyond smoke (medium)

Unit tests for the trickiest pure-functions:

- `core/morphology.py:replace_word` — pin behaviour for ~10 well-known
  Ukrainian words (gender / number / case agreement).
- `core/pos_filter.py:pos_filter` — interchangeable-POS edges (PRCL ↔ ADVB,
  ADJF ↔ NPRO).
- `core/synonym_dict.py:read_and_clean_synonym_dict` — fixture with one entry
  per branch (with/without antonyms, with/without hand-parsed).
- `core/data.py:load_dataset` — three tiny CSVs, one per dataset.

## 8. CI (low — once tests exist)

GitHub Actions:

- `pip install -r requirements.txt` (cached).
- run unit tests (no GPU needed).
- run the smoke attack on `tests/fixtures/reviews_tiny.csv` with a small
  HuggingFace model (e.g. `prajjwal1/bert-tiny`) for both attacks.

Should complete in <5 minutes per push.

## 9. Lint / format (low)

`ruff` + `black` + `isort`. Pre-commit hooks. The codebase has mixed quote
styles, trailing whitespace, and inconsistent imports.

## 10. Replace pymorphy2 (low)

`pymorphy2` is unmaintained. Try `pymorphy3` (drop-in) or `stanza` for
Ukrainian. Affects `src/core/morphology.py` only; the rest of the codebase
talks to a thin `morph` wrapper.

---

## Things that exist in the paper but not the repo

For completeness — these were called out in the original README but no code
was committed; outside the scope of this refactor:

- `models/` — fine-tuning scripts and checkpoints for Ukr-RoBERTa,
  XLM-RoBERTa-large, SBERT.
- `evaluation/human_eval/` — human-evaluation templates from the paper.
- GPT-4o probing code.
- The 9 k-group synonym lexicon itself (only its loader is here).
