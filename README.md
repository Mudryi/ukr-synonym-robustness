# Precision vs. Perturbation — Ukrainian Synonym-Substitution Robustness

Codebase for the UNLP 2025 paper: <https://aclanthology.org/2025.unlp-1.15/>

This repo provides a **unified runner** for two adversarial-attack methods
against Ukrainian text classifiers:

- **TextFooler-UA** — dictionary-based synonyms with pymorphy2 inflection-aware replacement, SBERT semantic-similarity filter, POS-compatibility filter.
- **BERT-Attack-UA** — MLM-based substitutes (XLM-RoBERTa-large), fastText cosine-similarity filter, normal-form check. Two modes: `classic` (top-k MLM logits) and `fill-mask` (HuggingFace pipeline).

Both attacks share a common interface, common metrics, and a single result
schema, so head-to-head comparisons are straightforward.

---

## Layout

```
src/
  core/         shared utilities (tokenization, morphology, stopwords, similarity, ...)
  attacks/      base.py + textfooler.py + bert_attack.py
  evaluation/   metrics.py + writer.py (summary.json + examples.jsonl + runs.csv)
  cli/          run_attack.py — single entry point
configs/        per-dataset YAML defaults (reviews / news / unlp)
scripts/        one-shot helpers (download_fasttext_uk.py)
data/           user-provided datasets (gitignored)
resources/      user-provided model resources, e.g. fastText (gitignored)
results/        per-run outputs (gitignored)
```

---

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Download the Ukrainian fastText model (used by BERT-Attack):

```bash
python scripts/download_fasttext_uk.py --dest resources/fasttext_uk
```

---

## External data the attacks expect

### TextFooler synonym files

Three files (paths configurable). See `data/synonyms/README.md` for schemas.

| File | Flag | Env var | Required |
|---|---|---|:-:|
| Main synonym dictionary | `--synonym-dict` | `UKR_SYN_DICT` | yes |
| Hand-curated additions | `--hand-parsed` | `UKR_SYN_HAND` | no |
| Antonym dictionary | `--antonyms` | `UKR_SYN_ANTONYMS` | no |

### BERT-Attack fastText model

Resolved (in order): `--fasttext-path`, `FASTTEXT_UK_PATH`, `resources/fasttext_uk/cbow.uk.300.bin`. A startup `FileNotFoundError` is raised if missing.

### Datasets

CSV files with columns:

- **reviews**: `text`, `label` (1–5; the loader subtracts 1 to make labels zero-indexed).
- **news**: `title`, `target` (one of `бізнес`, `новини`, `політика`, `спорт`, `технології`).
- **unlp**: `text`, `label` (0/1).

---

## Run

### TextFooler

```bash
python -m src.cli.run_attack \
  --config configs/reviews.yaml \
  --attack textfooler \
  --target-model xlm-roberta-base \
  --target-checkpoint /path/to/finetuned/ckpt \
  --synonym-dict /path/to/synonimy_info_clean.json \
  --hand-parsed /path/to/hand_parsed_top_100.json \
  --antonyms /path/to/antonimy.jsonlines \
  --output-dir results/textfooler_reviews_xlmr/
```

### BERT-Attack (fill-mask, default)

```bash
python -m src.cli.run_attack \
  --config configs/news.yaml \
  --attack bert_attack \
  --bert-mode fill-mask \
  --target-model xlm-roberta-base \
  --target-checkpoint /path/to/finetuned/ckpt \
  --output-dir results/bertatk_news_xlmr/
```

### BERT-Attack (classic mode)

```bash
python -m src.cli.run_attack \
  --config configs/news.yaml \
  --attack bert_attack \
  --bert-mode classic \
  --target-model xlm-roberta-base \
  --target-checkpoint /path/to/finetuned/ckpt \
  --output-dir results/bertatk_news_classic_xlmr/
```

A CLI flag always overrides the YAML default; pass `--n-samples 50` for a quick smoke run.

---

## Result schema

Every run writes:

- `<output-dir>/examples.jsonl` — one JSON record per attacked sample (streamed; safe for long runs):

  ```json
  {"id": 0, "attack": "textfooler", "orig_text": "…", "adv_text": "…",
   "true_label": 1, "orig_label": 1, "adv_label": 0,
   "status": "SUCCESS", "num_changes": 3, "num_queries": 47,
   "change_rate": 0.06, "replacements": [{"position": 12, "orig": "x", "new": "y"}],
   "semantic_sim": 0.83}
  ```

- `<output-dir>/summary.json` — run-level metrics:

  ```json
  {"attack": "textfooler", "dataset": "reviews",
   "n_total": 1000, "n_skipped_orig_wrong": 150, "n_attacked": 850,
   "n_success": 612, "original_accuracy": 0.85, "after_attack_accuracy": 0.238,
   "attack_success_rate": 0.72, "avg_queries": 124.6, "avg_change_rate": 0.13,
   "avg_semantic_sim": 0.81, "wall_time_sec": 1843.2, ...}
  ```

- `results/runs.csv` (or alongside `--output-dir`) — append-only one-row-per-run table for cross-run comparison.

`status` is one of `SUCCESS`, `FAILED`, `BUDGET_EXCEEDED`, `SKIPPED_ORIG_WRONG`.

---

## License

See `LICENSE`.
