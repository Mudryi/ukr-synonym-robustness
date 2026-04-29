# llm_eval — local LLM evaluation on existing attack outputs

Self-contained harness that loads an instruction-tuned LLM (default:
`INSAIT-Institute/MamayLM-Gemma-3-12B-IT-v1.0`) and measures how it
classifies the original and adversarial texts produced by `src/cli/run_attack.py`.

## Install

```bash
pip install -r requirements-llm.txt
```

## Smoke test (5 examples)

```bash
python -m llm_eval.cli \
  --attack-dir results/textfooler/reviews__ukr_roberta \
  --dataset reviews \
  --llm-config llm_eval/configs/mamay.yaml \
  --output-dir /tmp/llm_smoke \
  --limit 5
```

Then inspect `/tmp/llm_smoke/predictions.jsonl` and confirm
`llm_orig_parsed=true` for all rows.

## One full cell

```bash
python -m llm_eval.cli \
  --attack-dir results/textfooler/reviews__ukr_roberta \
  --dataset reviews \
  --llm-config llm_eval/configs/mamay.yaml \
  --output-dir results/llm/mamay__textfooler__reviews__ukr_roberta
```

Resumable — re-running with the same `--output-dir` skips already-processed `id`s.

## Full grid (deferred — run only after one cell looks clean)

```bash
python -m llm_eval.cli --grid \
  --llm-config llm_eval/configs/mamay.yaml \
  --results-root results
```

Sweeps `results/textfooler/*/` and `results/bert_attack/*/`, writing to
`results/llm/<llm>__<attack>__<dataset>__<classifier>/`.

## Hardware notes

- **RTX 3090 / 4090 (24 GB)** — uses 4-bit nf4 quantization with bf16 compute.
  Weights ≈ 7 GB; total memory with KV cache ≈ 12 GB. Decode ~2–4 s per call.
- **Apple Silicon M3/M4 (32 GB+)** — fp16 on MPS, no quantization.
  Weights ≈ 24 GB. Decode ~5–10 s per call. Good for development; slow for the
  full grid.
- **A6000 (48 GB) / A100 / H100** — set `backend: cuda_bf16` in the config to
  skip quantization for max quality.

Override the backend:

```yaml
# llm_eval/configs/mamay.yaml
backend: cuda_bf16
```

## Output schema

Per cell (`results/llm/<llm>__<attack>__<dataset>__<classifier>/`):

- `predictions.jsonl` — one row per example with `id, true_label,
  classifier_*_label, classifier_status, llm_orig_raw, llm_adv_raw,
  llm_orig_label, llm_adv_label, llm_*_parsed`.
- `summary.json` — aggregated metrics (`llm_clean_acc`, `llm_adv_acc`,
  `robustness_gap`, `llm_consistency`, `llm_transfer_asr`,
  `n_transfer_eligible`) plus full config + prompt hash.

Cross-cell: `results/llm/runs.csv`.

## Metrics

- `llm_clean_acc` — accuracy on `orig_text`.
- `llm_adv_acc` — accuracy on `adv_text`.
- `robustness_gap` — `clean_acc − adv_acc`.
- `llm_consistency` — fraction with same label on orig and adv.
- `llm_transfer_asr` — among examples where the classifier was successfully
  attacked **and** the LLM was originally correct on `orig_text`, the
  fraction where the LLM is wrong on `adv_text`. Operational definition of
  "the attack transferred."

## Layout

```
llm_eval/
  client.py        # MamayClient + LLMConfig (auto-detects CUDA/MPS/CPU)
  prompts.py       # Ukrainian zero-shot templates for reviews/news/unlp
  parser.py        # Strict label parser (no silent default class)
  evaluator.py     # Per-cell loop + JSONL writer
  metrics.py       # Aggregation → summary.json + runs.csv
  cli.py           # python -m llm_eval.cli ...
  configs/mamay.yaml
```
