# LLM robustness benchmark

Source: `results/llm/runs.csv` + per-run `predictions.jsonl` for the joined metrics. One row per LLM. Counts aggregate across the 3 classifier cells per (LLM × dataset × attack).

**Definitions**
- **orig**: LLM accuracy on `orig_text` (over parsed_orig rows).
- **adv**: LLM accuracy on `adv_text` (over parsed_adv rows).
- **Δ**: absolute drop, `orig − adv`. **%**: relative drop, `Δ / orig`.
- **cond-ASR** (conditional ASR): `P(adv wrong | orig correct) = (orig_correct AND adv_wrong) / orig_correct`. Computed over rows where both orig and adv parsed.
- **flip**: `(orig_label ≠ adv_label) / n_total` — any LLM label change, right or wrong. Parse-failed rows contribute 0 to the numerator.

## Main table — per attack × dataset

| LLM | TextFooler<br>reviews | TextFooler<br>news | TextFooler<br>unlp | BERT-Attack<br>reviews | BERT-Attack<br>news | BERT-Attack<br>unlp |
|---|---|---|---|---|---|---|
| Mamay | 0.450 → 0.311<br>Δ +0.139 (+30.9%)<br>cond-ASR 0.334 · flip 0.200 | 0.754 → 0.744<br>Δ +0.010 (+1.3%)<br>cond-ASR 0.038 · flip 0.053 | 0.688 → 0.646<br>Δ +0.043 (+6.2%)<br>cond-ASR 0.165 · flip 0.184 | 0.450 → 0.339<br>Δ +0.110 (+24.6%)<br>cond-ASR 0.270 · flip 0.171 | 0.754 → 0.701<br>Δ +0.054 (+7.1%)<br>cond-ASR 0.104 · flip 0.116 | 0.688 → 0.649<br>Δ +0.039 (+5.7%)<br>cond-ASR 0.085 · flip 0.078 |
| Lapa | 0.286 → 0.207<br>Δ +0.079 (+27.5%)<br>cond-ASR 0.308 · flip 0.138 | 0.648 → 0.634<br>Δ +0.014 (+2.2%)<br>cond-ASR 0.061 · flip 0.074 | 0.610 → 0.620<br>Δ -0.010 (-1.6%)<br>cond-ASR 0.066 · flip 0.090 | 0.286 → 0.238<br>Δ +0.048 (+16.7%)<br>cond-ASR 0.206 · flip 0.104 | 0.648 → 0.601<br>Δ +0.047 (+7.2%)<br>cond-ASR 0.124 · flip 0.136 | 0.610 → 0.615<br>Δ -0.005 (-0.9%)<br>cond-ASR 0.036 · flip 0.049 |
| Gemma-3 | 0.407 → 0.302<br>Δ +0.105 (+25.9%)<br>cond-ASR 0.297 · flip 0.189 | 0.758 → 0.743<br>Δ +0.015 (+2.0%)<br>cond-ASR 0.046 · flip 0.067 | — | — | — | — |

## TextFooler — expanded view

| LLM | Dataset | orig_acc | adv_acc | abs_drop | pct_drop | cond_ASR | n_cond_eligible | flip_rate | n_total |
|---|---|---|---|---|---|---|---|---|---|
| Mamay | reviews | 0.450 | 0.311 | +0.139 | +30.9% | 0.334 | 13182 | 0.200 | 29307 |
| Mamay | news | 0.754 | 0.744 | +0.010 | +1.3% | 0.038 | 22628 | 0.053 | 30000 |
| Mamay | unlp | 0.688 | 0.646 | +0.043 | +6.2% | 0.165 | 789 | 0.184 | 1146 |
| Lapa | reviews | 0.286 | 0.207 | +0.079 | +27.5% | 0.308 | 8382 | 0.138 | 29307 |
| Lapa | news | 0.648 | 0.634 | +0.014 | +2.2% | 0.061 | 19258 | 0.074 | 30000 |
| Lapa | unlp | 0.610 | 0.620 | -0.010 | -1.6% | 0.066 | 699 | 0.090 | 1146 |
| Gemma-3 | reviews | 0.407 | 0.302 | +0.105 | +25.9% | 0.297 | 7948 | 0.189 | 19538 |
| Gemma-3 | news | 0.758 | 0.743 | +0.015 | +2.0% | 0.046 | 22674 | 0.067 | 30000 |

## BERT-Attack — expanded view

| LLM | Dataset | orig_acc | adv_acc | abs_drop | pct_drop | cond_ASR | n_cond_eligible | flip_rate | n_total |
|---|---|---|---|---|---|---|---|---|---|
| Mamay | reviews | 0.450 | 0.339 | +0.110 | +24.6% | 0.270 | 13182 | 0.171 | 29307 |
| Mamay | news | 0.754 | 0.701 | +0.054 | +7.1% | 0.104 | 22629 | 0.116 | 30000 |
| Mamay | unlp | 0.688 | 0.649 | +0.039 | +5.7% | 0.085 | 789 | 0.078 | 1146 |
| Lapa | reviews | 0.286 | 0.238 | +0.048 | +16.7% | 0.206 | 8382 | 0.104 | 29307 |
| Lapa | news | 0.648 | 0.601 | +0.047 | +7.2% | 0.124 | 19257 | 0.136 | 30000 |
| Lapa | unlp | 0.610 | 0.615 | -0.005 | -0.9% | 0.036 | 699 | 0.049 | 1146 |

## Coverage

| LLM | TextFooler × reviews | TextFooler × news | TextFooler × unlp | BERT-Attack × reviews | BERT-Attack × news | BERT-Attack × unlp |
|---|---|---|---|---|---|---|
| Mamay | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| Lapa | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 | 3/3 |
| Gemma-3 | 2/3 | 3/3 | — | — | — | — |
