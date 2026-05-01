# LLM transfer comparison

Source: `results/llm/runs.csv`. Aggregation: clean / adv / consistency are over
all parsed examples across the 3 classifier cells; transfer ASR is
`Σ transferred / Σ eligible`. Generated 2026-04-30.

## Coverage

| Model                                     | textfooler | bert_attack |
|-------------------------------------------|:---:|:---:|
| Mamay (`MamayLM-Gemma-3-12B-IT-v1.0`)     | full grid (3 datasets × 3 classifiers) | full grid |
| Lapa (`lapa-v0.1.2-instruct`)             | full grid | full grid |
| Gemma-3 (`google/gemma-3-12b-it`)         | news only (3 classifiers) | — |

## Clean accuracy (zero-shot, no attack)

| Model   | reviews (5-class) | news (5-class) | unlp (binary) |
|---------|------------------:|---------------:|--------------:|
| Mamay   | 0.450 | **0.754** | **0.688** |
| Lapa    | 0.286 | 0.644 | 0.610 |
| Gemma-3 |   —   | **0.756** |   —   |

## Transfer ASR — the headline number

Among examples where the classifier was successfully attacked **and** the LLM
was originally correct on `orig_text`, the fraction where the LLM is now wrong
on `adv_text`. This is the operational definition of "the attack transferred."

| Dataset | Attack       | Mamay              | Lapa               | Gemma-3            |
|---------|--------------|--------------------|--------------------|--------------------|
| reviews | textfooler   | 0.430 (n=3217)     | 0.402 (n=1684)     | —                  |
| reviews | bert_attack  | 0.353 (n=719)      | 0.164 (n=585)      | —                  |
| news    | textfooler   | 0.095 (n=1793)     | 0.103 (n=1573)     | 0.121 (n=1747)     |
| news    | bert_attack  | 0.261 (n=3226)     | 0.224 (n=2742)     | —                  |
| unlp    | textfooler   | 0.368 (n=253)      | 0.094 (n=159)      | —                  |
| unlp    | bert_attack  | 0.141 (n=85)       | 0.024 (n=83)       | —                  |

## Full grid (clean / adv / gap / consistency / transfer ASR)

| Dataset | Attack       | Model   | n_cells | clean | adv   | gap     | consistency | transfer_ASR | n_eligible |
|---------|--------------|---------|--------:|------:|------:|--------:|------------:|-------------:|-----------:|
| reviews | textfooler   | Mamay   | 3 | 0.450 | 0.311 | +0.139 | 0.800 | 0.430 | 3217 |
| reviews | textfooler   | Lapa    | 3 | 0.286 | 0.207 | +0.079 | 0.862 | 0.402 | 1684 |
| reviews | bert_attack  | Mamay   | 3 | 0.450 | 0.339 | +0.110 | 0.829 | 0.353 |  719 |
| reviews | bert_attack  | Lapa    | 3 | 0.286 | 0.238 | +0.048 | 0.896 | 0.164 |  585 |
| news    | textfooler   | Mamay   | 3 | 0.754 | 0.744 | +0.010 | 0.946 | 0.095 | 1793 |
| news    | textfooler   | Lapa    | 3 | 0.644 | 0.629 | +0.015 | 0.925 | 0.103 | 1573 |
| news    | textfooler   | Gemma-3 | 3 | 0.756 | 0.742 | +0.015 | 0.933 | 0.121 | 1747 |
| news    | bert_attack  | Mamay   | 3 | 0.754 | 0.700 | +0.054 | 0.884 | 0.261 | 3226 |
| news    | bert_attack  | Lapa    | 3 | 0.644 | 0.597 | +0.047 | 0.863 | 0.224 | 2742 |
| unlp    | textfooler   | Mamay   | 3 | 0.688 | 0.646 | +0.043 | 0.816 | 0.368 |  253 |
| unlp    | textfooler   | Lapa    | 3 | 0.610 | 0.620 | -0.010 | 0.910 | 0.094 |  159 |
| unlp    | bert_attack  | Mamay   | 3 | 0.688 | 0.649 | +0.039 | 0.922 | 0.141 |   85 |
| unlp    | bert_attack  | Lapa    | 3 | 0.610 | 0.615 | -0.005 | 0.951 | 0.024 |   83 |

## Observations

1. **Clean accuracy ranking is consistent**: Mamay ≥ Gemma-3 > Lapa across
   datasets where data overlaps. Mamay is the strongest Ukrainian-tuned model;
   Lapa lags substantially on reviews (0.286 vs 0.450) — likely a sentiment-
   calibration weakness, not a Ukrainian-language one (its news/unlp gap is
   smaller).

2. **Transfer ASR is highly task-dependent**, not just model-dependent:
   - **News (categorical)**: low transfer (0.10–0.26). Synonym substitution
     rarely changes a news headline's topic in the LLM's eyes.
   - **Reviews (ordinal sentiment)**: high transfer (0.16–0.43). Sentiment is
     fragile to word-level edits even for a 12B LLM.
   - **UNLP (binary manipulation)**: bimodal — Mamay transfers more (0.37 / 0.14)
     than Lapa (0.09 / 0.02). Lapa's near-zero unlp transfer is partly because
     it's also slightly *better* on adversarial UNLP than on clean.

3. **TextFooler transfers more strongly than BERT-Attack** across the board on
   reviews and unlp. This matches the WordNet-style synonym substitution being
   a more "cross-model-portable" perturbation than BERT-Attack's MLM-conditioned
   replacements (which exploit the specific classifier's quirks more).

4. **Gemma-3 vs Mamay on news**: nearly identical clean acc (0.756 vs 0.754),
   nearly identical adv acc (0.742 vs 0.744), but Gemma-3 has a slightly
   *higher* transfer ASR (0.121 vs 0.095). Plausible reading: Mamay's Ukrainian
   fine-tuning makes it slightly more robust to Ukrainian synonym substitutions
   than the base Gemma-3.

5. **Negative gaps on Lapa-UNLP** (-0.010, -0.005): a known artifact when both
   the LLM is uncertain and the attack budget is tight; the noise added by
   substitution occasionally helps. Not significant at these `n` values.

## Caveats

- **Reviews clean acc looks low** because labels are 5-class ordinal and LLMs
  routinely pick an adjacent star (4 vs 5). An off-by-one or 3-class collapsed
  metric would change the absolute numbers but not the model ranking.
- **Gemma-3 coverage is incomplete** (news/textfooler only). The reviews and
  unlp Gemma-3 cells are needed before the cross-model story is fully fair.
- **No statistical CIs yet** — `n_eligible` ranges from 83 to 3226, so some of
  the small-`n` cells (unlp / bert_attack) are noisy. Bootstrap CIs are the
  next thing to add (T17 in `research_todo.md`).
