I would shape the paper around **three layers**:

1. **Core robustness benchmark**: 3 classifier models × 4 datasets × 2 attacks.
2. **Transfer to modern LLMs**: test whether adversarial examples that break classifiers also break Ukrainian/open LLMs.
3. **Quality + explanation + defense**: WSD-aware validation, linguistic analysis, and at least one lightweight defense.

TextFooler and BERT-Attack are good baselines because they are established adversarial text attacks: TextFooler was proposed as a strong utility-preserving attack for text classification and entailment, while BERT-Attack uses a masked language model to generate substitutes in a semantic-preserving way. ([arXiv][1]) For the modern LLM part, pin exact versions because this space changes quickly: MamayLM is Ukrainian-focused and reported as a 9B model, Lapa LLM is described as based on Gemma-3-12B with a Ukrainian focus, and Gemma 4 is now officially documented as an open-weight Gemma family. ([Hugging Face][2])

---

# Recommended experiment plan

## Step 1 — Define the paper’s research questions

Use 4 clear RQs.

### RQ1 — Robustness benchmark

How much do Ukrainian text classifiers degrade under TextFooler and BERT-Attack across different classification tasks?

### RQ2 — Dataset/task sensitivity

Are some Ukrainian NLP tasks more vulnerable than others: news, reviews, propaganda, government-request sentiment?

### RQ3 — Transfer to modern LLMs

Do adversarial examples that break fine-tuned classifiers also break modern Ukrainian-capable LLMs under zero-shot and few-shot prompting?

### RQ4 — WSD and defense

Can context-aware / WSD-aware filtering improve attack validity, and can simple defenses recover robustness?

This gives the paper a strong story: **benchmark → transfer → linguistic explanation → mitigation**.

---

# Step 2 — Core benchmark: your planned 3 × 4 × 2 setup

You already have the right base matrix:

| Component    | Plan                                                         |
| ------------ | ------------------------------------------------------------ |
| Models       | 3 models from previous paper                                 |
| Datasets     | News, reviews, UNLP propaganda, government-request sentiment |
| Attacks      | TextFooler, BERT-Attack                                      |
| Main results | 3 × 4 × 2 = 24 attack settings                               |

For each setting, report:

| Metric                       | Meaning                                               |
| ---------------------------- | ----------------------------------------------------- |
| Clean Accuracy / Macro-F1    | Normal model performance before attack                |
| Attacked Accuracy / Macro-F1 | Performance after attack                              |
| Attack Success Rate          | % of originally correct samples flipped by attack     |
| Robust Accuracy              | % still correct after attack                          |
| Perturbation Rate            | % of changed words/tokens                             |
| Semantic Similarity          | Whether original and attacked text are still close    |
| Query Count / Runtime        | Practical cost of attack                              |
| Valid Attack Success Rate    | attack success only among semantically valid examples |

The most important metric is **Attack Success Rate on originally correct samples**, not just attacked accuracy. Otherwise the result is distorted by examples the model already misclassified.

---

# Step 3 — Do not attack all samples blindly

Use this protocol:

1. Evaluate each model on the clean test set.
2. Select only samples correctly classified by the model.
3. Run TextFooler and BERT-Attack on those samples.
4. Save both successful and failed attacks.
5. Store every substitution: original word, replacement word, POS, sentence, label before/after, confidence before/after.

This gives you material for later WSD and linguistic analysis.

For publication quality, I would use **at least 300–500 correctly classified samples per dataset per model**, if compute allows. If not, use a balanced stratified subset and clearly report the sampling method.

---

# Step 4 — Add a “validity layer” for adversarial examples

This is very important. Reviewers may otherwise say:

> The model failed because the attack changed the meaning or label.

So create an attack-quality validation subset.

I suggest:

| Subset                         |                 Size |
| ------------------------------ | -------------------: |
| Successful TextFooler attacks  |   50–100 per dataset |
| Successful BERT-Attack attacks |   50–100 per dataset |
| Failed attacks                 |    20–50 per dataset |
| Total                          | around 400–800 pairs |

Annotate each original/adversarial pair for:

1. **Meaning preserved?** yes / partially / no
2. **Label preserved?** yes / unclear / no
3. **Fluency acceptable?** yes / no
4. **Substitution sense correct?** yes / no
5. **Attack valid?** yes / no

Then introduce a stronger metric:

> **Valid Attack Success Rate = successful attacks that preserve meaning and label / originally correct samples**

This metric can become one of your paper’s main contributions.

---

# Step 5 — Add WSD-aware analysis

This is where your previous WSD background becomes very useful.

You can add WSD in two ways: **analysis** and **attack improvement**.

## 5.1 WSD as analysis

For each substitution, classify it into one of these categories:

| Category                    | Example meaning                                  |
| --------------------------- | ------------------------------------------------ |
| Correct-sense synonym       | valid substitution                               |
| Wrong-sense synonym         | word has another sense in context                |
| Register/style shift        | formal/informal shift affects prediction         |
| Domain shift                | word technically synonym but unnatural in domain |
| Morphological error         | wrong case/gender/number                         |
| Named entity / number issue | should not have been changed                     |
| Label-changing substitution | meaning changed enough to change gold label      |

Then compare:

* TextFooler vs BERT-Attack
* dataset vs dataset
* model vs model
* successful vs failed attacks

Possible finding:

> A large fraction of successful attacks exploit sense ambiguity rather than true meaning-preserving synonymy.

That would be a strong insight.

---

## 5.2 WSD-aware attack variant

Create a third attack condition:

1. Run TextFooler / BERT-Attack candidate generation.
2. Before accepting a replacement, pass it through a WSD/context filter.
3. Keep only candidates that preserve the original word sense.
4. Run the attack with this stricter candidate set.

You can call it something like:

> **WSD-filtered SSA**
> or
> **Context-Aware SSA**

Then compare:

| Attack                 | Expected result                          |
| ---------------------- | ---------------------------------------- |
| Original TextFooler    | higher attack success, lower validity    |
| Original BERT-Attack   | higher attack success, variable validity |
| WSD-filtered attack    | lower attack success, higher validity    |
| WSD-filtered valid-ASR | more trustworthy robustness estimate     |

This is probably your best methodological addition.

---

# Step 6 — LLM transfer experiment

Your current plan is good, but adjust it.

You wrote:

> test on samples that broke most of the models

Do that, but not only that. Use three subsets.

## Subset A — Random clean control

Random clean samples from each dataset.

Purpose: measure normal zero/few-shot LLM ability.

## Subset B — Successful adversarial examples

Samples where TextFooler/BERT-Attack broke at least one classifier.

Purpose: test transferability.

## Subset C — “Hard adversarial” examples

Samples where the same adversarial example broke most or all of the 3 classifiers.

Purpose: test whether LLMs are more robust than task-specific classifiers.

This avoids cherry-picking and gives a fairer design.

---

## LLM models

Include groups, not just names.

| Group                     | Models                           |
| ------------------------- | -------------------------------- |
| Ukrainian-focused         | MamayLM, Lapa                    |
| General open-weight       | Qwen, Gemma 2, Gemma 4           |
| Optional strong reference | GPT / Claude / Gemini if allowed |

For each model, pin:

* model name
* version
* parameter size
* quantization
* inference backend
* temperature
* prompt
* date of inference

This is especially important for Lapa, Mamay, Qwen, and Gemma because model versions can change.

---

## Prompting protocol

Use at least two prompting modes:

### Zero-shot

Ukrainian instruction, constrained labels.

Example:

```text
Ти класифікатор текстів. Обери рівно одну мітку зі списку:
[LABEL_1, LABEL_2, LABEL_3]

Текст:
"..."

Відповідь дай лише назвою мітки.
```

### Few-shot

Use 2–4 examples per class if possible.

Important: use the same examples for all models.

Also run with:

* temperature = 0
* fixed max tokens
* deterministic decoding if available
* strict label parser

Metrics:

| Metric                           | Meaning                                               |
| -------------------------------- | ----------------------------------------------------- |
| Clean LLM Accuracy / F1          | baseline                                              |
| Adversarial LLM Accuracy / F1    | after perturbation                                    |
| LLM Transfer Attack Success Rate | examples that flip LLM label                          |
| Consistency                      | whether original and attacked versions get same label |
| Robustness gap                   | clean score − adversarial score                       |

A very good result table:

| Dataset | Attack | Classifier ASR | Mamay ASR | Lapa ASR | Qwen ASR | Gemma 2 ASR | Gemma 4 ASR |

This will clearly show whether modern LLMs are more robust or just fail differently.

---

# Step 7 — Add a defense experiment

You do not need many defenses. One or two well-designed defenses are enough.

I suggest these two.

## Defense 1 — Adversarial data augmentation

Train classifier with a mix of:

* clean training data
* TextFooler-generated examples
* BERT-Attack-generated examples
* optionally WSD-filtered examples

Then test on **unseen adversarial examples**, not the same ones used for training.

Report:

| Training setup | Clean F1 | TextFooler robust F1 | BERT-Attack robust F1 | Valid-ASR |
| -------------- | -------: | -------------------: | --------------------: | --------: |

Important: show whether robustness improves without destroying clean performance.

This is the most publication-friendly defense.

---

## Defense 2 — Consistency voting

For each input:

1. Generate 3–5 meaning-preserving variants.
2. Classify each version.
3. Use majority vote or probability averaging.
4. If predictions disagree strongly, mark as uncertain.

This is simple and practical.

For LLMs, you can do:

* classify original
* classify adversarial version
* classify paraphrased version
* take consistent label

This defense is attractive because it works without retraining.

---

# Step 8 — Add ablation studies

You do not need all of these, but choose 3–4.

## Ablation A — Attack budget

Run attacks with different maximum word-change limits:

* 5%
* 10%
* 20%
* maybe unrestricted baseline

This shows whether models fail after small or large changes.

## Ablation B — Semantic similarity threshold

Try stricter and weaker thresholds.

Example:

* loose
* medium
* strict

Then show:

* ASR decreases as semantic constraint becomes stricter
* valid attack rate increases

This is very important for credibility.

## Ablation C — POS-based vulnerability

Analyze which substitutions are most damaging:

* nouns
* verbs
* adjectives
* adverbs

This fits Ukrainian very well because morphology and lexical choice matter.

## Ablation D — Word frequency

Check whether rare words cause more failures.

Expected analysis:

* substitutions to rarer synonyms may cause stronger performance drops
* Ukrainian-specific or domain-specific words may be more brittle

## Ablation E — Model confidence drop

Even if prediction does not flip, measure confidence change.

This helps show “near failures.”

---

# Step 9 — Statistical testing

For publication quality, add confidence intervals and significance testing.

Use:

* bootstrap 95% confidence intervals for F1 / ASR
* McNemar’s test for paired clean vs adversarial predictions
* paired bootstrap for model comparisons
* effect sizes, not only p-values

At minimum, report confidence intervals for:

* attack success rate
* robust accuracy
* LLM transfer success rate

This makes results look much more mature.

---

# Step 10 — Suggested final experiment package

If you want a **solid full paper**, I would run this exact package:

## Mandatory

1. Clean benchmark on 3 models × 4 datasets
2. TextFooler + BERT-Attack on correctly classified samples
3. Validity annotation of adversarial examples
4. LLM transfer evaluation on clean + adversarial + hard adversarial subsets
5. WSD-aware analysis of substitutions
6. One defense: adversarial training or consistency voting

## Strong optional additions

7. WSD-filtered attack variant
8. POS / frequency / morphology analysis
9. Attack budget ablation
10. Few-shot vs zero-shot LLM comparison

If you can only add one major new thing beyond your current plan, add:

> **WSD-aware validity analysis + Valid Attack Success Rate**

If you can add two:

> **WSD-aware attack filtering + one defense**

---

# Recommended paper tables and figures

## Table 1 — Datasets

| Dataset | Task | Classes | Train/Test size | Domain | Language |
| ------- | ---- | ------: | --------------: | ------ | -------- |

## Table 2 — Clean model performance

| Model | News | Reviews | Propaganda | Gov requests |
| ----- | ---: | ------: | ---------: | -----------: |

## Table 3 — Attack results

| Dataset | Model | Attack | Clean F1 | Adv F1 | ASR | Perturbation % | Semantic sim |
| ------- | ----- | ------ | -------: | -----: | --: | -------------: | -----------: |

## Table 4 — Valid attack analysis

| Attack | Raw ASR | % meaning preserved | % label preserved | Valid-ASR |
| ------ | ------: | ------------------: | ----------------: | --------: |

## Table 5 — LLM transfer

| Dataset | Attack | Mamay | Lapa | Qwen | Gemma 2 | Gemma 4 |
| ------- | ------ | ----: | ---: | ---: | ------: | ------: |

## Table 6 — Defense results

| Defense | Clean F1 | Adv F1 | Robust accuracy | Valid-ASR reduction |
| ------- | -------: | -----: | --------------: | ------------------: |

## Figure 1 — Research pipeline

Clean data → classifier training → attacks → validity/WSD filtering → LLM transfer → defense.

## Figure 2 — Robustness drop by dataset

Bar chart: clean F1 vs adversarial F1.

## Figure 3 — WSD error categories

Stacked bar chart of substitution error types.

---

# Suggested final contribution statement

Your paper can claim something like:

1. We provide a systematic robustness evaluation of Ukrainian text classifiers under TextFooler and BERT-Attack across four classification datasets.
2. We evaluate whether adversarial examples transfer from fine-tuned classifiers to modern Ukrainian-capable LLMs under zero-shot and few-shot prompting.
3. We introduce a WSD-aware validity analysis for Ukrainian synonym substitution attacks and distinguish raw attack success from semantically valid attack success.
4. We test simple robustness defenses and analyze their trade-off between clean accuracy and adversarial robustness.

That is much stronger than just:

> We attacked several models and reported accuracy drops.

---

# Best step-by-step execution order

1. Freeze datasets, labels, train/test splits, and preprocessing.
2. Reproduce clean performance of the 3 previous models on all 4 datasets.
3. Run TextFooler and BERT-Attack only on correctly classified test samples.
4. Save detailed attack logs with substitutions and confidence changes.
5. Build the adversarial subset for LLM testing: random clean, successful attacks, hard attacks.
6. Run zero-shot LLM classification.
7. Run few-shot LLM classification.
8. Human-check or LLM-assisted-check a balanced subset for semantic validity.
9. Add WSD-aware substitution analysis.
10. Implement WSD-filtered attack if time allows.
11. Run one defense: adversarial training or consistency voting.
12. Add ablations: attack budget, semantic threshold, POS/frequency analysis.
13. Run bootstrap confidence intervals and paired significance tests.
14. Write results around the story: **Ukrainian models are vulnerable, attack validity matters, LLMs are not automatically robust, WSD-aware filtering gives a more trustworthy robustness estimate, and simple defenses help but do not fully solve the issue.**

My strongest recommendation: do **not** skip the validity/WSD part. That is what can make the paper feel like a real contribution rather than only a benchmark.

[1]: https://arxiv.org/abs/1907.11932?utm_source=chatgpt.com "Is BERT Really Robust? A Strong Baseline for Natural Language Attack on Text Classification and Entailment"
[2]: https://huggingface.co/blog/INSAIT-Institute/mamaylm-ukr?utm_source=chatgpt.com "MamayLM, передова мовна модель для української мови"
