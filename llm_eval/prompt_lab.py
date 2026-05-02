"""Prompt-experimentation lab for LLM zero/few-shot classification.

Use this when an LLM scores poorly on a dataset (e.g. reviews) and you want
to iterate on the system/user prompt without re-running the full attack
evaluator. Edit ``PROMPT_VARIANTS`` below, pick one with ``--variant``, and
this script will:

1. Load ``--n-test`` rows from the dataset's test split.
2. Optionally sample ``--n-shot`` few-shot exemplars from the train split,
   stratified per label so every class is represented.
3. Run the chosen LLM on each test row and print accuracy, per-class
   accuracy, parse-fail rate, and a small confusion matrix.

Example:

    python -m llm_eval.prompt_lab \\
        --dataset reviews \\
        --variant v1_explicit_anchors \\
        --llm-config llm_eval/configs/mamay.yaml \\
        --n-test 30 --n-shot 5

Train/test paths default to the locations in ``dataset_path_local.txt``;
override with ``--train-path`` / ``--test-path`` if your layout differs.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from tqdm import tqdm

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from llm_eval.client import MamayClient, load_config  # noqa: E402
from llm_eval.parser import parse_label  # noqa: E402
from llm_eval.prompts import PROMPTS  # noqa: E402
from src.core.data import load_dataset  # noqa: E402


# --- Edit prompts here -----------------------------------------------------
#
# Each variant is a dict per dataset with two strings: ``system`` and
# ``user_template``. The user template MUST contain ``{text}`` (the input)
# and MAY contain ``{shots}`` (the rendered few-shot block — empty when
# ``--n-shot 0``). Keep ``label_vocab``/``label_names`` from PROMPTS — the
# parser already handles them.

PROMPT_VARIANTS: dict[str, dict[str, dict]] = {
    # --- baseline: matches llm_eval/prompts.py exactly -------------------
    "baseline": {
        "reviews": {
            "system": PROMPTS["reviews"]["system"],
            "user_template": "{shots}" + PROMPTS["reviews"]["user_template"],
        },
        "news": {
            "system": PROMPTS["news"]["system"],
            "user_template": "{shots}" + PROMPTS["news"]["user_template"],
        },
        "unlp": {
            "system": PROMPTS["unlp"]["system"],
            "user_template": "{shots}" + PROMPTS["unlp"]["user_template"],
        },
    },

    # --- v1: spell out what each star means with anchor adjectives -------
    "v1_explicit_anchors": {
        "reviews": {
            "system": (
                "Ви — класифікатор тональності відгуків покупців на українські "
                "товари. Прочитайте відгук і поставте оцінку від 1 до 5:\n"
                "1 — дуже негативний (товар жахливий, повертаю, не рекомендую);\n"
                "2 — негативний (є серйозні недоліки, розчарований);\n"
                "3 — нейтральний (є і плюси, і мінуси, посередньо);\n"
                "4 — позитивний (загалом задоволений, є дрібні зауваження);\n"
                "5 — дуже позитивний (у захваті, рекомендую усім).\n"
                "Зважайте на сарказм і змішані емоції. Відповідайте ЛИШЕ "
                "однією цифрою від 1 до 5 без пояснень."
            ),
            "user_template": "{shots}Відгук:\n{text}\n\nОцінка:",
        },
    },

    # --- v2: tighter rubric + chain-of-thought banned --------------------
    "v2_strict_rubric": {
        "reviews": {
            "system": (
                "Завдання: класифікація тональності українських відгуків "
                "за шкалою 1–5.\n\n"
                "Правила:\n"
                "• 5 — лише захоплені відгуки без істотних претензій.\n"
                "• 4 — позитивні відгуки з дрібними зауваженнями.\n"
                "• 3 — змішані: і похвала, і критика приблизно нарівні.\n"
                "• 2 — переважно негативні, але без різких слів.\n"
                "• 1 — різко негативні, скарги, прохання повернути товар.\n\n"
                "Виведіть РІВНО ОДНУ цифру від 1 до 5. Без слів, без пояснень, "
                "без лапок."
            ),
            "user_template": "{shots}Відгук: {text}\nОцінка (1–5):",
        },
    },

    # --- add more variants here as you iterate ---------------------------
}


@dataclass
class LabResult:
    n: int
    n_parsed: int
    n_correct: int
    accuracy: float
    parse_fail_rate: float
    per_class_acc: dict[int, float]
    per_class_n: dict[int, int]
    confusion: dict[tuple[int, int], int]
    rows: list[dict]


# --- defaults: dataset → (test_path, train_path) ---------------------------
DEFAULT_PATHS = {
    "reviews": (
        "/home/mudryi/phd_projects/xml-roberta-finetune-reviews/cross_domain_uk_reviews/test_reviews.csv",
        "/home/mudryi/phd_projects/xml-roberta-finetune-reviews/cross_domain_uk_reviews/train_reviews.csv",
    ),
    "news": (
        "/home/mudryi/phd_projects/xml-roberta-finetune-reviews/ua-news/test.csv",
        "/home/mudryi/phd_projects/xml-roberta-finetune-reviews/ua-news/train.csv",
    ),
    "unlp": (
        "/home/mudryi/phd_projects/xml-roberta-finetune-reviews/unlp_sharedtask_dataset/test.csv",
        "/home/mudryi/phd_projects/xml-roberta-finetune-reviews/unlp_sharedtask_dataset/train.csv",
    ),
}


def _label_to_surface(label: int, dataset: str) -> str:
    """Inverse of label_vocab: pick a canonical surface form to show in shots."""
    vocab = PROMPTS[dataset]["label_vocab"]
    for surface, idx in vocab.items():
        if idx == label:
            if dataset == "unlp":
                return surface.capitalize()  # "Так" / "Ні"
            return surface
    raise ValueError(f"no surface form for label {label} in dataset {dataset}")


def _sample_few_shot(
    train_examples: list[tuple[str, int]],
    n_shot: int,
    rng: random.Random,
    label_count: int,
) -> list[tuple[str, int]]:
    """Stratified sampling: try to give every class at least one example."""
    if n_shot <= 0:
        return []
    by_label: dict[int, list[tuple[str, int]]] = defaultdict(list)
    for text, lbl in train_examples:
        by_label[lbl].append((text, lbl))
    for lbl in by_label:
        rng.shuffle(by_label[lbl])

    shots: list[tuple[str, int]] = []
    # Round-robin over classes until we hit n_shot.
    cursor = {lbl: 0 for lbl in by_label}
    classes = sorted(by_label.keys())
    while len(shots) < n_shot:
        added = False
        for lbl in classes:
            if len(shots) >= n_shot:
                break
            i = cursor[lbl]
            if i < len(by_label[lbl]):
                shots.append(by_label[lbl][i])
                cursor[lbl] = i + 1
                added = True
        if not added:
            break  # exhausted
    rng.shuffle(shots)
    return shots


def _render_shots(
    shots: list[tuple[str, int]], dataset: str, max_chars: int = 600
) -> str:
    """Render few-shot block. Truncate long bodies (esp. news) to keep the
    prompt small enough for short context windows."""
    if not shots:
        return ""
    lines = ["Приклади:"]
    for text, lbl in shots:
        body = text.strip()
        if len(body) > max_chars:
            body = body[:max_chars].rstrip() + "…"
        surface = _label_to_surface(lbl, dataset)
        if dataset == "reviews":
            lines.append(f"Відгук: {body}\nОцінка: {surface}")
        elif dataset == "news":
            lines.append(f"Заголовок: {body}\nКатегорія: {surface}")
        elif dataset == "unlp":
            lines.append(f"Текст: {body}\nВідповідь: {surface}")
        else:
            lines.append(f"{body} → {surface}")
    return "\n\n".join(lines) + "\n\n"


def _build_messages(
    variant: dict, dataset: str, text: str, shots_block: str
) -> list[dict]:
    user = variant["user_template"].format(text=text, shots=shots_block)
    return [
        {"role": "system", "content": variant["system"]},
        {"role": "user", "content": user},
    ]


def run_lab(
    *,
    dataset: str,
    variant_name: str,
    client: MamayClient,
    test_path: Path,
    train_path: Optional[Path],
    n_test: int,
    n_shot: int,
    seed: int,
) -> LabResult:
    if variant_name not in PROMPT_VARIANTS:
        raise ValueError(
            f"unknown variant {variant_name!r}; have: {sorted(PROMPT_VARIANTS)}"
        )
    if dataset not in PROMPT_VARIANTS[variant_name]:
        raise ValueError(
            f"variant {variant_name!r} has no entry for dataset {dataset!r}; "
            f"datasets available in this variant: "
            f"{sorted(PROMPT_VARIANTS[variant_name])}"
        )
    variant = PROMPT_VARIANTS[variant_name][dataset]

    rng = random.Random(seed)

    test_examples = load_dataset(dataset, test_path)
    rng.shuffle(test_examples)
    test_examples = test_examples[:n_test]

    label_count = len(PROMPTS[dataset]["label_vocab"])
    shots: list[tuple[str, int]] = []
    if n_shot > 0:
        if train_path is None or not Path(train_path).exists():
            raise FileNotFoundError(
                f"few-shot requested (n_shot={n_shot}) but train_path is missing: "
                f"{train_path!r}"
            )
        # Cap train load to keep things snappy; load_dataset already caps at 10k.
        train_examples = load_dataset(dataset, train_path)
        # Avoid leaking test rows into shots (cheap text-equality check).
        test_texts = {t for t, _ in test_examples}
        train_examples = [
            (t, l) for t, l in train_examples if t not in test_texts
        ]
        shots = _sample_few_shot(train_examples, n_shot, rng, label_count)
    shots_block = _render_shots(shots, dataset)

    rows: list[dict] = []
    for text, true_lbl in tqdm(test_examples, desc=f"{variant_name}/{dataset}"):
        messages = _build_messages(variant, dataset, text, shots_block)
        raw = client.generate(messages)
        pred = parse_label(raw, dataset)
        rows.append(
            {
                "true_label": true_lbl,
                "pred_label": pred,
                "raw": raw,
                "text": text,
            }
        )

    n = len(rows)
    parsed = [r for r in rows if r["pred_label"] is not None]
    n_parsed = len(parsed)
    n_correct = sum(1 for r in parsed if r["pred_label"] == r["true_label"])
    accuracy = n_correct / n if n else 0.0
    parse_fail_rate = (n - n_parsed) / n if n else 0.0

    per_class_total: Counter = Counter(r["true_label"] for r in rows)
    per_class_correct: Counter = Counter(
        r["true_label"] for r in parsed if r["pred_label"] == r["true_label"]
    )
    per_class_acc = {
        lbl: per_class_correct[lbl] / per_class_total[lbl]
        for lbl in per_class_total
    }
    confusion: dict[tuple[int, int], int] = defaultdict(int)
    for r in parsed:
        confusion[(r["true_label"], r["pred_label"])] += 1

    return LabResult(
        n=n,
        n_parsed=n_parsed,
        n_correct=n_correct,
        accuracy=accuracy,
        parse_fail_rate=parse_fail_rate,
        per_class_acc=per_class_acc,
        per_class_n=dict(per_class_total),
        confusion=dict(confusion),
        rows=rows,
    )


def _print_result(result: LabResult, dataset: str) -> None:
    label_names = PROMPTS[dataset]["label_names"]
    print()
    print(f"  n={result.n}  parsed={result.n_parsed}  "
          f"correct={result.n_correct}")
    print(f"  accuracy        : {result.accuracy:.3f}")
    print(f"  parse_fail_rate : {result.parse_fail_rate:.3f}")
    print("  per-class accuracy:")
    for lbl in sorted(result.per_class_n):
        name = label_names[lbl] if lbl < len(label_names) else str(lbl)
        acc = result.per_class_acc.get(lbl, 0.0)
        print(f"    {lbl} ({name:<14}) n={result.per_class_n[lbl]:<3} "
              f"acc={acc:.3f}")
    if result.confusion:
        print("  confusion (true → pred : count):")
        for (t, p), c in sorted(result.confusion.items()):
            tn = label_names[t] if t < len(label_names) else str(t)
            pn = label_names[p] if p < len(label_names) else str(p)
            mark = "" if t == p else "  ✗"
            print(f"    {t} ({tn}) → {p} ({pn}) : {c}{mark}")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True, choices=sorted(DEFAULT_PATHS),
                   help="reviews | news | unlp")
    p.add_argument("--variant", default="baseline",
                   help=f"prompt variant key from PROMPT_VARIANTS "
                        f"(have: {sorted(PROMPT_VARIANTS)})")
    p.add_argument("--llm-config", default=None,
                   help="path to YAML (defaults to MamayLM)")
    p.add_argument("--n-test", type=int, default=30,
                   help="number of test rows to evaluate (default 30)")
    p.add_argument("--n-shot", type=int, default=0,
                   help="few-shot exemplars sampled from train (default 0)")
    p.add_argument("--seed", type=int, default=1914)
    p.add_argument("--test-path", default=None,
                   help="override default test CSV")
    p.add_argument("--train-path", default=None,
                   help="override default train CSV (used for few-shot)")
    p.add_argument("--save-rows", default=None,
                   help="optional path to dump per-example predictions as JSONL")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    test_path = Path(args.test_path or DEFAULT_PATHS[args.dataset][0])
    train_path = Path(args.train_path or DEFAULT_PATHS[args.dataset][1])

    cfg = load_config(args.llm_config)
    cfg.seed = args.seed
    print(f"[lab] dataset={args.dataset}  variant={args.variant}  "
          f"n_test={args.n_test}  n_shot={args.n_shot}  model={cfg.model_id}")
    client = MamayClient(cfg)

    result = run_lab(
        dataset=args.dataset,
        variant_name=args.variant,
        client=client,
        test_path=test_path,
        train_path=train_path,
        n_test=args.n_test,
        n_shot=args.n_shot,
        seed=args.seed,
    )
    _print_result(result, args.dataset)

    if args.save_rows:
        out_path = Path(args.save_rows)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            for r in result.rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[lab] wrote per-example rows to {out_path}")


if __name__ == "__main__":
    main()
