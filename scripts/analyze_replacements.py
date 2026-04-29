"""Aggregate (origᴸ → newᴸ) replacement statistics from successful attacks.

Walks ``results/<attack>/<dataset>__<model>/examples.jsonl``, keeps only rows
with ``status == "SUCCESS"``, lemmatizes both sides of every replacement using
``src.core.morphology.get_normal_form`` (pymorphy2-uk — the same normalizer
used inside the attacks, see :mod:`src.core.morphology`), and reports which
(origᴸ → newᴸ) pairs flipped the classifier most often.

Per-attack tables, a combined cross-attack table, and a "most-attacked source
word" view are written to a single markdown file.

Usage
-----

    # Default: scan everything under results/, top-30 rows per table.
    python scripts/analyze_replacements.py

    # Bigger ranking, cross subset only.
    python scripts/analyze_replacements.py --top-n 50 \
        --attacks textfooler --datasets reviews

    # Custom output location.
    python scripts/analyze_replacements.py --output /tmp/replacements.md
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.core.morphology import get_normal_form  # noqa: E402

RESULTS = REPO / "results"
DATASETS = ["reviews", "news", "unlp"]
MODELS = ["ukr_roberta", "sbert_mpnet", "xlmr_base"]
ATTACK_DIRS = ["textfooler", "bert_attack"]


@lru_cache(maxsize=200_000)
def lemma(word: str) -> str:
    """Lowercase + pymorphy2 normal form. Cached because the same surface forms
    repeat heavily across the ~150k+ replacements in the grid."""
    w = word.lower().strip()
    if not w:
        return ""
    return get_normal_form(w)


def iter_success_replacements(jsonl_path: Path):
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("status") != "SUCCESS":
                continue
            for r in rec.get("replacements", []):
                yield r.get("orig", ""), r.get("new", "")


def collect(results_dir: Path, attacks, datasets, models):
    """attack_dir -> list[(orig_lemma, new_lemma, dataset, model)]."""
    data: dict[str, list[tuple[str, str, str, str]]] = defaultdict(list)
    success_docs: dict[str, int] = defaultdict(int)
    for atk in attacks:
        for ds in datasets:
            for mk in models:
                jsonl = results_dir / atk / f"{ds}__{mk}" / "examples.jsonl"
                if not jsonl.exists():
                    print(f"  skip (missing): {jsonl}")
                    continue
                cell_docs = 0
                cell_repls = 0
                with open(jsonl, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        rec = json.loads(line)
                        if rec.get("status") != "SUCCESS":
                            continue
                        cell_docs += 1
                        for r in rec.get("replacements", []):
                            o = lemma(r.get("orig", ""))
                            n = lemma(r.get("new", ""))
                            if not o or not n or o == n:
                                continue
                            data[atk].append((o, n, ds, mk))
                            cell_repls += 1
                success_docs[atk] += cell_docs
                print(f"  {atk}/{ds}__{mk}: {cell_docs} success docs, "
                      f"{cell_repls} replacement events")
    return data, success_docs


def render_pair_table(rows, title: str, top_n: int) -> str:
    pair_counter: Counter = Counter()
    cells_per_pair: dict[tuple[str, str], set[tuple[str, str]]] = defaultdict(set)
    for o, n, ds, mk in rows:
        pair_counter[(o, n)] += 1
        cells_per_pair[(o, n)].add((ds, mk))

    total = sum(pair_counter.values())
    if total == 0:
        return f"### {title}\n\n_no successful replacements_\n"

    lines = [
        f"### {title}",
        "",
        f"Total successful replacement events: **{total}** "
        f"(unique pairs: {len(pair_counter)})",
        "",
        "| rank | orig (lemma) | → | new (lemma) | count | share | "
        "distinct cells (dataset×model) |",
        "|---:|---|:---:|---|---:|---:|---:|",
    ]
    for i, ((o, n), c) in enumerate(pair_counter.most_common(top_n), 1):
        share = c / total * 100
        cells = len(cells_per_pair[(o, n)])
        lines.append(f"| {i} | {o} | → | {n} | {c} | {share:.2f}% | {cells} |")
    return "\n".join(lines) + "\n"


def render_orig_table(rows, title: str, top_n: int) -> str:
    counter = Counter(o for o, _, _, _ in rows)
    total = sum(counter.values())
    if total == 0:
        return ""
    lines = [
        f"### {title} — most-attacked source lemmas",
        "",
        "Source word (regardless of substitute) most often involved in a "
        "successful flip.",
        "",
        "| rank | orig (lemma) | count | share |",
        "|---:|---|---:|---:|",
    ]
    for i, (o, c) in enumerate(counter.most_common(top_n), 1):
        lines.append(f"| {i} | {o} | {c} | {c / total * 100:.2f}% |")
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", type=Path, default=RESULTS)
    ap.add_argument("--attacks", nargs="+", choices=ATTACK_DIRS, default=ATTACK_DIRS)
    ap.add_argument("--datasets", nargs="+", choices=DATASETS, default=DATASETS)
    ap.add_argument("--models", nargs="+", choices=MODELS, default=MODELS)
    ap.add_argument("--top-n", type=int, default=30)
    ap.add_argument("--output", type=Path, default=None,
                    help="Markdown output path "
                         "(default: <results-dir>/replacement_analysis.md).")
    args = ap.parse_args(argv)

    out_path = args.output or args.results_dir / "replacement_analysis.md"

    print(f"scanning {args.results_dir}")
    data, success_docs = collect(
        args.results_dir, args.attacks, args.datasets, args.models,
    )

    parts: list[str] = [
        "# Replacement analysis — successful attacks only\n",
        "Pairs ranked by how often a `(origᴸ → newᴸ)` substitution appeared "
        "in a `SUCCESS`-flagged adversarial example. Both sides are "
        "lower-cased and lemmatized via "
        "`src.core.morphology.get_normal_form` (pymorphy2-uk). Identical-"
        "lemma replacements (pure inflection changes) are dropped.\n",
        "Filters: "
        f"attacks={args.attacks} datasets={args.datasets} models={args.models}",
        "",
        "Successful documents per attack: "
        + ", ".join(f"{a}={success_docs.get(a, 0)}" for a in args.attacks),
        "",
    ]

    for atk in args.attacks:
        rows = data.get(atk, [])
        parts.append(f"## attack = {atk}\n")
        parts.append(render_pair_table(rows, f"{atk} — top replacement pairs",
                                       args.top_n))
        parts.append(render_orig_table(rows, atk, args.top_n))

    all_rows = [r for atk in args.attacks for r in data.get(atk, [])]
    parts.append("## across all attacks\n")
    parts.append(render_pair_table(all_rows, "combined — top replacement pairs",
                                   args.top_n))
    parts.append(render_orig_table(all_rows, "combined", args.top_n))

    text = "\n".join(parts)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    print(f"\nwrote {out_path}")
    print("=" * 72)
    print(text)


if __name__ == "__main__":
    main()
