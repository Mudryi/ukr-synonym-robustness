"""Build the cross-LLM robustness comparison table.

Reads ``results/llm/runs.csv`` (produced by ``llm_eval/cli.py``) and emits
a markdown report at ``results/llm_benchmark.md`` with one row per LLM and
per-(attack × dataset) cells.

Per-cell metrics aggregated across the 3 classifier rows per
(model, dataset, attack):

- ``original_accuracy``  — Σ correct_orig / Σ parsed_orig
- ``adversarial_acc``    — Σ correct_adv  / Σ parsed_adv
- ``absolute_drop``      — original − adversarial
- ``percentage_drop``    — absolute_drop / original  (relative drop)
- ``conditional_ASR``    — Σ (orig_correct AND adv_wrong) / Σ orig_correct
                           — independent of which classifier was attacked.
- ``flip_rate``          — Σ (orig_label ≠ adv_label) / Σ n_total
                           — any change in the LLM's label, right or wrong.

The two new metrics need a per-row join of orig vs adv that ``runs.csv``
doesn't carry, so we re-read each run's ``predictions.jsonl`` from the
``output_dir`` recorded in the CSV.

Re-run after new (model × dataset × attack × classifier) cells are added:

    python scripts/build_llm_benchmark.py
    python scripts/build_llm_benchmark.py --runs-csv results/llm/runs.csv \\
                                          --output results/llm_benchmark.md
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


# ----------------------------------------------------------------------------
# Display order — extend these lists when new models/datasets/attacks land.
# Names not in *_ORDER are appended in alphabetical order.

MODEL_ORDER = ["Mamay", "Lapa", "Gemma-3", "Qwen3"]
DATASET_ORDER = ["reviews", "news", "unlp"]
ATTACK_ORDER = ["textfooler", "bert_attack"]
ATTACK_DISPLAY = {"textfooler": "TextFooler", "bert_attack": "BERT-Attack"}


def short_model_name(model_id: str) -> str:
    m = model_id.lower()
    if "mamay" in m:
        return "Mamay"
    if "lapa" in m:
        return "Lapa"
    if "gemma-3" in m:
        return "Gemma-3"
    if "qwen3" in m:
        return "Qwen3"
    if "qwen" in m:
        return "Qwen"
    return model_id.split("/")[-1]


# ----------------------------------------------------------------------------

def load_runs(runs_csv: Path) -> list[dict]:
    """Deduplicate by (model, dataset, attack, classifier) keeping the row
    with the largest wall_time_sec — drops resume artifacts (wall_time≈0)."""
    with open(runs_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    best: dict[tuple, dict] = {}
    for r in rows:
        key = (r["llm_model"], r["dataset"], r["attack"], r["classifier"])
        wt = float(r.get("wall_time_sec") or 0.0)
        if key not in best or wt > float(best[key].get("wall_time_sec") or 0.0):
            best[key] = r
    return list(best.values())


def per_run_counts(predictions_path: Path) -> dict:
    """Compute raw counts for one run by streaming its predictions.jsonl.

    Returns counts (not rates) so we can sum across classifier cells without
    losing information from the denominators.
    """
    counts = {
        "n_total": 0,
        "n_parsed_orig": 0,
        "n_parsed_adv": 0,
        "n_parsed_both": 0,
        "n_correct_orig": 0,           # over parsed_orig
        "n_correct_adv": 0,            # over parsed_adv
        "n_orig_correct_both_parsed": 0,        # parsed_both AND orig_correct
        "n_orig_correct_and_adv_wrong": 0,      # numerator for conditional ASR
        "n_flipped": 0,                          # parsed_both AND orig_label != adv_label
    }
    with open(predictions_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            counts["n_total"] += 1
            po = r["llm_orig_parsed"]
            pa = r["llm_adv_parsed"]
            if po:
                counts["n_parsed_orig"] += 1
                if r["llm_orig_label"] == r["true_label"]:
                    counts["n_correct_orig"] += 1
            if pa:
                counts["n_parsed_adv"] += 1
                if r["llm_adv_label"] == r["true_label"]:
                    counts["n_correct_adv"] += 1
            if po and pa:
                counts["n_parsed_both"] += 1
                if r["llm_orig_label"] != r["llm_adv_label"]:
                    counts["n_flipped"] += 1
                if r["llm_orig_label"] == r["true_label"]:
                    counts["n_orig_correct_both_parsed"] += 1
                    if r["llm_adv_label"] != r["true_label"]:
                        counts["n_orig_correct_and_adv_wrong"] += 1
    return counts


def aggregate(rows: list[dict], runs_csv: Path) -> dict[tuple, dict]:
    """Aggregate counts across classifier cells per (model, dataset, attack).

    ``output_dir`` in runs.csv is interpreted relative to the runs.csv parent
    if it isn't an absolute path that resolves on its own.
    """
    runs_root = runs_csv.parent.resolve()
    agg: dict[tuple, dict] = defaultdict(lambda: {
        "n_total": 0, "n_parsed_orig": 0, "n_parsed_adv": 0, "n_parsed_both": 0,
        "n_correct_orig": 0, "n_correct_adv": 0,
        "n_orig_correct_both_parsed": 0, "n_orig_correct_and_adv_wrong": 0,
        "n_flipped": 0, "n_classifier_cells": 0,
    })
    for r in rows:
        out = Path(r["output_dir"])
        pred = out / "predictions.jsonl"
        if not pred.exists():
            # output_dir may be repo-relative; re-resolve from runs.csv parent's parent (project root).
            pred = (runs_root.parent / out / "predictions.jsonl").resolve()
        if not pred.exists():
            print(f"[warn] skipping {r['output_dir']} — predictions.jsonl missing")
            continue
        c = per_run_counts(pred)
        key = (short_model_name(r["llm_model"]), r["dataset"], r["attack"])
        a = agg[key]
        for k in c:
            a[k] += c[k]
        a["n_classifier_cells"] += 1
    return agg


def cell_metrics(a: dict) -> dict:
    """Convert raw counts to reportable metrics."""
    def div(num, den):
        return (num / den) if den else 0.0
    orig = div(a["n_correct_orig"], a["n_parsed_orig"])
    adv = div(a["n_correct_adv"], a["n_parsed_adv"])
    abs_drop = orig - adv
    pct_drop = div(abs_drop, orig)
    cond_asr = div(a["n_orig_correct_and_adv_wrong"], a["n_orig_correct_both_parsed"])
    flip_rate = div(a["n_flipped"], a["n_total"])
    return {
        "orig": orig, "adv": adv,
        "abs_drop": abs_drop, "pct_drop": pct_drop,
        "cond_asr": cond_asr, "n_cond_eligible": a["n_orig_correct_both_parsed"],
        "flip_rate": flip_rate,
    }


def fmt_main_cell(m: dict | None) -> str:
    if m is None:
        return "—"
    return (
        f"{m['orig']:.3f} → {m['adv']:.3f}<br>"
        f"Δ {m['abs_drop']:+.3f} ({m['pct_drop']*100:+.1f}%)<br>"
        f"cond-ASR {m['cond_asr']:.3f} · flip {m['flip_rate']:.3f}"
    )


def ordered(present: set[str], canonical: list[str]) -> list[str]:
    head = [x for x in canonical if x in present]
    tail = sorted(x for x in present if x not in canonical)
    return head + tail


# ----------------------------------------------------------------------------

def build_markdown(agg: dict[tuple, dict], runs_csv: Path) -> str:
    models = ordered({k[0] for k in agg}, MODEL_ORDER)
    datasets = ordered({k[1] for k in agg}, DATASET_ORDER)
    attacks = ordered({k[2] for k in agg}, ATTACK_ORDER)

    L: list[str] = []
    L.append("# LLM robustness benchmark")
    L.append("")
    L.append(
        f"Source: `{runs_csv}` + per-run `predictions.jsonl` for the joined "
        "metrics. One row per LLM. Counts aggregate across the 3 classifier "
        "cells per (LLM × dataset × attack)."
    )
    L.append("")
    L.append("**Definitions**")
    L.append("- **orig**: LLM accuracy on `orig_text` (over parsed_orig rows).")
    L.append("- **adv**: LLM accuracy on `adv_text` (over parsed_adv rows).")
    L.append("- **Δ**: absolute drop, `orig − adv`. **%**: relative drop, `Δ / orig`.")
    L.append("- **cond-ASR** (conditional ASR): "
             "`P(adv wrong | orig correct) = (orig_correct AND adv_wrong) / orig_correct`. "
             "Computed over rows where both orig and adv parsed.")
    L.append("- **flip**: `(orig_label ≠ adv_label) / n_total` — any LLM label change, "
             "right or wrong. Parse-failed rows contribute 0 to the numerator.")
    L.append("")

    # ---- main compact table -------------------------------------------------

    header = ["LLM"]
    for atk in attacks:
        for ds in datasets:
            header.append(f"{ATTACK_DISPLAY.get(atk, atk)}<br>{ds}")
    L.append("## Main table — per attack × dataset")
    L.append("")
    L.append("| " + " | ".join(header) + " |")
    L.append("|" + "|".join(["---"] * len(header)) + "|")
    for m in models:
        row = [m]
        for atk in attacks:
            for ds in datasets:
                key = (m, ds, atk)
                row.append(fmt_main_cell(cell_metrics(agg[key]) if key in agg else None))
        L.append("| " + " | ".join(row) + " |")
    L.append("")

    # ---- per-attack expanded views -----------------------------------------

    for atk in attacks:
        L.append(f"## {ATTACK_DISPLAY.get(atk, atk)} — expanded view")
        L.append("")
        cols = [
            "LLM", "Dataset", "orig_acc", "adv_acc", "abs_drop", "pct_drop",
            "cond_ASR", "n_cond_eligible", "flip_rate", "n_total",
        ]
        L.append("| " + " | ".join(cols) + " |")
        L.append("|" + "|".join(["---"] * len(cols)) + "|")
        for m in models:
            for ds in datasets:
                key = (m, ds, atk)
                if key not in agg:
                    continue
                a = agg[key]
                met = cell_metrics(a)
                L.append(
                    f"| {m} | {ds} | "
                    f"{met['orig']:.3f} | {met['adv']:.3f} | "
                    f"{met['abs_drop']:+.3f} | {met['pct_drop']*100:+.1f}% | "
                    f"{met['cond_asr']:.3f} | {met['n_cond_eligible']} | "
                    f"{met['flip_rate']:.3f} | {a['n_total']} |"
                )
        L.append("")

    # ---- coverage matrix ---------------------------------------------------

    L.append("## Coverage")
    L.append("")
    cov_cols = ["LLM"] + [
        f"{ATTACK_DISPLAY.get(a, a)} × {d}" for a in attacks for d in datasets
    ]
    L.append("| " + " | ".join(cov_cols) + " |")
    L.append("|" + "|".join(["---"] * len(cov_cols)) + "|")
    for m in models:
        row = [m]
        for atk in attacks:
            for ds in datasets:
                key = (m, ds, atk)
                row.append(f"{agg[key]['n_classifier_cells']}/3" if key in agg else "—")
        L.append("| " + " | ".join(row) + " |")
    L.append("")

    return "\n".join(L)


# ----------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--runs-csv", type=Path, default=Path("results/llm/runs.csv"))
    p.add_argument("--output", type=Path, default=Path("results/llm_benchmark.md"))
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if not args.runs_csv.exists():
        raise SystemExit(f"runs.csv not found at {args.runs_csv}")
    rows = load_runs(args.runs_csv)
    agg = aggregate(rows, args.runs_csv)
    md = build_markdown(agg, args.runs_csv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(md, encoding="utf-8")
    print(f"wrote {args.output}")
    print(f"  {len({k[0] for k in agg})} models × "
          f"{len({k[1] for k in agg})} datasets × "
          f"{len({k[2] for k in agg})} attacks = "
          f"{len(agg)} (model, dataset, attack) cells "
          f"({sum(a['n_classifier_cells'] for a in agg.values())} classifier rows)")


if __name__ == "__main__":
    main()
