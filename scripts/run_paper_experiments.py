"""Reproduce the paper grid: 3 datasets × 3 models × 2 attacks = 18 runs.

Each run is a subprocess call to ``python -m src.cli.run_attack`` with the
right config + checkpoint + attack. Results land under
``results/<attack>/<dataset>__<model>/``. After the grid finishes, two
markdown summary tables are written (one per attack) into ``results/``.

Usage
-----

    # Full reproduction
    python scripts/run_paper_experiments.py

    # Smoke test (fast — 50 examples per cell)
    python scripts/run_paper_experiments.py --n-samples 50

    # Re-run subset
    python scripts/run_paper_experiments.py \\
        --datasets reviews unlp --attacks textfooler --models xlmr_base

    # Skip cells that already produced summary.json
    python scripts/run_paper_experiments.py --skip-existing

    # Just rebuild the tables from existing summary.json files
    python scripts/run_paper_experiments.py --tables-only
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CONFIGS = REPO / "configs"
RESULTS = REPO / "results"

# External resources (paths discovered on this machine).
SYN_DICT = "/home/mudryi/phd_projects/synonym_attack/synonyms_dictionaries/synonimy_info_clean.json"
HAND_PARSED = "/home/mudryi/phd_projects/textfooler_ukr/hand_parsed_top_100.json"
ANTONYMS = "/home/mudryi/phd_projects/synonym_attack/synonyms_dictionaries/antonimy.jsonlines"
FASTTEXT_PATH = REPO / "resources/fasttext_uk/cbow.uk.300.bin"

TM = "/home/mudryi/phd_projects/xml-roberta-finetune-reviews/trained_models"

# (model_key, HuggingFace tokenizer/base name, finetuned checkpoint dir)
# Triples mirror the per-dataset comments inside configs/*.yaml.
MODELS = {
    "reviews": [
        ("ukr_roberta", "youscan/ukr-roberta-base",
         f"{TM}/7ddc/model_7ddc_7_600"),
        ("sbert_mpnet", "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
         f"{TM}/7yuz/model_7yuz_4_1200"),
        ("xlmr_base", "xlm-roberta-base",
         f"{TM}/tmdk/model_tmdk_7_600"),
    ],
    "news": [
        ("ukr_roberta", "youscan/ukr-roberta-base",
         f"{TM}/npz4/model_npz4_9_1000"),
        ("sbert_mpnet", "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
         f"{TM}/1kjq/model_1kjq_9_2500"),
        ("xlmr_base", "xlm-roberta-base",
         f"{TM}/3rzr/model_3rzr_9_2500"),
    ],
    "unlp": [
        ("ukr_roberta", "youscan/ukr-roberta-base",
         f"{TM}/1ozc/model_1ozc_14"),
        ("sbert_mpnet", "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
         f"{TM}/zzl4/model_zzl4_14"),
        ("xlmr_base", "xlm-roberta-base",
         f"{TM}/p0g9/model_p0g9_14"),
    ],
}

DATASETS = ["reviews", "news", "unlp"]
ATTACKS = ["textfooler", "bert_attack"]

# Pretty column labels for the summary tables.
MODEL_LABEL = {
    "ukr_roberta": "ukr-roberta-base",
    "sbert_mpnet": "paraphrase-mpnet",
    "xlmr_base": "xlm-roberta-base",
}


def output_dir(attack: str, dataset: str, model_key: str) -> Path:
    return RESULTS / attack / f"{dataset}__{model_key}"


def build_cmd(*, attack: str, dataset: str, model_key: str, hf_name: str,
              ckpt: str, n_samples: int | None, extra: list[str]) -> list[str]:
    out = output_dir(attack, dataset, model_key)
    cmd = [
        sys.executable, "-m", "src.cli.run_attack",
        "--config", str(CONFIGS / f"{dataset}.yaml"),
        "--attack", attack,
        "--target-model", hf_name,
        "--target-checkpoint", ckpt,
        "--output-dir", str(out),
    ]
    if attack == "textfooler":
        cmd += [
            "--synonym-dict", SYN_DICT,
            "--hand-parsed", HAND_PARSED,
            "--antonyms", ANTONYMS,
        ]
    else:  # bert_attack
        cmd += ["--fasttext-path", str(FASTTEXT_PATH)]
    if n_samples is not None:
        cmd += ["--n-samples", str(n_samples)]
    cmd += extra
    return cmd


def run_one(cmd: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"$ {' '.join(shlex.quote(c) for c in cmd)}")
    print(f"  log -> {log_path}")
    started = time.time()
    with open(log_path, "w", encoding="utf-8") as logf:
        proc = subprocess.run(cmd, cwd=REPO, stdout=logf, stderr=subprocess.STDOUT)
    elapsed = time.time() - started
    print(f"  exit={proc.returncode} ({elapsed/60:.1f} min)")
    return proc.returncode


# ---------- summary tables ----------

def _fmt(value, *, pct: bool = False, places: int = 3) -> str:
    if value is None:
        return "—"
    if pct:
        return f"{value * 100:.1f}"
    return f"{value:.{places}f}"


def build_table(attack: str) -> str:
    """Return a markdown table for one attack. Rows = datasets, cols = models.
    Each cell stacks: ASR / orig_acc → adv_acc / queries / change_rate."""
    header = (
        f"## Summary — {attack}\n\n"
        f"Cell format: **ASR%** · orig_acc → adv_acc · avg_queries · "
        f"avg_change_rate · avg_sem_sim\n\n"
    )
    cols = ["dataset"] + [MODEL_LABEL[m] for m in ("ukr_roberta", "sbert_mpnet", "xlmr_base")]
    sep = "|" + "|".join(["---"] * len(cols)) + "|"
    lines = ["| " + " | ".join(cols) + " |", sep]

    for dataset in DATASETS:
        row = [dataset]
        for model_key, _, _ in MODELS[dataset]:
            sp = output_dir(attack, dataset, model_key) / "summary.json"
            if not sp.exists():
                row.append("missing")
                continue
            s = json.loads(sp.read_text(encoding="utf-8"))
            cell = (
                f"**{_fmt(s.get('attack_success_rate'), pct=True)}%** · "
                f"{_fmt(s.get('original_accuracy'))} → {_fmt(s.get('after_attack_accuracy'))} · "
                f"q={_fmt(s.get('avg_queries'), places=1)} · "
                f"Δ={_fmt(s.get('avg_change_rate'))} · "
                f"sim={_fmt(s.get('avg_semantic_sim'))}"
            )
            row.append(cell)
        lines.append("| " + " | ".join(row) + " |")
    return header + "\n".join(lines) + "\n"


def write_tables() -> Path:
    RESULTS.mkdir(parents=True, exist_ok=True)
    md_path = RESULTS / "summary_tables.md"
    parts = ["# Paper-grid attack results\n"]
    for attack in ATTACKS:
        parts.append(build_table(attack))
        parts.append("")
    md_path.write_text("\n".join(parts), encoding="utf-8")
    print(f"\nwrote tables -> {md_path}")
    print("=" * 72)
    print(md_path.read_text(encoding="utf-8"))
    return md_path


# ---------- preflight ----------

def preflight(args) -> list[str]:
    problems: list[str] = []
    for path, label in [
        (SYN_DICT, "synonym dict"),
        (HAND_PARSED, "hand-parsed synonyms"),
        (ANTONYMS, "antonyms"),
        (FASTTEXT_PATH, "fastText UA"),
    ]:
        if not Path(path).exists():
            problems.append(f"missing {label}: {path}")
    for ds in args.datasets:
        cfg = CONFIGS / f"{ds}.yaml"
        if not cfg.exists():
            problems.append(f"missing config: {cfg}")
        for mk, _, ckpt in MODELS[ds]:
            if mk not in args.models:
                continue
            if not Path(ckpt).exists():
                problems.append(f"missing checkpoint for {ds}/{mk}: {ckpt}")
    return problems


# ---------- main ----------

def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--datasets", nargs="+", choices=DATASETS, default=DATASETS)
    p.add_argument("--models", nargs="+",
                   choices=list(MODEL_LABEL.keys()), default=list(MODEL_LABEL.keys()))
    p.add_argument("--attacks", nargs="+", choices=ATTACKS, default=ATTACKS)
    p.add_argument("--n-samples", type=int, default=None,
                   help="If set, attack only this many examples per cell (smoke test).")
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip cells whose summary.json already exists.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print commands without executing.")
    p.add_argument("--tables-only", action="store_true",
                   help="Don't run anything; just rebuild summary_tables.md from existing runs.")
    p.add_argument("--continue-on-error", action="store_true",
                   help="Keep going even if a cell fails (default: stop on first failure).")
    p.add_argument("--extra", nargs=argparse.REMAINDER, default=[],
                   help="Extra flags forwarded verbatim to run_attack (after `--`).")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    if args.tables_only:
        write_tables()
        return

    problems = preflight(args)
    if problems:
        print("preflight FAILED:")
        for p_ in problems:
            print(f"  - {p_}")
        sys.exit(1)

    cells = [
        (attack, ds, mk, hf, ckpt)
        for attack in args.attacks
        for ds in args.datasets
        for (mk, hf, ckpt) in MODELS[ds]
        if mk in args.models
    ]
    print(f"planning {len(cells)} runs "
          f"(attacks={args.attacks}, datasets={args.datasets}, models={args.models})")

    failures: list[tuple] = []
    grid_started = time.time()
    for i, (attack, ds, mk, hf, ckpt) in enumerate(cells, 1):
        out = output_dir(attack, ds, mk)
        summary_path = out / "summary.json"
        log_path = out / "run.log"
        print(f"\n[{i}/{len(cells)}] attack={attack} dataset={ds} model={mk}")
        if args.skip_existing and summary_path.exists():
            print(f"  skip (summary.json exists at {summary_path})")
            continue
        cmd = build_cmd(
            attack=attack, dataset=ds, model_key=mk, hf_name=hf, ckpt=ckpt,
            n_samples=args.n_samples, extra=args.extra,
        )
        if args.dry_run:
            print("  DRY-RUN: " + " ".join(shlex.quote(c) for c in cmd))
            continue
        rc = run_one(cmd, log_path)
        if rc != 0:
            failures.append((attack, ds, mk, rc, log_path))
            if not args.continue_on_error:
                print("aborting on first failure (use --continue-on-error to keep going)")
                break

    grid_elapsed = time.time() - grid_started
    print(f"\ngrid wall time: {grid_elapsed/60:.1f} min")
    if failures:
        print(f"\n{len(failures)} cells FAILED:")
        for attack, ds, mk, rc, log_path in failures:
            print(f"  - {attack}/{ds}/{mk} (exit={rc}, see {log_path})")

    if not args.dry_run:
        write_tables()

    if failures:
        sys.exit(2)


if __name__ == "__main__":
    main()
