"""CLI: run the LLM evaluator on a single cell or sweep a grid of cells.

Single cell:
    python -m llm_eval.cli \\
        --attack-dir results/textfooler/reviews__ukr_roberta \\
        --dataset reviews \\
        --llm-config llm_eval/configs/mamay.yaml \\
        --output-dir results/llm/mamay__textfooler__reviews__ukr_roberta \\
        [--limit 50] [--seed 1914]

Grid (sweeps results/textfooler and results/bert_attack):
    python -m llm_eval.cli --grid \\
        --llm-config llm_eval/configs/mamay.yaml \\
        --results-root results \\
        [--limit 50]
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch

from .client import MamayClient, load_config
from .evaluator import evaluate_run


VALID_DATASETS = {"reviews", "news", "unlp"}


def _infer_dataset(attack_dir: Path) -> str | None:
    name = attack_dir.name
    head = name.split("__", 1)[0] if "__" in name else name
    return head if head in VALID_DATASETS else None


def _llm_short_name(model_id: str) -> str:
    """Compact slug for output dir naming. e.g. INSAIT-.../MamayLM-... → mamay."""
    tail = model_id.split("/")[-1].lower()
    if "mamay" in tail:
        return "mamay"
    return tail.replace(".", "_")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="LLM evaluator on existing attack outputs.")
    p.add_argument("--llm-config", type=str, default=None,
                   help="Path to YAML with model_id, backend, decoding params.")
    p.add_argument("--seed", type=int, default=1914)
    p.add_argument("--limit", type=int, default=None,
                   help="Cap examples per cell — useful for smoke tests.")
    p.add_argument("--no-resume", action="store_true",
                   help="Overwrite predictions.jsonl instead of appending.")

    # Single cell
    p.add_argument("--attack-dir", type=str, default=None,
                   help="Path to results/<attack>/<dataset>__<classifier>/.")
    p.add_argument("--dataset", choices=sorted(VALID_DATASETS), default=None,
                   help="Dataset (auto-inferred from --attack-dir name if omitted).")
    p.add_argument("--output-dir", type=str, default=None)

    # Grid
    p.add_argument("--grid", action="store_true",
                   help="Sweep over <results-root>/{textfooler,bert_attack}/*/")
    p.add_argument("--results-root", type=str, default="results",
                   help="Base directory containing attack output folders.")

    return p.parse_args(argv)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _run_single(client: MamayClient, attack_dir: Path, dataset: str,
                output_dir: Path, limit: int | None, resume: bool):
    print(f"\n=== {attack_dir} → {output_dir} (dataset={dataset}) ===")
    summary = evaluate_run(
        attack_dir=attack_dir,
        dataset=dataset,
        client=client,
        output_dir=output_dir,
        limit=limit,
        resume=resume,
    )
    print(
        f"clean_acc={summary.llm_clean_acc:.3f} "
        f"adv_acc={summary.llm_adv_acc:.3f} "
        f"gap={summary.robustness_gap:.3f} "
        f"consistency={summary.llm_consistency:.3f} "
        f"transfer_asr={summary.llm_transfer_asr:.3f} "
        f"(n_eligible={summary.n_transfer_eligible}, "
        f"parse_fail orig/adv={summary.n_parse_failed_orig}/{summary.n_parse_failed_adv})"
    )


def main(argv=None):
    args = parse_args(argv)
    _seed_everything(args.seed)

    cfg = load_config(args.llm_config)
    cfg.seed = args.seed
    client = MamayClient(cfg)
    llm_slug = _llm_short_name(cfg.model_id)

    if args.grid:
        root = Path(args.results_root)
        cells = []
        for attack in ("textfooler", "bert_attack"):
            attack_root = root / attack
            if not attack_root.exists():
                continue
            for cell in sorted(attack_root.iterdir()):
                if not cell.is_dir() or not (cell / "examples.jsonl").exists():
                    continue
                ds = _infer_dataset(cell)
                if ds is None:
                    print(f"[skip] cannot infer dataset from {cell}")
                    continue
                out = root / "llm" / f"{llm_slug}__{attack}__{cell.name}"
                cells.append((cell, ds, out))
        if not cells:
            sys.exit(f"no attack cells found under {root}")
        for cell, ds, out in cells:
            _run_single(client, cell, ds, out, args.limit, resume=not args.no_resume)
        return

    # single-cell mode
    if not args.attack_dir or not args.output_dir:
        sys.exit("--attack-dir and --output-dir are required (or use --grid).")
    attack_dir = Path(args.attack_dir)
    dataset = args.dataset or _infer_dataset(attack_dir)
    if dataset is None:
        sys.exit(f"could not infer --dataset from {attack_dir}; pass it explicitly.")
    _run_single(
        client, attack_dir, dataset, Path(args.output_dir),
        args.limit, resume=not args.no_resume,
    )


if __name__ == "__main__":
    main()
