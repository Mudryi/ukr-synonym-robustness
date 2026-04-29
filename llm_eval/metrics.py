"""Aggregate per-example LLM predictions into a run-level summary."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable


@dataclass
class LLMSummary:
    llm_model: str
    dataset: str
    attack: str | None
    classifier: str | None
    attack_dir: str
    prompt_hash: str
    backend: str
    config: dict = field(default_factory=dict)

    n_total: int = 0
    n_parse_failed_orig: int = 0
    n_parse_failed_adv: int = 0
    n_classifier_success: int = 0          # rows where attack flipped the classifier

    llm_clean_acc: float = 0.0             # over parseable orig
    llm_adv_acc: float = 0.0               # over parseable adv
    robustness_gap: float = 0.0            # clean − adv
    llm_consistency: float = 0.0           # both parsed and labels match
    llm_transfer_asr: float = 0.0          # see _compute_transfer_asr
    n_transfer_eligible: int = 0           # denominator for transfer ASR

    wall_time_sec: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


def _safe_div(num: int, den: int) -> float:
    return (num / den) if den else 0.0


def aggregate(rows: Iterable[dict], **meta) -> LLMSummary:
    rows = list(rows)
    n_total = len(rows)
    n_pf_orig = sum(1 for r in rows if not r["llm_orig_parsed"])
    n_pf_adv = sum(1 for r in rows if not r["llm_adv_parsed"])

    parsed_orig = [r for r in rows if r["llm_orig_parsed"]]
    parsed_adv = [r for r in rows if r["llm_adv_parsed"]]
    parsed_both = [r for r in rows if r["llm_orig_parsed"] and r["llm_adv_parsed"]]

    n_clean_correct = sum(1 for r in parsed_orig if r["llm_orig_label"] == r["true_label"])
    n_adv_correct = sum(1 for r in parsed_adv if r["llm_adv_label"] == r["true_label"])

    clean_acc = _safe_div(n_clean_correct, len(parsed_orig))
    adv_acc = _safe_div(n_adv_correct, len(parsed_adv))
    consistency = _safe_div(
        sum(1 for r in parsed_both if r["llm_orig_label"] == r["llm_adv_label"]),
        len(parsed_both),
    )

    # Transfer ASR: among rows where the classifier was successfully attacked
    # AND the LLM was originally correct on orig, how often does the LLM also
    # become wrong on adv? This isolates whether the same perturbation transfers.
    eligible = [
        r for r in rows
        if r["classifier_status"] == "SUCCESS"
        and r["llm_orig_parsed"]
        and r["llm_adv_parsed"]
        and r["llm_orig_label"] == r["true_label"]
    ]
    transferred = sum(1 for r in eligible if r["llm_adv_label"] != r["true_label"])
    transfer_asr = _safe_div(transferred, len(eligible))

    n_classifier_success = sum(1 for r in rows if r["classifier_status"] == "SUCCESS")

    return LLMSummary(
        n_total=n_total,
        n_parse_failed_orig=n_pf_orig,
        n_parse_failed_adv=n_pf_adv,
        n_classifier_success=n_classifier_success,
        llm_clean_acc=clean_acc,
        llm_adv_acc=adv_acc,
        robustness_gap=clean_acc - adv_acc,
        llm_consistency=consistency,
        llm_transfer_asr=transfer_asr,
        n_transfer_eligible=len(eligible),
        **meta,
    )


def write_summary(summary: LLMSummary, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "summary.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary.to_dict(), f, ensure_ascii=False, indent=2)
    return path


def append_runs_csv(summary: LLMSummary, runs_csv: Path, output_dir: Path) -> None:
    runs_csv.parent.mkdir(parents=True, exist_ok=True)
    row = summary.to_dict()
    row.pop("config", None)
    row["output_dir"] = str(output_dir)
    new_file = not runs_csv.exists() or runs_csv.stat().st_size == 0
    with open(runs_csv, "a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new_file:
            w.writeheader()
        w.writerow(row)
