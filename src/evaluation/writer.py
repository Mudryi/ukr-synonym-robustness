"""Stream per-sample results to JSONL and finalise summary.json + runs.csv."""

from __future__ import annotations

import csv
import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

from ..attacks.base import AttackResult
from .metrics import Summary, aggregate


class ResultWriter:
    """Streams ``AttackResult``s to JSONL; finalises ``summary.json`` and
    appends one row to a top-level ``runs.csv`` for cross-run comparison."""

    def __init__(self, output_dir: str | Path, runs_csv: str | Path | None = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.examples_path = self.output_dir / "examples.jsonl"
        self.summary_path = self.output_dir / "summary.json"
        # default the cross-run CSV to the parent of this run's dir
        self.runs_csv = (
            Path(runs_csv) if runs_csv is not None else self.output_dir.parent / "runs.csv"
        )
        self._fh = None
        self._results: list[AttackResult] = []

    # context manager for guaranteed flush

    def __enter__(self):
        self._fh = open(self.examples_path, "w", encoding="utf-8")
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._fh:
            self._fh.flush()
            self._fh.close()
            self._fh = None

    # ----------

    def write(self, result: AttackResult) -> None:
        if self._fh is None:
            raise RuntimeError("ResultWriter must be used as a context manager")
        self._results.append(result)
        self._fh.write(json.dumps(asdict(result), ensure_ascii=False) + "\n")
        self._fh.flush()

    def finalize(
        self,
        *,
        attack: str,
        dataset: str,
        target_model: str,
        target_checkpoint: str,
        config: dict,
        wall_time_sec: float,
    ) -> Summary:
        summary = aggregate(
            self._results,
            attack=attack,
            dataset=dataset,
            target_model=target_model,
            target_checkpoint=target_checkpoint,
            config=config,
            wall_time_sec=wall_time_sec,
        )
        with open(self.summary_path, "w", encoding="utf-8") as f:
            json.dump(summary.to_dict(), f, ensure_ascii=False, indent=2)
        self._append_runs_csv(summary)
        return summary

    def _append_runs_csv(self, summary: Summary) -> None:
        row = summary.to_dict()
        # flatten config out of the row (keep only summary scalars)
        row.pop("config", None)
        # add output dir for traceability
        row["output_dir"] = str(self.output_dir)
        self.runs_csv.parent.mkdir(parents=True, exist_ok=True)

        new_file = not self.runs_csv.exists() or self.runs_csv.stat().st_size == 0
        with open(self.runs_csv, "a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if new_file:
                writer.writeheader()
            writer.writerow(row)


def append_jsonl(path: str | Path, records: Iterable[dict]) -> None:
    """Utility: append already-serialisable dicts to a JSONL file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
