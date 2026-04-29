"""Per-cell evaluator: read an attack run's examples.jsonl, query the LLM
on each ``orig_text`` and ``adv_text``, write predictions.jsonl + summary.json.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Iterator, Optional

from tqdm import tqdm

from .client import MamayClient
from .metrics import LLMSummary, aggregate, append_runs_csv, write_summary
from .parser import parse_label
from .prompts import build_messages, prompt_hash


def _iter_examples(path: Path) -> Iterator[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _seen_ids(predictions_path: Path) -> set[int]:
    if not predictions_path.exists():
        return set()
    ids: set[int] = set()
    with open(predictions_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ids.add(json.loads(line)["id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return ids


def evaluate_run(
    *,
    attack_dir: Path,
    dataset: str,
    client: MamayClient,
    output_dir: Path,
    limit: Optional[int] = None,
    resume: bool = True,
) -> LLMSummary:
    attack_dir = Path(attack_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    examples_path = attack_dir / "examples.jsonl"
    if not examples_path.exists():
        raise FileNotFoundError(f"no examples.jsonl in {attack_dir}")

    predictions_path = output_dir / "predictions.jsonl"
    seen = _seen_ids(predictions_path) if resume else set()
    write_mode = "a" if seen else "w"

    rows: list[dict] = []
    if resume and seen:
        # Re-load existing rows so we can aggregate over them too.
        with open(predictions_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        print(f"[eval] resuming: {len(rows)} predictions already on disk")

    started = time.time()
    n_processed = 0
    with open(predictions_path, write_mode, encoding="utf-8") as out:
        all_examples = list(_iter_examples(examples_path))
        if limit is not None:
            all_examples = all_examples[:limit]
        for ex in tqdm(all_examples, desc=f"llm-eval {attack_dir.name}"):
            ex_id = ex["id"]
            if ex_id in seen:
                continue

            orig_msgs = build_messages(dataset, ex["orig_text"])
            adv_msgs = build_messages(dataset, ex["adv_text"])

            orig_raw = client.generate(orig_msgs)
            adv_raw = client.generate(adv_msgs)

            orig_lbl = parse_label(orig_raw, dataset)
            adv_lbl = parse_label(adv_raw, dataset)

            row = {
                "id": ex_id,
                "true_label": ex["true_label"],
                "classifier_orig_label": ex["orig_label"],
                "classifier_adv_label": ex["adv_label"],
                "classifier_status": ex["status"],
                "llm_orig_raw": orig_raw,
                "llm_adv_raw": adv_raw,
                "llm_orig_label": orig_lbl,
                "llm_adv_label": adv_lbl,
                "llm_orig_parsed": orig_lbl is not None,
                "llm_adv_parsed": adv_lbl is not None,
            }
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()
            rows.append(row)
            n_processed += 1

    wall = time.time() - started
    print(f"[eval] processed {n_processed} new examples in {wall:.1f}s")

    classifier, attack_name = _parse_attack_dir(attack_dir)
    summary = aggregate(
        rows,
        llm_model=client.cfg.model_id,
        dataset=dataset,
        attack=attack_name,
        classifier=classifier,
        attack_dir=str(attack_dir),
        prompt_hash=prompt_hash(dataset),
        backend=client.backend,
        config={
            "model_id": client.cfg.model_id,
            "backend": client.backend,
            "max_new_tokens": client.cfg.max_new_tokens,
            "do_sample": client.cfg.do_sample,
            "temperature": client.cfg.temperature,
            "top_p": client.cfg.top_p,
            "seed": client.cfg.seed,
            "limit": limit,
            "dataset": dataset,
            "attack_dir": str(attack_dir),
        },
        wall_time_sec=wall,
    )
    write_summary(summary, output_dir)
    append_runs_csv(summary, output_dir.parent / "runs.csv", output_dir)
    return summary


def _parse_attack_dir(attack_dir: Path) -> tuple[str | None, str | None]:
    """Infer (classifier, attack) from the canonical layout
    ``results/<attack>/<dataset>__<classifier>``."""
    name = attack_dir.name  # "<dataset>__<classifier>"
    parent = attack_dir.parent.name  # "<attack>"
    classifier = name.split("__", 1)[1] if "__" in name else None
    return classifier, parent
