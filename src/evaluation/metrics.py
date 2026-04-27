"""Aggregate per-sample :class:`AttackResult`s into a run-level summary."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable

from ..attacks.base import AttackResult, Status


@dataclass
class Summary:
    attack: str
    dataset: str
    target_model: str
    target_checkpoint: str
    config: dict
    n_total: int
    n_skipped_orig_wrong: int
    n_attacked: int
    n_success: int
    n_failed: int
    n_budget_exceeded: int
    original_accuracy: float
    after_attack_accuracy: float
    attack_success_rate: float
    avg_queries: float
    avg_change_rate: float
    avg_semantic_sim: float | None
    wall_time_sec: float

    def to_dict(self) -> dict:
        return asdict(self)


def aggregate(
    results: Iterable[AttackResult],
    *,
    attack: str,
    dataset: str,
    target_model: str,
    target_checkpoint: str,
    config: dict,
    wall_time_sec: float,
) -> Summary:
    results = list(results)
    n_total = len(results)
    n_skipped = sum(1 for r in results if r.status == Status.SKIPPED_ORIG_WRONG.value)
    n_success = sum(1 for r in results if r.status == Status.SUCCESS.value)
    n_failed = sum(1 for r in results if r.status == Status.FAILED.value)
    n_budget = sum(1 for r in results if r.status == Status.BUDGET_EXCEEDED.value)
    n_attacked = n_total - n_skipped

    # original accuracy = fraction predicted correctly on the input
    n_orig_correct = sum(1 for r in results if r.true_label == r.orig_label)
    original_acc = (n_orig_correct / n_total) if n_total else 0.0

    # after-attack accuracy = fraction still predicted correctly after attack
    n_after = sum(1 for r in results if r.true_label == r.adv_label)
    after_acc = (n_after / n_total) if n_total else 0.0

    asr = (n_success / n_attacked) if n_attacked else 0.0

    queries_pool = [r.num_queries for r in results if r.status != Status.SKIPPED_ORIG_WRONG.value]
    avg_q = (sum(queries_pool) / len(queries_pool)) if queries_pool else 0.0

    change_pool = [r.change_rate for r in results if r.status == Status.SUCCESS.value]
    avg_change = (sum(change_pool) / len(change_pool)) if change_pool else 0.0

    sim_pool = [r.semantic_sim for r in results if r.semantic_sim is not None]
    avg_sim = (sum(sim_pool) / len(sim_pool)) if sim_pool else None

    return Summary(
        attack=attack,
        dataset=dataset,
        target_model=target_model,
        target_checkpoint=target_checkpoint,
        config=config,
        n_total=n_total,
        n_skipped_orig_wrong=n_skipped,
        n_attacked=n_attacked,
        n_success=n_success,
        n_failed=n_failed,
        n_budget_exceeded=n_budget,
        original_accuracy=original_acc,
        after_attack_accuracy=after_acc,
        attack_success_rate=asr,
        avg_queries=avg_q,
        avg_change_rate=avg_change,
        avg_semantic_sim=avg_sim,
        wall_time_sec=wall_time_sec,
    )
