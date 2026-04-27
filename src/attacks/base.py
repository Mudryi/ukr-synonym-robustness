"""Common interface and result schema for both attacks.

Every attack implements :class:`BaseAttack` and returns an :class:`AttackResult`
from ``run()``. The runner / writer never care which attack produced the
result, so head-to-head comparison is just a matter of pointing the runner at
a different ``--attack`` flag.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from enum import Enum


class Status(str, Enum):
    SKIPPED_ORIG_WRONG = "SKIPPED_ORIG_WRONG"  # model already mis-classified — skip
    SUCCESS = "SUCCESS"                          # adv example flips the label
    FAILED = "FAILED"                             # exhausted candidates, label unchanged
    BUDGET_EXCEEDED = "BUDGET_EXCEEDED"          # max_changes_frac hit before flipping


@dataclass
class AttackResult:
    id: int
    attack: str
    orig_text: str
    adv_text: str
    true_label: int
    orig_label: int
    adv_label: int
    status: str
    num_changes: int
    num_queries: int
    change_rate: float
    replacements: list[dict] = field(default_factory=list)
    semantic_sim: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)


class BaseAttack(ABC):
    """All attacks expose ``name`` and ``run(text, label, idx) -> AttackResult``."""

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @abstractmethod
    def run(self, text: str, true_label: int, idx: int) -> AttackResult:
        ...
