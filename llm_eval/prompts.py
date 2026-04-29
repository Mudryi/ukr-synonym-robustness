"""Per-dataset zero-shot prompt templates (Ukrainian).

Each dataset exposes:
- ``SYSTEM`` and ``USER_TEMPLATE`` (str) — applied to the LLM via
  ``tokenizer.apply_chat_template`` with ``[{role:system,...},{role:user,...}]``.
- ``LABEL_VOCAB`` (dict[str, int]) — surface forms the LLM is expected to
  emit, mapped to integer labels matching ``src/core/data.py``.
- ``LABEL_NAMES`` (list[str]) — index → human-readable label.

The system+user split keeps the task framing isolated from the input text
so attacks that try to escape the prompt (prompt injection via the review
body) stay inside the user turn.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

# Reuse the canonical news label vocabulary from the main package.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from src.core.data import NEWS_LABELS  # noqa: E402


# --- Reviews: 5-class sentiment, integer label 0..4 == stars 1..5 -----------

REVIEWS_SYSTEM = (
    "Ви — класифікатор тональності відгуків покупців на українські товари. "
    "Прочитайте відгук і визначте тональність за шкалою від 1 до 5, де "
    "1 — дуже негативний, 2 — негативний, 3 — нейтральний, "
    "4 — позитивний, 5 — дуже позитивний. "
    "Відповідайте ЛИШЕ однією цифрою від 1 до 5 без пояснень."
)
REVIEWS_USER_TEMPLATE = "Відгук:\n{text}\n\nОцінка:"
REVIEWS_LABEL_VOCAB = {"1": 0, "2": 1, "3": 2, "4": 3, "5": 4}
REVIEWS_LABEL_NAMES = ["1 зірка", "2 зірки", "3 зірки", "4 зірки", "5 зірок"]


# --- News: 5-class topic classification -------------------------------------

NEWS_SYSTEM = (
    "Ви — класифікатор новин українською мовою. Прочитайте заголовок новини "
    "і віднесіть його до однієї з категорій: бізнес, новини, політика, спорт, "
    "технології. Відповідайте ЛИШЕ назвою категорії одним словом."
)
NEWS_USER_TEMPLATE = "Заголовок:\n{text}\n\nКатегорія:"
NEWS_LABEL_VOCAB = {label: i for i, label in enumerate(NEWS_LABELS)}
NEWS_LABEL_NAMES = list(NEWS_LABELS)


# --- UNLP: binary manipulation detection ------------------------------------

UNLP_SYSTEM = (
    "Ви — детектор маніпулятивного контенту в українських текстах. "
    "Прочитайте текст і визначте, чи містить він ознаки маніпуляції "
    "(упередженість, маніпулятивні мовні засоби, дезінформацію). "
    "Відповідайте ЛИШЕ: Так (якщо є маніпуляція) або Ні (якщо немає)."
)
UNLP_USER_TEMPLATE = "Текст:\n{text}\n\nВідповідь:"
UNLP_LABEL_VOCAB = {"ні": 0, "так": 1}
UNLP_LABEL_NAMES = ["без маніпуляції", "маніпуляція"]


# --- Registry ---------------------------------------------------------------

PROMPTS = {
    "reviews": {
        "system": REVIEWS_SYSTEM,
        "user_template": REVIEWS_USER_TEMPLATE,
        "label_vocab": REVIEWS_LABEL_VOCAB,
        "label_names": REVIEWS_LABEL_NAMES,
    },
    "news": {
        "system": NEWS_SYSTEM,
        "user_template": NEWS_USER_TEMPLATE,
        "label_vocab": NEWS_LABEL_VOCAB,
        "label_names": NEWS_LABEL_NAMES,
    },
    "unlp": {
        "system": UNLP_SYSTEM,
        "user_template": UNLP_USER_TEMPLATE,
        "label_vocab": UNLP_LABEL_VOCAB,
        "label_names": UNLP_LABEL_NAMES,
    },
}


def get_prompt(dataset: str) -> dict:
    if dataset not in PROMPTS:
        raise ValueError(
            f"unknown dataset {dataset!r}; supported: {list(PROMPTS)}"
        )
    return PROMPTS[dataset]


def build_messages(dataset: str, text: str) -> list[dict]:
    p = get_prompt(dataset)
    return [
        {"role": "system", "content": p["system"]},
        {"role": "user", "content": p["user_template"].format(text=text)},
    ]


def prompt_hash(dataset: str) -> str:
    """Stable short hash of the prompt — pinned in summary.json so we can
    detect if a later run silently changed prompt wording."""
    p = get_prompt(dataset)
    payload = json.dumps(
        {"system": p["system"], "user_template": p["user_template"]},
        ensure_ascii=False, sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:12]
