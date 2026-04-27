"""POS tagging and POS-compatibility filter used by TextFooler."""

from __future__ import annotations

from .morphology import (
    INTERCHANGEABLE_POS,
    PYMORPHY_TO_UNIVERSAL,
    get_pos_safe,
    morph,
)


def get_pos(sent: list[str], tagset: str = "universal") -> list[str]:
    pos_list: list[str] = []
    for word in sent:
        parsed = morph.parse(word)[0]
        pymorphy_pos = get_pos_safe(parsed)
        if tagset == "default":
            pos_list.append(pymorphy_pos)
        else:
            pos_list.append(PYMORPHY_TO_UNIVERSAL.get(pymorphy_pos, "X"))
    return pos_list


def pos_filter(ori_pos: str, new_pos_list: list[str]) -> list[bool]:
    """True for substitutes whose POS is identical or interchangeable with the original."""
    results: list[bool] = []
    for new_pos in new_pos_list:
        if ori_pos == new_pos:
            results.append(True)
        elif new_pos in INTERCHANGEABLE_POS.get(ori_pos, set()):
            results.append(True)
        else:
            results.append(False)
    return results
