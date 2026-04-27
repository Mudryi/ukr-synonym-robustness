"""Load and clean the Ukrainian synonym dictionary used by TextFooler.

The original TextFooler code lived in ``prepare_synonym_dict.py``. This version:

* fixes the bug where ``return synonym_dict`` was indented inside the
  cleanup ``for`` loop, returning after one iteration;
* takes paths as arguments (no hardcoded ``/home/mudryi/...`` paths);
* tolerates missing optional inputs (hand-parsed, antonyms) gracefully.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

from .morphology import morph
from ._synonym_dict_constants import deletions, remove_list


def read_and_process_synonym_dict(path_to_dict: str | Path) -> dict[str, list[str]]:
    with open(path_to_dict, "r", encoding="utf-8") as f:
        raw = json.load(f)

    out: dict[str, list[str]] = {}
    for entry in tqdm(raw, desc="Processing synonym dict"):
        if len(entry["synsets"]) == 0:
            continue
        lemma = entry["lemma"].lower()
        out.setdefault(lemma, [])
        for synset in entry["synsets"]:
            normal = {p.normal_form for w in synset["clean"] for p in morph.parse(w)}
            out[lemma].extend(normal)
        out[lemma] = list(set(out[lemma]) - {lemma})
    return out


def remove_synonyms(synonym_dict: dict[str, list[str]], to_delete: dict[str, list[str]]) -> None:
    """In-place: drop hand-curated bad pairs."""
    for key, values in to_delete.items():
        if key in synonym_dict:
            for v in values:
                if v in synonym_dict[key]:
                    synonym_dict[key].remove(v)


def add_synonyms_from_antonyms_info(
    synonym_dict: dict[str, list[str]], antonyms: dict[str, dict]
) -> dict[str, list[str]]:
    for lemma, entry in antonyms.items():
        if lemma in synonym_dict:
            synonym_dict[lemma] = list(set(synonym_dict[lemma] + entry["synonyms"]))
        else:
            synonym_dict[lemma] = entry["synonyms"]
    return synonym_dict


def remove_antonyms_from_synonym_dict(
    synonym_dict: dict[str, list[str]], antonyms: dict[str, dict]
) -> dict[str, list[str]]:
    for lemma, entry in antonyms.items():
        if lemma in synonym_dict:
            synonym_dict[lemma] = [w for w in synonym_dict[lemma] if w not in entry["antonyms"]]
    return synonym_dict


def load_entries_by_lemma(path: str | Path) -> dict[str, dict]:
    """Read a JSON-lines antonym file. Tries UTF-8 (with BOM) then CP1251."""
    for enc in ("utf-8-sig", "cp1251"):
        try:
            with open(path, "r", encoding=enc, errors="strict") as f:
                data: dict[str, dict] = {}
                for lineno, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        fixed = line.encode("utf-8").decode("unicode_escape")
                        entry = json.loads(fixed)
                    lemma = entry.get("lemma")
                    if not lemma:
                        raise ValueError(f"line {lineno}: missing 'lemma' field")
                    lemma = lemma.lower()
                    data[lemma] = {
                        "url": entry.get("url"),
                        "synonyms": [s.lower() for s in entry.get("synonyms", [])],
                        "antonyms": [s.lower() for s in entry.get("antonyms", [])],
                        "samples": entry.get("samples", []),
                    }
                return data
        except (UnicodeDecodeError, ValueError) as e:
            print(f"failed to load {path} with encoding {enc!r}: {e}")
    raise UnicodeError(f"could not decode {path!r} with utf-8-sig or cp1251")


def load_dict_from_json(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def merge_synonym_dicts(
    d1: dict[str, list[str]],
    d2: dict[str, list[str]],
    *,
    dedupe: bool = True,
) -> dict[str, list[str]]:
    merged: dict[str, list[str]] = defaultdict(list)
    for word, syns in d1.items():
        merged[word].extend(syns)
    for word, syns in d2.items():
        merged[word].extend(syns)
    if dedupe:
        for word, syns in merged.items():
            seen: set[str] = set()
            unique: list[str] = []
            for s in syns:
                if s not in seen:
                    seen.add(s)
                    unique.append(s)
            merged[word] = unique
    return dict(merged)


def read_and_clean_synonym_dict(
    path_to_dict: str | Path,
    *,
    hand_parsed_path: str | Path | None = None,
    antonyms_path: str | Path | None = None,
) -> dict[str, list[str]]:
    """Build the merged, cleaned synonym dictionary.

    Args:
        path_to_dict:    Required. Main synonym JSON file.
        hand_parsed_path: Optional. Hand-curated additions JSON (word -> [synonyms]).
        antonyms_path:    Optional. Antonym JSONLines (one entry per line).
    """
    synonym_dict = read_and_process_synonym_dict(path_to_dict)

    if hand_parsed_path:
        hand = load_dict_from_json(hand_parsed_path)
        synonym_dict = merge_synonym_dicts(synonym_dict, hand)

    if antonyms_path:
        antonyms = load_entries_by_lemma(antonyms_path)
        synonym_dict = add_synonyms_from_antonyms_info(synonym_dict, antonyms)
        synonym_dict = remove_antonyms_from_synonym_dict(synonym_dict, antonyms)

    remove_synonyms(synonym_dict, deletions)

    # Bug-fix: in the original code, `return synonym_dict` was indented inside
    # this loop and short-circuited after one key. The cleanup must run over
    # every entry before returning.
    for key in synonym_dict:
        synonym_dict[key] = [w for w in synonym_dict[key] if w not in remove_list]
    return synonym_dict
