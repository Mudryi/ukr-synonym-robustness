"""Pymorphy2-based Ukrainian morphology helpers shared by both attacks.

The non-trivial piece is :func:`replace_word`, which inflects a candidate
substitute into the same grammatical form as the target word. It is kept
verbatim from the original TextFooler code path because empirical attack
quality depends on its handling of edge cases.
"""

from __future__ import annotations

import re

import pymorphy2

from .tokenization import tokenize_ukrainian


morph = pymorphy2.MorphAnalyzer(lang="uk")


# ---------- POS tagsets ----------

PYMORPHY_TO_UNIVERSAL = {
    "NOUN": "NOUN",
    "ADJF": "ADJ",
    "ADJS": "ADJ",
    "COMP": "ADJ",
    "VERB": "VERB",
    "INFN": "VERB",
    "PRTF": "VERB",
    "PRTS": "VERB",
    "GRND": "VERB",
    "ADVB": "ADV",
    "NUMR": "NUM",
    "NPRO": "PRON",
    "PREP": "ADP",
    "CONJ": "CONJ",
    "PRCL": "PRT",
    "INTJ": "X",
    None: "X",
}

INTERCHANGEABLE_POS = {
    "PRCL": {"ADVB", "CONJ"},
    "ADVB": {"PRCL", "Prnt", "PRED", "GRND"},
    "Prnt": {"ADVB"},
    "CONJ": {"PRCL"},
    "ADJF": {"NPRO"},
    "NPRO": {"ADJF"},
    "PRED": {"ADVB"},
    "GRND": {"ADVB"},
}

FORCE_ADVB = {"завжди", "завше", "навіщо", "загалом"}


# ---------- Lookups ----------

def get_pos_safe(parsing_result) -> str:
    if parsing_result.word in FORCE_ADVB:
        return "ADVB"
    return parsing_result.tag.POS or str(parsing_result.tag).split(",")[0]


def get_normal_form(word: str) -> str:
    parsed = morph.parse(word)
    if parsed:
        return parsed[0].normal_form
    return word


def compare_normal_forms(word: str, substitution: str) -> bool:
    return morph.parse(word)[0].normal_form == morph.parse(substitution)[0].normal_form


# ---------- Inflection-aware replacement ----------

def get_correct_parsed_result(parsing_results, target_pos, target_gender=None):
    acceptable_pos = {target_pos} | INTERCHANGEABLE_POS.get(target_pos, set())

    for result in parsing_results:
        if (
            get_pos_safe(result) in acceptable_pos
            and target_gender
            and result.tag.gender == target_gender
        ):
            return result

    for result in parsing_results:
        if get_pos_safe(result) in acceptable_pos:
            return result

    if target_pos == "ADVB":
        for result in parsing_results:
            if result.tag.POS == "ADJF" and {"neut", "nomn"}.issubset(result.tag.grammemes):
                return result

    if target_pos == "ADJF":
        for result in parsing_results:
            if result.tag.POS == "ADVB":
                return result

    for result in parsing_results:
        if result.word.lower() in FORCE_ADVB and target_pos == "ADVB":
            return result

    return None


def lower_grammar_restrictions(grammemes):
    grammemes_to_remove = ("Refl", "compb", "COMP", "Qual")
    return {g for g in grammemes if g not in grammemes_to_remove}


def stepwise_inflect(
    parse,
    target_grammemes,
    preferred_order=(
        "plur", "sing", "femn", "masc", "neut",
        "nomn", "accs", "gent", "datv", "loct", "ablt",
        "anim", "inan",
    ),
):
    current = parse
    applied: set[str] = set()

    sorted_grammemes = sorted(
        target_grammemes,
        key=lambda g: preferred_order.index(g) if g in preferred_order else len(preferred_order),
    )

    for gram in sorted_grammemes:
        attempt = current.inflect(applied | {gram})
        if attempt is not None:
            current = attempt
            applied |= {gram}
    return current


def replace_word(sentence: str, target: str, replacement: str, *, debug: bool = False):
    """Inflect ``replacement`` to match the grammatical form of ``target`` in
    ``sentence`` and return the new token list. ``None`` if no replacement was
    possible (e.g. POS mismatch, inflection fails)."""
    target_normal_forms = [p.normal_form for p in morph.parse(target)]
    tokens = tokenize_ukrainian(sentence)

    replaced = False
    new_tokens: list[str] = []

    for token in tokens:
        if re.match(r"\w+", token):
            parse_options = morph.parse(token)
            replaced_curr = False

            for parsed_word in parse_options:
                if parsed_word.normal_form not in target_normal_forms:
                    continue

                target_pos = get_pos_safe(parsed_word)
                target_gender = (
                    parsed_word.tag.gender if target_pos in ("NOUN", "ADJF") else None
                )

                replacement_parses = morph.parse(replacement)
                matched = get_correct_parsed_result(
                    replacement_parses, target_pos, target_gender
                )
                if not matched:
                    if debug:
                        print(f"bad match {target} -> {replacement}")
                    continue

                grammemes = parsed_word.tag.grammemes
                cleaned = lower_grammar_restrictions(grammemes)
                inflected = stepwise_inflect(matched, cleaned)
                if not inflected:
                    if debug:
                        print(f"bad inflect {target} -> {replacement}")
                    continue

                new_word = inflected.word
                if token.istitle():
                    new_word = new_word.capitalize()

                new_tokens.append(new_word)
                replaced = True
                replaced_curr = True
                break

            if not replaced_curr:
                new_tokens.append(token)
        else:
            new_tokens.append(token)

    if not replaced:
        if debug:
            print(f"no replacement for {target}")
        return None
    return new_tokens
