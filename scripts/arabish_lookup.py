#!/usr/bin/env python3
"""Arabic -> Arabish (ASCII transliteration) helper.

This is a lightweight, rule-based transliteration tuned for LRW-AR word labels.
It is intentionally simple (1:1 or small digraph mappings) to produce stable
class identifiers usable as filenames, tokenizer symbols, or model output labels.

If you later choose a more linguistically accurate system (e.g. Buckwalter),
you can replace the mapping here; downstream prepared data remains valid
because only the "word" field inside indices is affected when loading with
use_arabish=True.
"""
from __future__ import annotations

import re

# Core mapping (Arabizi style, ASCII-safe) — emphatics simplified.
_ARABIC_BASE = {
    "ا": "a", "أ": "a", "إ": "i", "آ": "aa",
    "ب": "b", "ت": "t", "ث": "th", "ج": "j", "ح": "7", "خ": "kh",
    "د": "d", "ذ": "dh", "ر": "r", "ز": "z", "س": "s", "ش": "sh",
    "ص": "s", "ض": "d", "ط": "t", "ظ": "z",
    "ع": "3", "غ": "gh", "ف": "f", "ق": "q", "ك": "k", "ل": "l",
    "م": "m", "ن": "n", "ه": "h", "و": "w", "ؤ": "w", "ي": "y", "ى": "a",
    "ئ": "y", "ة": "a", "ء": "'",
}

# Diacritics & tatweel to strip.
_STRIP = set(list("\u064b\u064c\u064d\u064e\u064f\u0650\u0651\u0652\u0653\u0654\u0655\u0670\u0640"))

_RE_LEADING_AL = re.compile(r"^ال")  # definite article

def _strip_diacritics(s: str) -> str:
    return "".join(ch for ch in s if ch not in _STRIP)

def _normalize_word(w: str) -> str:
    # Basic normalizations: different alif/hamza forms unified earlier via mapping keys.
    return w

def to_arabish(word: str, simplify: bool = True, keep_apostrophe=False, collapse_double=True) -> str:
    w = word.strip()
    if not w:
        return w
    w = _strip_diacritics(_normalize_word(w))

    # Preserve original for possible heuristics.
    orig = w

    # Detect and transliterate leading 'ال' definite article to 'al' (common user expectation)
    leading_al = False
    if _RE_LEADING_AL.match(w):
        w_body = w[2:]
        leading_al = True
    else:
        w_body = w

    out_chars: list[str] = []
    for ch in w_body:
        out_chars.append(_ARABIC_BASE.get(ch, ch))
    core = "".join(out_chars)

    if leading_al:
        core = "al" + core

    # Remove apostrophes unless explicitly kept
    if not keep_apostrophe:
        core = core.replace("'", "")

    # Simplify some digraph sequences: optional (already simple)
    if simplify:
        # Collapse emphatic distinctions already removed; ensure no accidental uppercase, etc.
        pass

    if collapse_double:
        # Collapse triple repeats first
        core = re.sub(r"([a-z3hw7q])\1{2,}", r"\1\1", core)
        # Then optional collapse of some double long vowels if created by آ -> aa + following alif
        core = re.sub(r"aaa", "aa", core)

    # Final safety: keep only accepted chars
    core = re.sub(r"[^a-z0-9]+", lambda m: "_" if len(m.group(0)) > 0 else "", core)
    core = core.strip("_")
    if not core:
        core = "unk"
    return core

if __name__ == "__main__":
    tests = ["كتاب", "لغة", "مدرسة", "إن", "أخبار", "سياسة", "الأمم", "الاتفاق", "إيران", "غزة"]
    for w in tests:
        print(f"{w} -> {to_arabish(w)}")
