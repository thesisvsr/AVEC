#!/usr/bin/env python3
from __future__ import annotations

"""Bengali -> Banglish (ASCII transliteration) helper.

Lightweight, rule-based transliteration aimed at producing stable, lowercase
ASCII tokens for word-level classification (filenames / class labels).

Design goals:
 - Deterministic 1:1 mapping (if collisions occur, the dataset class will
   disambiguate by suffixing _2, _3, ... so we DO NOT collapse duplicates here).
 - Keep only [a-z0-9_] characters; strip or approximate everything else.
 - Very small / dependency-free (no external transliteration libraries).

Simplifications:
 - Inherent vowel (ô / ɔ) rendered as "o" when a consonant appears without an
   explicit vowel mark.
 - Long vs short vowels generally not distinguished beyond a/e/i/o/u (some
   approximate digraphs retained: ai, oi, oi, ou, au) -> we map ঐ/ৈ -> oi, ঔ/ৌ -> ou.
 - Anusvara (ং) -> ng; Chandrabindu (ঁ) -> n; Visarga (ঃ) -> h.

If you later want a more linguistically faithful system, you can swap out
`to_banglish` while keeping prepared data intact (indices store the original
Bangla word; transliteration applied at load time when `use_banglish=True`).
"""

import re

# Independent vowels
_VOWELS = {
    "অ": "o",  # inherent /ɔ/
    "আ": "a",
    "ই": "i",
    "ঈ": "i",
    "উ": "u",
    "ঊ": "u",
    "ঋ": "ri",
    "এ": "e",
    "ঐ": "oi",
    "ও": "o",
    "ঔ": "ou",
}

# Vowel signs (diacritics) combining with a consonant base
_VOWEL_SIGNS = {
    "া": "a",
    "ি": "i",
    "ী": "i",
    "ু": "u",
    "ূ": "u",
    "ৃ": "ri",
    "ে": "e",
    "ৈ": "oi",
    "ো": "o",
    "ৌ": "ou",
}

# Consonants
_CONS = {
    "ক": "k",  "খ": "kh", "গ": "g",  "ঘ": "gh", "ঙ": "ng",
    "চ": "c",  "ছ": "ch", "জ": "j",  "ঝ": "jh", "ঞ": "ny",
    "ট": "t",  "ঠ": "th", "ড": "d",  "ঢ": "dh", "ণ": "n",
    "ত": "t",  "থ": "th", "দ": "d",  "ধ": "dh", "ন": "n",
    "প": "p",  "ফ": "ph", "ব": "b",  "ভ": "bh", "ম": "m",
    "য": "y",  "র": "r",  "ল": "l",  "শ": "sh", "ষ": "sh", "স": "s", "হ": "h",
    "ড়": "r",  "ঢ়": "rh", "য়": "y",
}

# Specials / signs
_SPECIAL = {
    "ং": "ng",  # anusvara
    "ঃ": "h",   # visarga
    "ঁ": "n",   # chandrabindu
    "্": "",    # virama (halant) -> suppress inherent vowel
}

# Accepted final pattern (filter everything else to underscore)
_SAFE_RE = re.compile(r"[^a-z0-9]+")

def _tokenize(chars: str):  # minimal segmentation (character-wise)
    for ch in chars:
        yield ch

def to_banglish(word: str, collapse_double=True) -> str:
    w = word.strip()
    if not w:
        return w
    out: list[str] = []
    prev_was_cons = False
    have_explicit_vowel = False
    for ch in _tokenize(w):
        if ch in _VOWELS:
            # Independent vowel overrides pending inherent vowel
            out.append(_VOWELS[ch])
            prev_was_cons = False
            have_explicit_vowel = True
        elif ch in _CONS:
            # If previous consonant without explicit vowel, append inherent vowel 'o'
            if prev_was_cons and not have_explicit_vowel:
                out.append("o")
            out.append(_CONS[ch])
            prev_was_cons = True
            have_explicit_vowel = False
        elif ch in _VOWEL_SIGNS:
            # Attach vowel sign to previous consonant (replace inherent)
            out.append(_VOWEL_SIGNS[ch])
            prev_was_cons = False
            have_explicit_vowel = True
        elif ch in _SPECIAL:
            # If virama, just suppress inherent vowel by marking state
            if ch == "্":
                prev_was_cons = False  # next consonant treated anew
                have_explicit_vowel = True  # suppress adding 'o'
            else:
                out.append(_SPECIAL[ch])
                prev_was_cons = False
                have_explicit_vowel = True
        else:
            # Unknown char: treat as separator
            if prev_was_cons and not have_explicit_vowel:
                out.append("o")
            prev_was_cons = False
            have_explicit_vowel = True
    # Flush trailing inherent vowel
    if prev_was_cons and not have_explicit_vowel:
        out.append("o")
    token = "".join(out)
    if collapse_double:
        # Collapse >2 repeats then allow limited doubles for readability
        token = re.sub(r"([a-z])\1{2,}", r"\1\1", token)
    token = _SAFE_RE.sub("_", token)
    token = token.strip("_")
    if not token:
        token = "unk"
    return token

if __name__ == "__main__":  # simple smoke test
    tests = ["বাংলা", "অংশগ্রহণ", "অবস্থা", "অনুমান", "অধিকার", "অবশ্যই", "অঙ্কন", "অংশ"]
    for t in tests:
        print(f"{t} -> {to_banglish(t)}")