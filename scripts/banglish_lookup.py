from __future__ import annotations

from pathlib import Path
from functools import lru_cache
from typing import List, Dict
import unicodedata

BN_PATH = Path("datasets/LipBengal/word_list.txt")
EN_PATH = Path("datasets/LipBengal/word_list_banglish.txt")


def _read_nonempty_lines(p: Path) -> List[str]:
    if not p.exists():
        raise FileNotFoundError(f"Missing list file: {p}")
    text = p.read_text(encoding="utf-8")
    # Keep spaces inside lines; drop empty lines
    return [ln.rstrip("\n") for ln in text.splitlines() if ln.strip() != ""]


@lru_cache(maxsize=1)
def _mapping() -> Dict[str, str]:
    bn = _read_nonempty_lines(BN_PATH)
    en = _read_nonempty_lines(EN_PATH)
    if len(bn) != len(en):
        raise ValueError(
            f"List length mismatch: {len(bn)} Bengali vs {len(en)} Banglish. "
            "Ensure both files are generated from the same source and aligned by index."
        )
    # Build both exact and normalized maps
    m: Dict[str, str] = {}
    for b, e in zip(bn, en):
        m[b] = e
        nb = _normalize_bn(b)
        if nb not in m:
            m[nb] = e
        nb_nospace = nb.replace(" ", "")
        if nb_nospace not in m:
            m[nb_nospace] = e
    return m


def _normalize_bn(s: str) -> str:
    # Normalize Unicode form and whitespace; strip zero-width joiners
    if not isinstance(s, str):
        return s
    s = s.replace("\u00A0", " ")  # NBSP -> space
    s = s.replace("\u200c", "").replace("\u200d", "")  # ZWNJ/ZWJ
    s = unicodedata.normalize("NFC", s)
    s = " ".join(s.split())  # collapse whitespace
    return s


def _transliterate_fallback(w: str) -> str:
    """Lightweight Bangla->Banglish transliteration for unknown words."""
    INDEP_VOWELS = {
        "অ": "o", "আ": "a", "ই": "i", "ঈ": "ii", "উ": "u", "ঊ": "uu",
        "ঋ": "ri", "ৠ": "ri", "ঌ": "li", "ৡ": "li",
        "এ": "e", "ঐ": "oi", "ও": "o", "ঔ": "ou",
    }
    VOWEL_SIGNS = {
        "া": "a", "ি": "i", "ী": "ii", "ু": "u", "ূ": "uu",
        "ৃ": "ri", "ে": "e", "ৈ": "oi", "ো": "o", "ৌ": "ou",
    }
    CONSONANTS = {
        "ক": "k", "খ": "kh", "গ": "g", "ঘ": "gh", "ঙ": "ng",
        "চ": "ch", "ছ": "chh", "জ": "j", "ঝ": "jh", "ঞ": "ny",
        "ট": "t", "ঠ": "th", "ড": "d", "ঢ": "dh", "ণ": "n",
        "ত": "t", "থ": "th", "দ": "d", "ধ": "dh", "ন": "n",
        "প": "p", "ফ": "f", "ব": "b", "ভ": "bh", "ম": "m",
        "য": "j", "র": "r", "ল": "l", "শ": "sh", "ষ": "sh", "স": "s", "হ": "h",
        "ড়": "r", "ঢ়": "rh", "য়": "y", "য়": "y", "ৎ": "t",
    }
    VIRAMA = "্"
    NASAL = {"ং": "ng", "ঁ": "n"}
    VISARGA = {"ঃ": "h"}

    def _token(token: str) -> str:
        s = token
        out = []
        i = 0
        n = len(s)
        while i < n:
            c = s[i]
            if c in INDEP_VOWELS:
                out.append(INDEP_VOWELS[c]); i += 1; continue
            if c in CONSONANTS:
                base = CONSONANTS[c]
                if i + 1 < n and s[i + 1] in VOWEL_SIGNS:
                    out.append(base + VOWEL_SIGNS[s[i + 1]]); i += 2; continue
                if i + 1 < n and s[i + 1] == VIRAMA:
                    out.append(base); i += 2; continue
                if i < n - 1:
                    out.append(base + "o")
                else:
                    out.append(base)
                i += 1; continue
            if c in VOWEL_SIGNS:
                out.append(VOWEL_SIGNS[c]); i += 1; continue
            if c in NASAL:
                out.append(NASAL[c]); i += 1; continue
            if c in VISARGA:
                out.append(VISARGA[c]); i += 1; continue
            out.append(c); i += 1
        return "".join(out).lower()

    return " ".join(_token(t) for t in w.split(" "))


def to_banglish(word: str) -> str:
    """
    Given a Bengali word exactly as it appears in datasets/LipBengal/word_list.txt,
    return the corresponding Banglish string from datasets/LipBengal/word_list_banglish.txt.
    """
    m = _mapping()
    # Try exact then normalized keys and no-space variant
    keys = [word]
    nword = _normalize_bn(word)
    if nword != word:
        keys.append(nword)
    nword_ns = nword.replace(" ", "")
    if nword_ns not in keys:
        keys.append(nword_ns)
    for k in keys:
        if k in m:
            return m[k]
    # Fallback: approximate transliteration
    return _transliterate_fallback(nword)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python3 scripts/banglish_lookup.py <bengali word> [more words...]")
        raise SystemExit(2)
    for w in sys.argv[1:]:
        print(to_banglish(w))