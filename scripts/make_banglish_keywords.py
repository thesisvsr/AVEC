from pathlib import Path

# IO paths
inp = Path("datasets/LipBengal/word_list.txt")
out = Path("datasets/LipBengal/word_list_banglish.txt")

if not inp.exists():
    raise SystemExit(f"Input not found: {inp}")

# Basic Banglish transliterator (ASCII, lowercase).
# Heuristic: add inherent 'o' after consonants except at end of a word or before a virama.
INDEP_VOWELS = {
    "অ": "o", "আ": "a", "ই": "i", "ঈ": "ii", "উ": "u", "ঊ": "uu",
    "ঋ": "ri", "ৠ": "ri", "ঌ": "li", "ৡ": "li",
    "এ": "e", "ঐ": "oi", "ও": "o", "ঔ": "ou",
}

VOWEL_SIGNS = {
    "া": "a", "ি": "i", "ী": "ii", "ু": "u", "ূ": "uu",
    "ৃ": "ri", "ে": "e", "ৈ": "oi", "ো": "o", "ৌ": "ou",
}

# Common consonants and specials
CONSONANTS = {
    "ক": "k", "খ": "kh", "গ": "g", "ঘ": "gh", "ঙ": "ng",
    "চ": "ch", "ছ": "chh", "জ": "j", "ঝ": "jh", "ঞ": "ny",
    "ট": "t", "ঠ": "th", "ড": "d", "ঢ": "dh", "ণ": "n",
    "ত": "t", "থ": "th", "দ": "d", "ধ": "dh", "ন": "n",
    "প": "p", "ফ": "f",  # everyday Banglish usually uses 'f'
    "ব": "b", "ভ": "bh", "ম": "m",
    "য": "j", "র": "r", "ল": "l", "শ": "sh", "ষ": "sh", "স": "s", "হ": "h",
    "ড়": "r", "ঢ়": "rh", "য়": "y", "য়": "y", "঵": "b", "ৎ": "t",
    "ৎ": "t", "ঙ্খ": "nkh", "ঙ্ঘ": "ngh",  # rare clusters
}

# Marks and signs
VIRAMA = "্"
NASAL = {"ং": "ng", "ঁ": "n"}  # chandrabindu/anusvara -> approx 'n/ng'
VISARGA = {"ঃ": "h"}

# Precompose some frequent clusters for better output
# Handle 'ক্ষ' as 'ksh' when seen as the cluster ক্ + ষ
def normalize_clusters(s: str) -> str:
    return s.replace("ক্ষ", "ক্‌ষ")  # mark explicit virama position (ZWJ-like)
    # We’ll still map ক + VIRAMA + ষ -> k + sh w/o vowel in between.

def is_bengali_letter(ch: str) -> bool:
    return "\u0980" <= ch <= "\u09FF"

def transliterate_word(w: str) -> str:
    # Keep spaces inside multi-token entries; run per token separated by spaces
    tokens = w.split(" ")
    return " ".join(_trans_token(t) for t in tokens)

def _trans_token(token: str) -> str:
    s = token
    out = []
    i = 0
    n = len(s)
    while i < n:
        c = s[i]

        # Independent vowels
        if c in INDEP_VOWELS:
            out.append(INDEP_VOWELS[c])
            i += 1
            continue

        # Consonants
        if c in CONSONANTS:
            base = CONSONANTS[c]
            # vowel sign?
            if i + 1 < n and s[i + 1] in VOWEL_SIGNS:
                out.append(base + VOWEL_SIGNS[s[i + 1]])
                i += 2
                continue
            # virama?
            if i + 1 < n and s[i + 1] == VIRAMA:
                out.append(base)  # no inherent vowel
                i += 2
                continue
            # default: inherent 'o' unless final char of token
            if i < n - 1:
                out.append(base + "o")
            else:
                out.append(base)
            i += 1
            continue

        # Vowel signs without a base (rare) – treat as independent
        if c in VOWEL_SIGNS:
            out.append(VOWEL_SIGNS[c])
            i += 1
            continue

        # Nasalization and visarga
        if c in NASAL:
            out.append(NASAL[c])
            i += 1
            continue
        if c in VISARGA:
            out.append(VISARGA[c])
            i += 1
            continue

        # Punctuation, ASCII, spaces, etc.
        out.append(c)
        i += 1

    return "".join(out).lower()

# Process lines
lines = [ln.rstrip("\n") for ln in inp.read_text(encoding="utf-8").splitlines()]
banglish = [transliterate_word(ln) if ln.strip() else "" for ln in lines]

out.parent.mkdir(parents=True, exist_ok=True)
out.write_text("\n".join(banglish) + "\n", encoding="utf-8")

print(f"Wrote {len(banglish)} Banglish keywords -> {out}")
print("Sample:")
for i, (b, r) in enumerate(zip(lines, banglish)):
    if i == 10: break
    print(f"  {b}  ->  {r}")