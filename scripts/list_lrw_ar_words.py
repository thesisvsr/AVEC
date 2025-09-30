#!/usr/bin/env python3
"""List unique LRW-AR words from indices or raw folders and optionally emit Arabish mapping.

Outputs:
  - words_arabic.txt (Arabic forms)
  - words_arabish.txt (transliterated)
  - mapping_ar_to_arabish.json (dictionary)

Priority of sources:
  1. indices/*.pt (contains 'word') if present
  2. sorted_labels.txt if present
  3. Fallback: directory names under each split
"""
from __future__ import annotations
import os, json, argparse
from pathlib import Path

try:
    import torch
except Exception:
    torch = None  # type: ignore

def load_indices_words(indices_dir: Path):
    words = set()
    if torch is None:
        return words
    for split in ("train","val","test"):
        f = indices_dir / f"{split}.pt"
        if not f.is_file():
            continue
        try:
            items = torch.load(f, map_location="cpu")
            for it in items:
                w = it.get("word_orig") or it.get("word")
                if isinstance(w, str) and w:
                    words.add(w)
        except Exception:
            continue
    return words

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="datasets/LRW-AR", help="LRW-AR root")
    ap.add_argument("--out_dir", default="datasets/LRW-AR", help="Output directory for word lists")
    ap.add_argument("--use_indices", action="store_true", help="Prefer indices if available")
    ap.add_argument("--simplify", action="store_true", help="Simplify emphatics in Arabish mapping")
    args = ap.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    words = set()
    if args.use_indices:
        words |= load_indices_words(root / "indices")
    if not words:
        labels_file = root / "sorted_labels.txt"
        if labels_file.is_file():
            for line in labels_file.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line:
                    words.add(line)
    if not words:
        for split in ("train","val","test"):
            for d in (root / split).glob("*"):
                if d.is_dir():
                    words.add(d.name)
    words = sorted(words)

    # Arabish mapping
    try:
        from scripts.arabish_lookup import to_arabish  # type: ignore
    except Exception:
        # Fallback identity; log once
        def to_arabish(x, simplify=True):
            return x
        print("Warning: could not import scripts.arabish_lookup.to_arabish; mapping will be identity.")
    mapping = {w: to_arabish(w, simplify=True) for w in words}

    (out_dir / "words_arabic.txt").write_text("\n".join(words), encoding="utf-8")
    (out_dir / "words_arabish.txt").write_text("\n".join(mapping[w] for w in words), encoding="utf-8")
    with (out_dir / "mapping_ar_to_arabish.json").open("w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)

    print(f"Arabic words: {len(words)} -> {out_dir/'words_arabic.txt'}")
    print(f"Arabish words: {out_dir/'words_arabish.txt'}")
    print(f"Mapping JSON: {out_dir/'mapping_ar_to_arabish.json'}")

if __name__ == "__main__":
    main()
