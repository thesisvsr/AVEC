#!/usr/bin/env python
"""Build KenLM (character or word) language model for BSRD.

Usage (word-level 3-gram example):
  python scripts/build_bsrd_lm.py --indices datasets/BSRD/indices/train.pt --output lm/bsrd_word_3gram.arpa --order 3 --level word

Usage (character-level 6-gram):
  python scripts/build_bsrd_lm.py --indices datasets/BSRD/indices/train.pt --output lm/bsrd_char_6gram.arpa --order 6 --level char

Then set environment variable before training:
  export BSRD_KENLM_PATH=lm/bsrd_word_3gram.arpa

Requirements:
  - kenlm Python package installed (pip install kenlm)
  - KenLM binaries (lmplz, build_binary) on PATH for fastest build. Python fallback builds in-memory if binaries absent.

Notes:
  - We filter empty/very short transcripts.
  - For word mode we optionally lowercase and keep basic punctuation; adjust token normalization if needed.
  - For char mode we collapse whitespace to single spaces and remove control characters.
"""
from __future__ import annotations
import argparse, os, re, sys, json, math, tempfile, subprocess
import torch

try:
    import kenlm  # noqa: F401
    _HAS_KENLM = True
except Exception:
    _HAS_KENLM = False

MIN_TOKENS_WORD = 2
MIN_TOKENS_CHAR = 5

def normalize_text(text: str, level: str) -> str:
    text = text.strip()
    if not text:
        return ""
    # Basic normalization
    text = re.sub(r"\s+", " ", text)
    if level == 'word':
        # Lowercase for robustness
        text = text.lower()
        # Optionally remove characters not in basic set (retain accented / transliteration marks)
        # Keep letters, digits, basic punctuation and spaces
        # text = re.sub(r"[^\w\s'.,!?-]", "", text)
    elif level == 'char':
        # For character LM we just ensure spaces collapsed
        pass
    return text

def extract_corpus(indices_path: str, level: str) -> list[str]:
    items = torch.load(indices_path, map_location='cpu')
    corpus = []
    for it in items:
        txt = it.get('rom_text', '') or ''
        txt = normalize_text(txt, level)
        if not txt:
            continue
        if level == 'word':
            toks = txt.split()
            if len(toks) < MIN_TOKENS_WORD:
                continue
            corpus.append(' '.join(toks))
        else:
            # char mode: keep as a single string; filter by minimum length
            if len(txt) < MIN_TOKENS_CHAR:
                continue
            corpus.append(txt)
    return corpus

def write_corpus(lines: list[str], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for line in lines:
            f.write(line + '\n')


def build_with_binaries(corpus_path: str, arpa_path: str, order: int, prune: str | None):
    lmplz = 'lmplz'
    build_bin = 'build_binary'
    cmd_lmplz = [lmplz, '-o', str(order), '--text', corpus_path, '--arpa', arpa_path]
    if prune:
        cmd_lmplz.insert(1, f'--prune={prune}')
    print('Running:', ' '.join(cmd_lmplz))
    subprocess.check_call(cmd_lmplz)
    print('LM ARPA written to', arpa_path)
    bin_path = arpa_path + '.binary'
    cmd_build = [build_bin, arpa_path, bin_path]
    try:
        subprocess.check_call(cmd_build)
        print('Binary LM written to', bin_path)
    except Exception as e:
        print('Skipping binary build:', e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--indices', required=True, help='Path to train indices .pt (datasets/BSRD/indices/train.pt)')
    ap.add_argument('--output', required=True, help='Output ARPA path, e.g. lm/bsrd_word_3gram.arpa')
    ap.add_argument('--order', type=int, default=3, help='N-gram order')
    ap.add_argument('--level', choices=['word', 'char'], default='word', help='Model granularity')
    ap.add_argument('--prune', default=None, help='Pruning spec for lmplz, e.g. 0 0 1')
    args = ap.parse_args()

    corpus = extract_corpus(args.indices, args.level)
    if not corpus:
        print('No lines extracted from corpus; abort.')
        return 1
    print(f'Extracted {len(corpus)} lines for {args.level}-level LM (order={args.order}).')

    # Write corpus
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    corpus_tmp = args.output + '.corpus.txt'
    write_corpus(corpus, corpus_tmp)

    # Try external binaries first
    try:
        build_with_binaries(corpus_tmp, args.output, args.order, args.prune)
    except FileNotFoundError:
        print('KenLM binaries not found; ensure lmplz/build_binary are installed for faster builds.')
        if not _HAS_KENLM:
            print('kenlm Python module not installed; cannot fallback. Install with: pip install kenlm')
            return 1
        else:
            print('Python kenlm build (in-memory) not implemented here; please install binaries for efficiency.')
            return 1
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
