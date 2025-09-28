#!/usr/bin/env python3
"""
Create quick visual previews of prepared LipBengal crops.

Loads a few prepared .pt files (saved as {"frames": uint8 (T,H,W)}) and
saves a contact sheet (grid) per sample as PNG.

Usage:
  python scripts/preview_prepared_lipbengal.py \
    --root datasets/LipBengal/prepared --split val --limit 6 --out media/previews
"""
from __future__ import annotations

import argparse
from pathlib import Path
import re
import math
import os
from typing import List

import numpy as np
from PIL import Image
import torch


def natural_key(p: Path):
    s = p.stem
    m = re.findall(r"\d+", s)
    return (int(m[-1]) if m else s,)


def find_prepared(root: Path, split: str) -> List[Path]:
    base = root / split
    if not base.exists():
        raise FileNotFoundError(f"Prepared split not found: {base}")
    return sorted(list(base.rglob("*.pt")), key=natural_key)


def sample_indices(n: int, limit: int) -> List[int]:
    if n <= limit:
        return list(range(n))
    # spread indices evenly across the range
    step = n / float(limit)
    idxs = [min(n - 1, int(i * step)) for i in range(limit)]
    # ensure uniqueness and order
    seen = set()
    uniq = []
    for i in idxs:
        if i not in seen:
            uniq.append(i)
            seen.add(i)
    # if we lost some due to rounding, fill gaps
    k = 0
    while len(uniq) < min(limit, n):
        if k not in seen:
            uniq.append(k)
            seen.add(k)
        k += 1
    return uniq


def contact_sheet(frames: np.ndarray, cols: int = 4, rows: int = 4, pad: int = 2) -> Image.Image:
    """
    frames: (T, H, W) uint8
    Returns a PIL Image of size ((W+pad)*cols+pad, (H+pad)*rows+pad)
    """
    assert frames.ndim == 3, f"Expected (T,H,W), got {frames.shape}"
    T, H, W = frames.shape
    n = cols * rows
    # choose n frames evenly spaced
    if T >= n:
        idxs = sample_indices(T, n)
    else:
        idxs = list(range(T)) + [T - 1] * (n - T)
    canvas = Image.new("L", (cols * W + pad * (cols + 1), rows * H + pad * (rows + 1)), color=0)
    for k, idx in enumerate(idxs):
        r = k // cols
        c = k % cols
        y = pad + r * (H + pad)
        x = pad + c * (W + pad)
        im = Image.fromarray(frames[idx]) if isinstance(frames[idx], np.ndarray) else Image.fromarray(np.array(frames[idx]))
        canvas.paste(im, (x, y))
    return canvas.convert("L")


def infer_labels_from_path(p: Path):
    # .../prepared/{split}/{speaker}/{word}/{hash}.pt
    try:
        word = p.parent.name
        speaker = p.parent.parent.name
        return speaker, word
    except Exception:
        return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="datasets/LipBengal/prepared")
    ap.add_argument("--split", default="val")
    ap.add_argument("--limit", type=int, default=6)
    ap.add_argument("--cols", type=int, default=4)
    ap.add_argument("--rows", type=int, default=4)
    ap.add_argument("--out", default="media/previews")
    args = ap.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    pts = find_prepared(root, args.split)
    if not pts:
        raise SystemExit(f"No prepared .pt files found under {root/args.split}")

    sel_idxs = sample_indices(len(pts), args.limit)
    sel = [pts[i] for i in sel_idxs]

    out_files = []
    for p in sel:
        try:
            data = torch.load(p, map_location="cpu")
            frames = data.get("frames")
            if torch.is_tensor(frames):
                frames = frames.detach().cpu().numpy()
            assert isinstance(frames, np.ndarray)
            assert frames.ndim == 3
            T, H, W = frames.shape
            sheet = contact_sheet(frames, cols=args.cols, rows=args.rows, pad=2)
            spk, word = infer_labels_from_path(p)
            name_bits = [args.split]
            if spk:
                name_bits.append(spk)
            if word:
                name_bits.append(word)
            name_bits.append(p.stem[:8])
            out_name = "_".join(name_bits) + ".png"
            out_path = out_dir / out_name
            sheet.save(out_path)
            out_files.append(out_path)
        except Exception as e:
            print(f"Failed {p}: {e}")
            continue

    print("Saved previews:")
    for f in out_files:
        print(f"- {f}")


if __name__ == "__main__":
    main()
