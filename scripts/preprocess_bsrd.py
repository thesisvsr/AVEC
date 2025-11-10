#!/usr/bin/env python3
"""
Preprocess BSRD video entries similarly to LipBengal preprocessing using FAN for mouth ROI.

Input: Existing indices files (train/val/test) at datasets/BSRD/indices/<split>.pt
Each item dict may contain keys like: speaker, id, video_path, rom_text, bn_text, prepared (optional), T

Output:
  - Prepared crops saved to: datasets/BSRD/prepared/<split>/<speaker>/<id_or_hash>.pt
    Format: torch.save({'frames': (T,H,W) uint8 tensor})
  - Updated indices file (overwritten) with 'prepared' path and updated 'T'.

Features:
  - FAN landmark detection (68-point) to isolate mouth (points 48..67) -> enlarge bbox -> crop -> grayscale 88x88.
  - Per-clip detection (single frame, middle) by default for speed; optional per-frame detection.
  - Overwrite or update-only modes.
  - Sharding support for distributed preprocessing across multiple processes.
  - Optional --limit for quick smoke test.
  - tqdm progress bar with per-step postfix (status + counts).

Example (full overwrite all splits with FAN on GPU):
  source .venv/bin/activate
  python scripts/preprocess_bsrd.py --splits train,val,test --landmarks fan --device cuda --overwrite

Quick test (first 10 train items):
  python scripts/preprocess_bsrd.py --splits train --landmarks fan --device cuda --overwrite --limit 10
"""
from __future__ import annotations

import os
import sys
import argparse
from pathlib import Path
import warnings
import hashlib
import json
from typing import List, Dict, Optional, Tuple

import torch
import numpy as np
from tqdm import tqdm

try:
    from PIL import Image
except Exception as e:
    raise RuntimeError(f"PIL (Pillow) required: {e}")

# Optional: OpenCV for faster grayscale + resize
try:
    import cv2  # type: ignore
    _CV2_OK = True
except Exception:
    cv2 = None  # type: ignore
    _CV2_OK = False

# FAN predictor
try:
    from ibug.face_alignment import FANPredictor  # type: ignore
    _FAN_OK = True
except Exception:
    FANPredictor = None  # type: ignore
    _FAN_OK = False

# Torchvision video decoding fallback
try:
    import torchvision
    _TV_OK = True
except Exception:
    torchvision = None
    _TV_OK = False

MOUTH_POINTS = list(range(48, 68))
IMG_SIZE = 88


def hash_key(entry: Dict) -> str:
    h = hashlib.sha1()
    for k in ('video_path','speaker','id','rom_text'):
        v = entry.get(k)
        if v:
            h.update(str(v).encode('utf-8'))
    return h.hexdigest()[:16]


def load_video_frames(path: str, max_frames: Optional[int] = None) -> List[np.ndarray]:
    """Decode video into list of RGB frames (uint8)."""
    frames: List[np.ndarray] = []
    if not os.path.isfile(path):
        return frames
    # Prefer torchvision if available
    if _TV_OK:
        try:
            vid, _, _ = torchvision.io.read_video(path, pts_unit='sec')  # (T,H,W,C) uint8
            if vid.ndim != 4:
                return frames
            for i in range(vid.shape[0]):
                frames.append(vid[i].numpy())
            if max_frames and len(frames) > max_frames:
                frames = frames[:max_frames]
            return frames
        except Exception:
            frames = []
    # Fallback: cv2
    if cv2 is not None:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            return frames
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
            if max_frames and len(frames) >= max_frames:
                break
        cap.release()
    return frames


def detect_mouth_bbox_fan(img_rgb: np.ndarray, fan_predictor) -> Optional[Tuple[int,int,int,int]]:
    try:
        out = fan_predictor(img_rgb)
        if out is None:
            return None
        if isinstance(out, tuple) and len(out) >= 1:
            landmarks_list = out[0]
        else:
            landmarks_list = out
        if not landmarks_list:
            return None
        pts = np.array(landmarks_list[0], dtype=np.float32)
        if pts.ndim != 2 or pts.shape[0] < 68:
            return None
        mouth = pts[MOUTH_POINTS, :]
        x0, y0 = mouth.min(axis=0)
        x1, y1 = mouth.max(axis=0)
        if x1 <= x0 or y1 <= y0:
            return None
        return int(x0), int(y0), int(x1), int(y1)
    except Exception:
        return None


def crop_mouth_sequence(frames: List[np.ndarray], bbox: Optional[Tuple[int,int,int,int]], enlarge=1.4, per_frame=False, fan=None) -> np.ndarray:
    """Return stacked grayscale mouth crops (T,H,W). If per_frame=True, redetect each frame."""
    crops: List[np.ndarray] = []
    T = len(frames)
    ref_bbox = bbox
    for t, img in enumerate(frames):
        if per_frame and fan is not None:
            bbox_t = detect_mouth_bbox_fan(img, fan) or ref_bbox
        else:
            bbox_t = ref_bbox
        if bbox_t is None:
            # center square fallback
            H,W = img.shape[:2]
            side = min(H,W)
            cy,cx = H//2, W//2
            half = side//2
            y0,y1 = cy-half, cy+half
            x0,x1 = cx-half, cx+half
        else:
            x0,y0,x1,y1 = bbox_t
            cx = (x0+x1)/2.0
            cy = (y0+y1)/2.0
            bw = (x1-x0)*enlarge
            bh = (y1-y0)*enlarge
            x0 = int(max(0, cx-bw/2))
            x1 = int(cx+bw/2)
            y0 = int(max(0, cy-bh/2))
            y1 = int(cy+bh/2)
        roi = img[y0:y1, x0:x1]
        if roi.size == 0:
            H,W = img.shape[:2]
            roi = img[max(0,H//2-32):min(H,H//2+32), max(0,W//2-32):min(W,W//2+32)]
        if _CV2_OK:
            gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
            crop = cv2.resize(gray, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
        else:
            crop = Image.fromarray(roi).convert('L').resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
            crop = np.array(crop, dtype=np.uint8)
        crops.append(crop)
    if not crops:
        crops = [np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)]
    return np.stack(crops, axis=0)


def process_split(split: str, args):
    indices_path = Path(args.indices_dir) / f"{split}.pt"
    if not indices_path.is_file():
        print(f"[WARN] Missing indices file for split '{split}': {indices_path}")
        return
    items: List[Dict] = torch.load(indices_path, map_location='cpu')
    if not isinstance(items, list):
        print(f"[WARN] Indices file malformed for split '{split}'")
        return
    if args.limit:
        items = items[:args.limit]
    out_root = Path(args.prepared_dir) / split
    out_root.mkdir(parents=True, exist_ok=True)

    # Init FAN
    fan = None
    if args.landmarks == 'fan':
        if not _FAN_OK:
            raise RuntimeError("FAN requested but ibug.face_alignment not installed")
        dev = args.device
        if dev == 'cuda' and not torch.cuda.is_available():
            print("[WARN] CUDA not available, using CPU for FAN")
            dev = 'cpu'
        fan = FANPredictor(device=dev)  # type: ignore
        print(f"Initialized FAN for split '{split}' on {dev}")

    processed = 0
    skipped = 0
    pbar = tqdm(items, desc=f"{split} (FAN)", unit="clip")
    for it in pbar:
        vid_path = it.get('video_path')
        if not vid_path or not os.path.isfile(vid_path):
            skipped += 1
            pbar.set_postfix({'skip':'missing'})
            continue
        hkey = hash_key(it)
        # Use speaker or 'unknown'
        speaker = it.get('speaker', 'spk')
        out_pt = out_root / speaker / f"{hkey}.pt"
        out_pt.parent.mkdir(parents=True, exist_ok=True)
        if (not args.overwrite) and out_pt.exists() and not args.update_only:
            # Already exists and we don't overwrite
            it['prepared'] = str(out_pt)
            processed += 1
            continue
        frames = load_video_frames(vid_path, max_frames=args.max_frames)
        if not frames:
            skipped += 1
            pbar.set_postfix({'skip':'decode'})
            continue
        mid_idx = len(frames)//2
        bbox = None
        if fan is not None:
            bbox = detect_mouth_bbox_fan(frames[mid_idx], fan)
        crops = crop_mouth_sequence(frames, bbox, enlarge=args.bbox_enlarge, per_frame=(args.bbox_mode=='per_frame'), fan=fan if args.bbox_mode=='per_frame' else None)
        arr = torch.from_numpy(crops)  # (T,H,W) uint8
        try:
            torch.save({'frames': arr}, out_pt)
        except Exception as e:
            skipped += 1
            pbar.set_postfix({'save':'fail'})
            continue
        it['prepared'] = str(out_pt)
        it['T'] = int(arr.shape[0])
        processed += 1
        if processed % 10 == 0:
            pbar.set_postfix({'proc': processed, 'skip': skipped})
    # Save updated indices
    torch.save(items, indices_path)
    print(f"Split '{split}': processed={processed} skipped={skipped} -> updated {indices_path}")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--splits', default='train,val,test', help='Comma-separated splits to process')
    ap.add_argument('--indices_dir', default='datasets/BSRD/indices', help='Indices directory')
    ap.add_argument('--prepared_dir', default='datasets/BSRD/prepared', help='Output directory for prepared crops')
    ap.add_argument('--landmarks', choices=['fan'], default='fan', help='Landmark method (currently only fan supported here)')
    ap.add_argument('--device', default='cuda', help='Device for FAN (cuda|cpu)')
    ap.add_argument('--bbox_mode', choices=['per_clip','per_frame'], default='per_clip', help='Redetect each frame or single middle frame')
    ap.add_argument('--bbox_enlarge', type=float, default=1.4, help='Enlarge factor around detected mouth bbox')
    ap.add_argument('--overwrite', action='store_true', help='Overwrite existing prepared files')
    ap.add_argument('--update_only', action='store_true', help='Skip items whose prepared exists (no recompute)')
    ap.add_argument('--max_frames', type=int, default=None, help='Optional max frames to decode per clip')
    ap.add_argument('--limit', type=int, default=None, help='Process only first N items (debug)')
    return ap.parse_args()


def main():
    args = parse_args()
    splits = [s.strip() for s in args.splits.split(',') if s.strip()]
    for sp in splits:
        process_split(sp, args)

if __name__ == '__main__':
    main()
