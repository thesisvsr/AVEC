#!/usr/bin/env python3
"""
Prepares LipBengal for training:
- Creates 60/20/20 speaker splits (by sorted speaker ids)
- Indexes utterances as lists of frame paths per split
- Optionally trains a char-level SentencePiece tokenizer from the vocabulary of words

Expected dataset layout:
    datasets/
      LipBengal/
        s1/
          WORD_A/
            01.jpg ...
          WORD_B/
            ...
        s2/
          ...

Outputs:
    datasets/LipBengal/splits/{train,val,test}_speakers.txt
    datasets/LipBengal/indices/{train,val,test}.pt   # list[dict]: {speaker, word, frames: [paths], T}
    datasets/LipBengal/tokenizer_char.model          # optional (char-level)

Run:
    python3 scripts/prepare_lipbengal.py --root datasets/LipBengal --min_len 6 --max_len 150 --vocab_size 256
"""

from __future__ import annotations

import os
import sys
import re
import math
import json
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import hashlib
import warnings
from tqdm import tqdm

# Make sure we can import the sibling 'nnet' package when running from scripts/
try:
    _REPO_ROOT = Path(__file__).resolve().parent.parent
    if str(_REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(_REPO_ROOT))
    _FDIR = _REPO_ROOT / "face_detection"
    if _FDIR.is_dir() and str(_FDIR) not in sys.path:
        sys.path.insert(0, str(_FDIR))
except Exception:
    pass

# Torch is used for saving indices and prepared tensors
try:
    import torch
except Exception:
    torch = None  # fallback to JSON for indices if torch missing

# SentencePiece is optional
try:
    import sentencepiece as spm
except Exception:
    spm = None

# Optional dependencies for landmark-based alignment
try:
    import mediapipe as mp  # type: ignore
    _MP_OK = True
except Exception:
    mp = None
    _MP_OK = False

# Optional FAN landmark predictor (GPU-capable via PyTorch)
try:
    from ibug.face_alignment import FANPredictor  # type: ignore
    _FAN_OK = True
except Exception:
    FANPredictor = None  # type: ignore
    _FAN_OK = False

# Optional LRS2-style lip crop (RetinaFace + FAN + mean-face warp)
try:
    from nnet.transforms import LipDetectCrop  # type: ignore
    _LIP_OK = True
except Exception:
    LipDetectCrop = None  # type: ignore
    _LIP_OK = False

from PIL import Image
import numpy as np

# Optional: OpenCV for faster grayscale/resize
try:
    import cv2  # type: ignore
    _CV2_OK = True
except Exception:
    cv2 = None  # type: ignore
    _CV2_OK = False

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def natural_key(p: Path):
    s = p.stem
    m = re.findall(r"\d+", s)
    return (int(m[-1]) if m else s,)


def list_frames(seq_dir: Path) -> List[Path]:
    return sorted([p for p in seq_dir.iterdir() if p.suffix.lower() in IMG_EXTS], key=natural_key)


def build_speaker_splits(speakers: List[str], train_ratio=0.6, val_ratio=0.2):
    n = len(speakers)
    n_train = int(math.floor(n * train_ratio))
    n_val = int(math.floor(n * val_ratio))
    # ensure at least 1 speaker per split when possible
    n_train = max(1, min(n - 2, n_train)) if n >= 3 else max(1, n_train)
    n_val = max(1, min(n - n_train - 1, n_val)) if n - n_train >= 2 else max(0, n - n_train - 1)
    n_test = max(0, n - n_train - n_val)
    return {
        "train": speakers[:n_train],
        "val": speakers[n_train:n_train + n_val],
        "test": speakers[n_train + n_val:],
    }


def index_split(root: Path, split_speakers: List[str], min_len=6, max_len: int | None = None) -> List[Dict]:
    items = []
    for spk in split_speakers:
        spk_dir = root / spk
        if not spk_dir.is_dir():
            continue
        for word_dir in sorted([d for d in spk_dir.iterdir() if d.is_dir()], key=lambda p: p.name):
            # Case A: frames directly in the word folder (one utterance)
            frames = list_frames(word_dir)
            if len(frames) >= min_len:
                frames_sel = frames[:max_len] if (max_len and len(frames) > max_len) else frames
                items.append({
                    "speaker": spk,
                    "word": word_dir.name,
                    "frames": [str(p) for p in frames_sel],
                    "T": len(frames_sel),
                })
            # Case B: nested sequences inside the word folder
            for seq_dir in sorted([d for d in word_dir.iterdir() if d.is_dir()], key=lambda p: p.name):
                f2 = list_frames(seq_dir)
                if len(f2) >= min_len:
                    frames_sel = f2[:max_len] if (max_len and len(f2) > max_len) else f2
                    items.append({
                        "speaker": spk,
                        "word": word_dir.name,
                        "frames": [str(p) for p in frames_sel],
                        "T": len(frames_sel),
                    })
    return items


def save_indices(items: List[Dict], out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if torch is not None:
        torch.save(items, out_path)
    else:
        # fallback to json
        with out_path.with_suffix(".json").open("w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False, indent=2)


def _hash_item(frames: List[str]) -> str:
    h = hashlib.sha1()
    for p in frames:
        h.update(p.encode("utf-8"))
    return h.hexdigest()[:16]


def _detect_mouth_bbox_mp(img_rgb: np.ndarray, mp_face_mesh) -> Optional[tuple[int, int, int, int]]:
    try:
        # Mediapipe expects RGB images in [0,1], but works with uint8 RGB as well
        h, w, _ = img_rgb.shape
        res = mp_face_mesh.process(img_rgb)
        if not res.multi_face_landmarks:
            return None
        lm = res.multi_face_landmarks[0].landmark
        # Subset of lip-related landmark indices (from MP FaceMesh)
        lip_idx = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 78, 191, 80, 81, 82, 13, 312, 311, 310, 415]
        xs = [int(lm[i].x * w) for i in lip_idx]
        ys = [int(lm[i].y * h) for i in lip_idx]
        x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
        # Basic sanity
        if x1 <= x0 or y1 <= y0:
            return None
        return x0, y0, x1, y1
    except Exception:
        return None


def _detect_mouth_bbox_fan(img_rgb: np.ndarray, fan_predictor) -> Optional[tuple[int, int, int, int]]:
    """Detect mouth bbox using FAN 68-point landmarks (indices 48..67)."""
    try:
        # FAN wrapper may return (landmarks_list, scores_list) or landmarks_list
        out = fan_predictor(img_rgb)
        if out is None:
            return None
        if isinstance(out, tuple) and len(out) >= 1:
            lms_list = out[0]
        else:
            lms_list = out
        if lms_list is None or len(lms_list) == 0:
            return None
        pts = lms_list[0]
        pts = np.array(pts, dtype=np.float32)
        if pts.ndim != 2 or pts.shape[0] < 68:
            return None
        mouth = pts[48:68, :]
        x0, y0 = mouth.min(axis=0)
        x1, y1 = mouth.max(axis=0)
        if x1 <= x0 or y1 <= y0:
            return None
        return int(x0), int(y0), int(x1), int(y1)
    except Exception:
        return None


def _crop_mouth_stack(
    frames: List[str],
    out_size=88,
    enlarge=1.6,
    mp_mesh=None,
    fan=None,
    fixed_bbox: Optional[Tuple[int, int, int, int]] = None,
) -> np.ndarray:
    """Return stacked uint8 grayscale mouth crops of shape (T, H, W).

    If fixed_bbox is provided, it will be used for all frames (fast path).
    Otherwise, will run landmark detection for each frame (slow path).
    """
    crops: List[np.ndarray] = []
    for fp in frames:
        with Image.open(fp) as im:
            im = im.convert("RGB")
            img = np.array(im)  # H, W, 3 (uint8)
        bbox = fixed_bbox
        if bbox is None:
            if fan is not None:
                bbox = _detect_mouth_bbox_fan(img, fan)
            if bbox is None and mp_mesh is not None:
                bbox = _detect_mouth_bbox_mp(img, mp_mesh)
        if bbox is None:
            # fallback to center crop square
            H, W = img.shape[:2]
            side = min(H, W)
            cy, cx = H // 2, W // 2
            half = side // 2
            y0, y1 = cy - half, cy + half
            x0, x1 = cx - half, cx + half
        else:
            x0, y0, x1, y1 = bbox
            cx = (x0 + x1) / 2.0
            cy = (y0 + y1) / 2.0
            bw = (x1 - x0) * float(enlarge)
            bh = (y1 - y0) * float(enlarge)
            x0 = int(max(0, cx - bw / 2))
            x1 = int(cx + bw / 2)
            y0 = int(max(0, cy - bh / 2))
            y1 = int(cy + bh / 2)
        roi = img[y0:y1, x0:x1]
        if _CV2_OK:
            # Convert RGB -> Gray and resize with OpenCV (faster than PIL for many small images)
            gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
            crop = cv2.resize(gray, (out_size, out_size), interpolation=cv2.INTER_AREA)
            crops.append(crop.astype(np.uint8, copy=False))
        else:
            crop = Image.fromarray(roi).convert("L").resize((out_size, out_size), Image.BILINEAR)
            crops.append(np.array(crop, dtype=np.uint8))
    if not crops:
        crops = [np.zeros((out_size, out_size), dtype=np.uint8)]
    x = np.stack(crops, axis=0)  # (T, H, W) uint8
    return x


def train_char_tokenizer(words: List[str], out_model: Path, vocab_size=256):
    if spm is None:
        print("Warning: sentencepiece not installed; skipping tokenizer training.")
        return
    out_model.parent.mkdir(parents=True, exist_ok=True)
    corpus_txt = out_model.with_suffix(".txt")
    with corpus_txt.open("w", encoding="utf-8") as f:
        for w in sorted(set(words)):
            f.write(w.strip() + "\n")
    spm.SentencePieceTrainer.Train(
        input=str(corpus_txt),
        model_prefix=str(out_model.with_suffix("")),
        model_type="char",
        vocab_size=vocab_size,
        character_coverage=1.0,
        add_dummy_prefix=False,
        input_sentence_size=1000000,
        shuffle_input_sentence=False,
        normalization_rule_name="identity",
    )
    print(f"Saved tokenizer model: {out_model}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="datasets/LipBengal", help="LipBengal root folder")
    ap.add_argument("--min_len", type=int, default=6, help="Min frames per utterance")
    ap.add_argument("--max_len", type=int, default=150, help="Max frames per utterance (truncate)")
    ap.add_argument("--train_ratio", type=float, default=0.6)
    ap.add_argument("--val_ratio", type=float, default=0.2)
    ap.add_argument("--tokenizer_model", default="datasets/LipBengal/tokenizer_char.model")
    ap.add_argument("--split_mode", choices=["by_speaker", "by_clip"], default="by_speaker", help="Split policy: by speakers (default) or randomized by clips (all speakers included in train)")
    ap.add_argument("--split_seed", type=int, default=42, help="Random seed for randomized by-clip split")
    ap.add_argument("--align", action="store_true", help="Align mouth ROI and cache prepared crops")
    ap.add_argument("--prepared_dir", default="datasets/LipBengal/prepared", help="Output dir for prepared crops")
    ap.add_argument("--splits", default=None, help="Comma-separated list of splits to process (e.g., 'train,val'). Default: all")
    ap.add_argument("--landmarks", choices=["mp", "fan", "lipcrop", "none"], default="mp", help="Alignment method: mediapipe (CPU), FAN-only bbox, lipcrop=LRS2-style (RetinaFace+FAN+mean-face warp), or none (center crop)")
    ap.add_argument("--device", default="cpu", help="Device for FAN (cuda or cpu)")
    ap.add_argument("--bbox_mode", choices=["per_clip", "per_frame"], default="per_clip", help="Detect bbox once per clip (fast) or every frame (slow)")
    ap.add_argument("--bbox_enlarge", type=float, default=1.2, help="Scale factor for bbox fallback around lips")
    ap.add_argument("--shard", type=int, default=0, help="Shard index [0..num_shards-1]")
    ap.add_argument("--num_shards", type=int, default=1, help="Total number of shards")
    ap.add_argument("--update_only", action="store_true", help="Skip items whose prepared file already exists")
    ap.add_argument("--overwrite", action="store_true", help="Recompute and overwrite existing prepared files")
    ap.add_argument("--drop_on_fail", action="store_true", help="If lip detection fails (center fallback), drop the item instead of saving and exclude from indices")
    ap.add_argument("--fail_list_dir", default="logs", help="Directory to write lists of items with failed lip detection (per split/shard)")
    ap.add_argument("--skip_tokenizer", action="store_true", help="Skip training tokenizer (useful for shard runs)")
    ap.add_argument("--vocab_size", type=int, default=256)
    args = ap.parse_args()

    root = Path(args.root)
    assert root.is_dir(), f"Not found: {root}"

    # Speakers: s1..sN sorted numerically
    speakers = sorted(
        [d.name for d in root.iterdir() if d.is_dir() and d.name.startswith("s") and d.name[1:].isdigit()],
        key=lambda n: int(n[1:])
    )
    assert speakers, "No speakers found (expected folders like s1, s2, ...)"

    # Build splits either by speakers or by clips
    splits_dir = root / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    assignment_path = splits_dir / f"by_clip_assign_seed{args.split_seed}.json"
    if args.split_mode == "by_speaker":
        # 60/20/20 by speakers
        splits = build_speaker_splits(speakers, args.train_ratio, args.val_ratio)
        for k, v in splits.items():
            (splits_dir / f"{k}_speakers.txt").write_text("\n\n".join(v).replace("\n\n", "\n"), encoding="utf-8")
            print(f"{k}: {len(v)} speakers")
    else:
        # Randomized by-clip split over all speakers with reproducible seed, ensuring all speakers appear in train
        import numpy as _np
        # Index all items once
        print("Indexing all clips for by-clip split assignment...")
        all_items = index_split(root, speakers, min_len=args.min_len, max_len=args.max_len)
        print(f"Total clips indexed: {len(all_items)}")
        # Build per-speaker index lists and global order
        per_spk = {}
        for i, it in enumerate(all_items):
            per_spk.setdefault(it["speaker"], []).append(i)
        order = _np.arange(len(all_items))
        rng = _np.random.RandomState(args.split_seed)
        rng.shuffle(order)
        n = len(all_items)
        n_train = int(math.floor(n * args.train_ratio))
        n_val = int(math.floor(n * args.val_ratio))
        # Initial assignment
        train_idx = set(order[:n_train].tolist())
        val_idx = set(order[n_train:n_train + n_val].tolist())
        test_idx = set(order[n_train + n_val:].tolist())
        # Ensure every speaker appears in train: move one clip from val/test if missing
        for spk, idxs in per_spk.items():
            if any(i in train_idx for i in idxs):
                continue
            moved = False
            for src_set in (val_idx, test_idx):
                for i in idxs:
                    if i in src_set:
                        src_set.remove(i)
                        train_idx.add(i)
                        moved = True
                        break
                if moved:
                    break
        # Build mapping from item hash to split
        def _item_hash(it):
            return _hash_item(it["frames"])  # 16-hex digest based on frame paths
        assign_map = {}
        for i in range(n):
            h = _item_hash(all_items[i])
            if i in train_idx:
                assign_map[h] = "train"
            elif i in val_idx:
                assign_map[h] = "val"
            else:
                assign_map[h] = "test"
        # Persist assignment so per-split shard runs can reuse it
        try:
            with assignment_path.open("w", encoding="utf-8") as f:
                json.dump(assign_map, f)
            print(f"Saved by-clip assignment: {assignment_path}")
        except Exception as e:
            warnings.warn(f"Failed to save clip assignment mapping: {e}")
        # Derive speaker presence per split for logging
        speakers_per_split = {"train": set(), "val": set(), "test": set()}
        for i, it in enumerate(all_items):
            h = _hash_item(it["frames"])
            sp = assign_map.get(h, "train")
            speakers_per_split[sp].add(it["speaker"])
        splits = {k: sorted(list(v)) for k, v in speakers_per_split.items()}
        for k, v in splits.items():
            (splits_dir / f"{k}_speakers.txt").write_text("\n".join(v), encoding="utf-8")
            print(f"{k}: {len(v)} speakers (by-clip)")

    # Which splits to process
    if args.splits:
        wanted = {s.strip() for s in args.splits.split(",") if s.strip()}
    else:
        wanted = set(splits.keys())

    # Index utterances for each selected split
    indices_dir = root / "indices"
    indices_dir.mkdir(parents=True, exist_ok=True)
    all_words = set()
    for split_name, spk_list in splits.items():
        if split_name not in wanted:
            continue
        # Build item list for this split
        if args.split_mode == "by_speaker":
            items = index_split(root, spk_list, min_len=args.min_len, max_len=args.max_len)
        else:
            # Load or rebuild assignment map
            assign_map = None
            if assignment_path.exists():
                try:
                    with assignment_path.open("r", encoding="utf-8") as f:
                        assign_map = json.load(f)
                except Exception as e:
                    warnings.warn(f"Failed to load assignment map: {e}")
            if assign_map is None:
                # Recompute deterministically if needed
                print("Recomputing by-clip assignment (map missing)...")
                import numpy as _np
                all_items = index_split(root, speakers, min_len=args.min_len, max_len=args.max_len)
                order = _np.arange(len(all_items))
                rng = _np.random.RandomState(args.split_seed)
                rng.shuffle(order)
                n = len(all_items)
                n_train = int(math.floor(n * args.train_ratio))
                n_val = int(math.floor(n * args.val_ratio))
                train_idx = set(order[:n_train].tolist())
                val_idx = set(order[n_train:n_train + n_val].tolist())
                def _item_hash2(it):
                    return _hash_item(it["frames"])  # noqa
                assign_map = {}
                for i in range(n):
                    h = _item_hash2(all_items[i])
                    assign_map[h] = "train" if i in train_idx else ("val" if i in val_idx else "test")
            # Build split-specific items by filtering all indexed clips
            all_items = index_split(root, speakers, min_len=args.min_len, max_len=args.max_len)
            items = []
            for it in all_items:
                h = _hash_item(it["frames"])
                if assign_map.get(h) == split_name:
                    items.append(it)
        print(f"Split '{split_name}': {len(items)} items indexed", flush=True)
        # Shard items for parallel runs if requested
        if args.num_shards > 1:
            items = [it for i, it in enumerate(items) if i % args.num_shards == args.shard]

        # Optional alignment and caching of prepared crops
        if args.align:
            if torch is None:
                print("Warning: torch not available; cannot save prepared tensors. Skipping alignment.")
            else:
                out_root = Path(args.prepared_dir) / split_name
                out_root.mkdir(parents=True, exist_ok=True)
                # Select landmark/alignment method
                mp_mesh = None
                fan = None
                lip = None
                method = args.landmarks
                if method == "lipcrop":
                    if not _LIP_OK:
                        warnings.warn("LipDetectCrop unavailable; falling back to MediaPipe/center crop.")
                        method = "mp"
                    else:
                        try:
                            # Uses GPU automatically if available (cuda:0), else CPU
                            lip = LipDetectCrop(mean_face_landmarks_path="media/20words_mean_face.npy")  # type: ignore
                            print(f"Initialized LipDetectCrop (GPU={'cuda' if torch.cuda.is_available() else 'cpu'}) for split '{split_name}'", flush=True)
                        except Exception as e:
                            warnings.warn(f"LipDetectCrop init failed ({e}); falling back to MediaPipe/center crop.")
                            lip = None
                            method = "mp"
                if method == "fan" and _FAN_OK:
                    dev = args.device
                    if dev == "cuda" and not torch.cuda.is_available():
                        warnings.warn("CUDA not available; falling back to CPU for FAN.")
                        dev = "cpu"
                    try:
                        fan = FANPredictor(device=dev)  # type: ignore
                    except Exception as e:
                        warnings.warn(f"FANPredictor init failed ({e}); falling back to MediaPipe/center crop.")
                        fan = None
                        method = "mp"
                if method == "mp" and _MP_OK:
                    try:
                        mp_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)
                        print(f"Initialized MediaPipe FaceMesh for split '{split_name}'", flush=True)
                    except Exception as e:
                        warnings.warn(f"Mediapipe FaceMesh init failed: {e}")
                        mp_mesh = None
                print(f"Processing split '{split_name}' with method '{method}'", flush=True)
                cnt_lip, cnt_bbox_fan, cnt_bbox_mp, cnt_center = 0, 0, 0, 0
                # Track failed detections if requested
                fail_list: list[dict] = []
                prepped = []
                pbar = tqdm(items, desc=f"{split_name} ({method})", unit="clip")
                for it in pbar:
                    h = _hash_item(it["frames"])
                    out_pt = out_root / it["speaker"] / it["word"] / f"{h}.pt"
                    out_pt.parent.mkdir(parents=True, exist_ok=True)
                    if args.update_only and not args.overwrite and out_pt.exists():
                        it["prepared"] = str(out_pt)
                        prepped.append(it)
                        pbar.set_postfix_str("cached")
                        continue
                    # compute if missing or overwrite requested
                    if args.overwrite or not out_pt.exists():
                        method_used = ""
                        arr = None
                        if lip is not None:
                            # LRS2-style alignment using LipDetectCrop (RetinaFace+FAN+mean-face warp)
                            try:
                                # Load frames to (T, H, W, 3) uint8
                                rgb_seq: list[np.ndarray] = []
                                for fp in it["frames"]:
                                    with Image.open(fp) as im:
                                        rgb_seq.append(np.array(im.convert("RGB")))
                                if not rgb_seq:
                                    raise RuntimeError("No frames found")
                                video_np = np.stack(rgb_seq, axis=0)
                                import torch as _torch  # local alias ensures torch is available
                                video_t = _torch.from_numpy(video_np)
                                # Detect and interpolate landmarks (exactly as LRS2)
                                lms = lip.detect_landmarks(video_t, verbose=0)
                                pre_lms = lip.landmarks_interpolate(lms)
                                if pre_lms:
                                    vcrop = lip.crop_patch(video_np, pre_lms)  # numpy in -> numpy out
                                    if vcrop is None:
                                        raise RuntimeError("crop_patch returned None")
                                    # Convert to grayscale 88x88 to match downstream normalization
                                    grays: list[np.ndarray] = []
                                    for f in vcrop:
                                        g = Image.fromarray(f).convert("L").resize((88, 88), Image.BILINEAR)
                                        grays.append(np.array(g, dtype=np.uint8))
                                    arr = np.stack(grays, axis=0)  # (T, 88, 88) uint8
                                    method_used = "lipcrop"
                                    cnt_lip += 1
                                else:
                                    # No landmarks -> delegate to fallback below
                                    arr = None
                            except Exception as e:
                                warnings.warn(f"LipDetectCrop failed ({e}); falling back to bbox-based crop.")
                                lip = None  # disable for subsequent items in this run
                                # fall through to bbox-based crop below
                        if arr is None:
                            fixed_bbox = None
                            # If primary lipcrop failed and no detectors are initialized, lazily init MediaPipe for fallback
                            if fan is None and _FAN_OK:
                                try:
                                    dev = "cuda" if (hasattr(torch, "cuda") and torch.cuda.is_available()) else "cpu"  # type: ignore
                                    fan = FANPredictor(device=dev)  # type: ignore
                                    print("Initialized FANPredictor for fallback bbox", flush=True)
                                except Exception as e:
                                    warnings.warn(f"FAN fallback init failed: {e}")
                                    fan = None
                            if fan is None and mp_mesh is None and _MP_OK:
                                try:
                                    mp_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)
                                except Exception as e:
                                    warnings.warn(f"Fallback MediaPipe init failed: {e}")
                                    mp_mesh = None
                            if args.bbox_mode == "per_clip" and (fan is not None or mp_mesh is not None):
                                # Use the middle frame for landmark detection
                                ref_idx = max(0, min(len(it["frames"]) - 1, len(it["frames"]) // 2))
                                with Image.open(it["frames"][ref_idx]) as im:
                                    im = im.convert("RGB")
                                    ref_img = np.array(im)
                                fixed_bbox = _detect_mouth_bbox_fan(ref_img, fan) if fan is not None else None
                                if fixed_bbox is None and mp_mesh is not None:
                                    fixed_bbox = _detect_mouth_bbox_mp(ref_img, mp_mesh)
                            arr = _crop_mouth_stack(
                                it["frames"], out_size=88, enlarge=args.bbox_enlarge, mp_mesh=mp_mesh, fan=fan, fixed_bbox=fixed_bbox
                            )
                            if fixed_bbox is not None and fan is not None:
                                method_used = "bbox_fan"
                                cnt_bbox_fan += 1
                            elif fixed_bbox is not None and mp_mesh is not None:
                                method_used = "bbox_mp"
                                cnt_bbox_mp += 1
                            else:
                                method_used = "center"
                                cnt_center += 1
                        # If requested, drop items where lip detection failed ('center' fallback)
                        if args.drop_on_fail and method_used == "center":
                            fail_list.append({
                                "speaker": it["speaker"],
                                "word": it["word"],
                                "T": it.get("T"),
                                "hash": h,
                                "frames": it.get("frames", [])[:3],  # sample a few for reference
                            })
                            # Do not save or include in indices
                            continue
                        # Save as uint8 grayscale stack (T, H, W)
                        torch.save({"frames": torch.from_numpy(arr), "meta": {"method": method_used}}, out_pt)
                        pbar.set_postfix_str("written")
                    it["prepared"] = str(out_pt)
                    prepped.append(it)
                items = prepped

                # Write fail list if any
                if args.drop_on_fail:
                    try:
                        fail_dir = Path(args.fail_list_dir)
                        fail_dir.mkdir(parents=True, exist_ok=True)
                        suffix = f"_{split_name}"
                        if args.num_shards > 1:
                            suffix += f"_shard{args.shard}_of_{args.num_shards}"
                        out_fail = fail_dir / f"missed_lips{suffix}.txt"
                        with out_fail.open("w", encoding="utf-8") as f:
                            for rec in fail_list:
                                f.write(f"{rec['speaker']}/{rec['word']}\t{rec.get('hash','')}\tT={rec.get('T','?')}\n")
                        print(f"Dropped (center-fallback) items for '{split_name}': {len(fail_list)} (see {out_fail})")
                    except Exception as e:
                        warnings.warn(f"Failed to write fail list: {e}")

        # Save indices: for shard runs, avoid clobber by writing shard-specific files
        if args.num_shards > 1:
            shard_idx_path = indices_dir / f"{split_name}_shard{args.shard}_of_{args.num_shards}.pt"
            save_indices(items, shard_idx_path)
            print(f"Saved shard indices: {shard_idx_path} ({len(items)} items)")
        else:
            save_indices(items, indices_dir / f"{split_name}.pt")
        # Also save a JSON preview (first 5)
        with (indices_dir / f"{split_name}_preview.json").open("w", encoding="utf-8") as f:
            json.dump(items[:5], f, ensure_ascii=False, indent=2)
        print(f"Indexed {len(items)} items for {split_name}{' (+prepared)' if args.align else ''}")
        if args.align:
            try:
                # Summaries were tracked inside the loop; recompute lightweight summary from saved metas
                methods = {"lipcrop":0,"bbox_fan":0,"bbox_mp":0,"center":0}
                from pathlib import Path as _P
                for it in items[:1000]:  # sample first 1000 for speed
                    p = it.get("prepared")
                    if not p:
                        continue
                    try:
                        d = torch.load(p, map_location="cpu")
                        m = d.get("meta", {}).get("method")
                        if m in methods:
                            methods[m]+=1
                    except Exception:
                        pass
                print(f"Sample method usage for '{split_name}': {methods}")
            except Exception:
                pass
        all_words.update([it["word"] for it in items])

    # Train char-level tokenizer from word set (if sentencepiece available)
    # Tokenizer: skip in shard runs or if explicitly requested
    if all_words and not args.skip_tokenizer and args.num_shards == 1:
        if spm is not None:
            train_char_tokenizer(sorted(all_words), Path(args.tokenizer_model), vocab_size=args.vocab_size)
            print(f"Tokenizer trained over {len(all_words)} unique words")
        else:
            print("SentencePiece not installed; skipping tokenizer training.")
    elif args.num_shards > 1 or args.skip_tokenizer:
        print("Skipping tokenizer training (use a non-sharded run to (re)train)")

    print("Done. Artifacts:")
    print(f"- Speaker lists: {splits_dir}")
    print(f"- Indices:       {indices_dir}")
    print(f"- Prepared:      {args.prepared_dir} ({'enabled' if args.align else 'disabled'})")
    print(f"- Tokenizer:     {args.tokenizer_model} {'(skipped)' if spm is None else ''}")


if __name__ == "__main__":
    main()
