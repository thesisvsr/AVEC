#!/usr/bin/env python3
"""
Prepare the BSRD dataset for visual CTC training.

Inputs (expected layout):
    datasets/BSRD/
        Face_Only_Videos_with_Audio/    video<speaker>_<clip>.mp4  (face crops with audio embedded)
        First_Transcriptions/           video<speaker>_<clip>.txt  (Bengali + word timings)
        Transliteration_Informal/       video<speaker>_<clip>.txt  (single-line romanized transliteration)

Outputs:
    datasets/BSRD/indices/{train,val,test}.pt      # list[dict]
        item keys:
          speaker (str, numeric id as string)
          clip_id (str)
          id (str, e.g., video75_0095)
          video_path (str, mp4 path)
          bn_text (str, Bengali full sentence)
          rom_text (str or None)
          words (list[{w,start,end}])
          T (int frames after decoding or after truncation)
          prepared (str, optional path to cached tensor)
    datasets/BSRD/vocab/char_vocab.json            # ordered list of characters (if not skipped)
    datasets/BSRD/vocab/char2idx.json

Optional prepared tensors (.pt) when --align / --prepare invoked:
    datasets/BSRD/prepared/<split>/<speaker>/<id>_<hash>.pt
        { 'frames': uint8 tensor (T, H, W), 'meta': {...} }

We avoid writing intermediate extracted jpg frames to save disk: we decode video directly with torchvision.io.read_video.

Usage examples:
    python scripts/prepare_bsrd.py --root datasets/BSRD --prepare --align --landmarks mp
    python scripts/prepare_bsrd.py --root datasets/BSRD --splits train,val --subset_fraction 0.1

Notes:
  - Alignment modes mirror LipBengal prepare script (mp, fan, lipcrop, none); defaults to mp (mediapipe if installed).
  - We cap frames with --max_frames (default 150) for memory consistency.
  - CTC char vocab built from transliteration (rom_text) across train split only (default) unless --vocab_all_splits.
"""

from __future__ import annotations

import os, re, json, argparse, math, hashlib, warnings
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np

try:
    import torch
except Exception:
    torch = None  # type: ignore

try:
    from PIL import Image
except Exception as e:
    raise RuntimeError("PIL (Pillow) required: pip install pillow") from e

try:
    import cv2  # type: ignore
    _CV2_OK = True
except Exception:
    cv2 = None  # type: ignore
    _CV2_OK = False

# Optional mediapipe and FAN (same pattern as LipBengal)
try:
    import mediapipe as mp  # type: ignore
    _MP_OK = True
except Exception:
    mp = None  # type: ignore
    _MP_OK = False
try:
    from ibug.face_alignment import FANPredictor  # type: ignore
    _FAN_OK = True
except Exception:
    FANPredictor = None  # type: ignore
    _FAN_OK = False

try:
    from nnet.transforms import LipDetectCrop  # type: ignore
    _LIP_OK = True
except Exception:
    LipDetectCrop = None  # type: ignore
    _LIP_OK = False

import torchvision

# ----------------- Utility -----------------

VID_RE = re.compile(r"video(\d+)_(\d+)\.mp4$")
TXT_RE = re.compile(r"video(\d+)_(\d+)\.txt$")

def parse_transcript_bn(path: Path) -> tuple[str, List[Dict]]:
    """Parse Bengali transcript with word-level timings.
    Returns (sentence, word_list) where word_list entries: {w,start,end} (seconds).
    """
    txt = path.read_text(encoding="utf-8", errors="ignore").strip().splitlines()
    if not txt:
        return "", []
    sentence = txt[0].strip()
    words = []
    for line in txt[1:]:
        line = line.strip()
        # Expected pattern: Word: TOKEN, Start: Xs, End: Ys
        if not line.startswith("Word:"):
            continue
        try:
            parts = line.split(',')
            w_part = parts[0]
            s_part = parts[1]
            e_part = parts[2]
            w = w_part.split(':', 1)[1].strip()
            start = float(s_part.split(':', 1)[1].strip().rstrip('s'))
            end = float(e_part.split(':', 1)[1].strip().rstrip('s'))
            if end < start:
                continue
            words.append({"w": w, "start": start, "end": end})
        except Exception:
            continue
    return sentence, words

def read_transliteration(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="ignore").strip()
    except Exception:
        return ""

def hash_id(*parts: str) -> str:
    h = hashlib.sha1()
    for p in parts:
        h.update(p.encode('utf-8'))
    return h.hexdigest()[:16]

def mouth_bbox_mp(img_rgb, mp_mesh):
    try:
        h, w, _ = img_rgb.shape
        res = mp_mesh.process(img_rgb)
        if not res.multi_face_landmarks:
            return None
        lm = res.multi_face_landmarks[0].landmark
        lip_idx = [61,146,91,181,84,17,314,405,321,375,78,191,80,81,82,13,312,311,310,415]
        xs = [int(lm[i].x * w) for i in lip_idx]
        ys = [int(lm[i].y * h) for i in lip_idx]
        x0,y0,x1,y1 = min(xs),min(ys),max(xs),max(ys)
        if x1<=x0 or y1<=y0:
            return None
        return x0,y0,x1,y1
    except Exception:
        return None

def mouth_bbox_fan(img_rgb, fan):
    try:
        out = fan(img_rgb)
        if not out:
            return None
        pts = out[0] if isinstance(out, (list, tuple)) else out
        pts = np.array(pts, dtype=np.float32)
        if pts.shape[0] < 68:
            return None
        mouth = pts[48:68]
        x0,y0 = mouth.min(axis=0)
        x1,y1 = mouth.max(axis=0)
        if x1<=x0 or y1<=y0:
            return None
        return int(x0), int(y0), int(x1), int(y1)
    except Exception:
        return None

def crop_mouth_stack_video(frames: List[np.ndarray], out_size=88, enlarge=1.4, mp_mesh=None, fan=None, bbox_mode="per_clip"):
    bbox = None
    if bbox_mode == "per_clip":
        mid = len(frames)//2
        ref = frames[mid]
        if fan is not None:
            bbox = mouth_bbox_fan(ref, fan)
        if bbox is None and mp_mesh is not None:
            bbox = mouth_bbox_mp(ref, mp_mesh)
    crops = []
    for img in frames:
        cb = bbox
        if cb is None and bbox_mode == "per_frame":
            if fan is not None:
                cb = mouth_bbox_fan(img, fan)
            if cb is None and mp_mesh is not None:
                cb = mouth_bbox_mp(img, mp_mesh)
        if cb is None:
            H,W,_ = img.shape
            side = min(H,W)
            cy,cx = H//2, W//2
            half = side//2
            y0,y1 = cy-half, cy+half
            x0,x1 = cx-half, cx+half
        else:
            x0,y0,x1,y1 = cb
            cx = (x0+x1)/2; cy=(y0+y1)/2
            bw=(x1-x0)*enlarge; bh=(y1-y0)*enlarge
            x0=int(max(0,cx-bw/2)); x1=int(cx+bw/2)
            y0=int(max(0,cy-bh/2)); y1=int(cy+bh/2)
        roi = img[y0:y1, x0:x1]
        if _CV2_OK:
            gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
            crop = cv2.resize(gray, (out_size,out_size), interpolation=cv2.INTER_AREA)
        else:
            crop = Image.fromarray(roi).convert("L").resize((out_size,out_size), Image.BILINEAR)
            crop = np.array(crop, dtype=np.uint8)
        crops.append(crop)
    if not crops:
        crops = [np.zeros((out_size,out_size), dtype=np.uint8)]
    return np.stack(crops, axis=0)

def build_splits(speakers: List[str], train_ratio=0.6, val_ratio=0.2):
    speakers_sorted = sorted(speakers, key=lambda x: int(x))
    n = len(speakers_sorted)
    n_train = int(math.floor(n*train_ratio))
    n_val = int(math.floor(n*val_ratio))
    return {
        'train': speakers_sorted[:n_train],
        'val': speakers_sorted[n_train:n_train+n_val],
        'test': speakers_sorted[n_train+n_val:]
    }

def collect_videos(video_dir: Path) -> Dict[str, Path]:
    """Collect videos mapping canonical stem 'video<speaker>_<clip>' to path.

    Some distributions prepend extra tokens (e.g. 'vidAud_face_video100_0011.mp4').
    We search the filename for the canonical pattern instead of requiring it to
    start at position 0. This lets transcript stems (video100_0011.txt) match
    the discovered video entries.
    """
    out = {}
    for p in video_dir.iterdir():
        if p.suffix.lower() != '.mp4':
            continue
        m = VID_RE.search(p.name)  # allow pattern anywhere in filename
        if not m:
            continue
        # Canonical stem is the matched substring without extension
        span_start, span_end = m.span()
        canonical = p.name[span_start:span_end].rsplit('.', 1)[0]
        out[canonical] = p
    if not out:
        print(f"Warning: no video files matched pattern in {video_dir}")
    return out

def decode_video_frames(path: Path, max_frames: Optional[int]=None) -> List[np.ndarray]:
    # torchvision returns (T, H, W, C) uint8
    try:
        vframes, _, _ = torchvision.io.read_video(str(path), pts_unit='sec')
    except Exception as e:
        raise RuntimeError(f"Failed to read video {path}: {e}")
    arr = vframes.numpy()  # (T,H,W,C)
    if max_frames and arr.shape[0] > max_frames:
        arr = arr[:max_frames]
    return [arr[i] for i in range(arr.shape[0])]

def build_char_vocab(texts: List[str]) -> List[str]:
    chars = set()
    for t in texts:
        for ch in t:
            chars.add(ch)
    ordered = sorted(chars)
    return ordered

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='datasets/BSRD')
    ap.add_argument('--train_ratio', type=float, default=0.6)
    ap.add_argument('--val_ratio', type=float, default=0.2)
    ap.add_argument('--splits', default=None, help="Comma sep subset of splits to process (default all)")
    ap.add_argument('--min_frames', type=int, default=6)
    ap.add_argument('--max_frames', type=int, default=150)
    ap.add_argument('--align', action='store_true', help='Perform mouth detection + caching prepared tensors')
    ap.add_argument('--bbox_mode', choices=['per_clip','per_frame'], default='per_clip')
    ap.add_argument('--landmarks', choices=['mp','fan','lipcrop','none'], default='mp')
    ap.add_argument('--prepared_dir', default='datasets/BSRD/prepared')
    ap.add_argument('--indices_dir', default='datasets/BSRD/indices')
    ap.add_argument('--subset_fraction', type=float, default=1.0)
    ap.add_argument('--subset_seed', type=int, default=0)
    ap.add_argument('--vocab_dir', default='datasets/BSRD/vocab')
    ap.add_argument('--skip_vocab', action='store_true')
    ap.add_argument('--vocab_all_splits', action='store_true')
    ap.add_argument('--prepare', action='store_true', help='Alias for --align (legacy naming)')
    ap.add_argument('--overwrite', action='store_true')
    args = ap.parse_args()

    if args.prepare:
        args.align = True

    root = Path(args.root)
    video_dir = root / 'Face_Only_Videos_with_Audio'
    bn_dir = root / 'First_Transcriptions'
    rom_dir = root / 'Transliteration_Informal'
    for d in (video_dir, bn_dir, rom_dir):
        assert d.is_dir(), f"Missing directory: {d}"

    video_map = collect_videos(video_dir)
    # Extract speakers from video stems
    speakers = set()
    for stem in video_map.keys():
        m = re.match(r'video(\d+)_', stem)
        if m:
            speakers.add(m.group(1))
    splits = build_splits(list(speakers), args.train_ratio, args.val_ratio)
    wanted = set(splits.keys()) if not args.splits else {s.strip() for s in args.splits.split(',') if s.strip()}

    indices_dir = Path(args.indices_dir); indices_dir.mkdir(parents=True, exist_ok=True)
    prepared_root = Path(args.prepared_dir)
    vocab_text_accum = []

    # Landmark initializations (lazy inside loop for fan/lipcrop due to cost)
    mp_mesh = None
    fan = None
    lip = None
    method = args.landmarks
    if args.align:
        if method == 'lipcrop' and not _LIP_OK:
            warnings.warn('lipcrop requested but unavailable -> mp fallback')
            method = 'mp'
        if method == 'fan' and not _FAN_OK:
            warnings.warn('FAN unavailable -> mp fallback')
            method = 'mp'
        if method == 'mp' and not _MP_OK:
            warnings.warn('MediaPipe unavailable -> center crop')
            method = 'none'
        if method == 'mp':
            try:
                mp_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)  # type: ignore
                print('Initialized MediaPipe FaceMesh')
            except Exception as e:
                warnings.warn(f'MediaPipe init failed: {e}')
                mp_mesh = None
        if method == 'fan':
            try:
                dev = 'cuda' if (hasattr(torch, 'cuda') and torch.cuda.is_available()) else 'cpu'
                fan = FANPredictor(device=dev)  # type: ignore
                print(f'Initialized FANPredictor ({dev})')
            except Exception as e:
                warnings.warn(f'FAN init failed: {e}')
                fan = None
        if method == 'lipcrop':
            try:
                lip = LipDetectCrop(mean_face_landmarks_path='media/20words_mean_face.npy')  # type: ignore
                print('Initialized LipDetectCrop')
            except Exception as e:
                warnings.warn(f'LipDetectCrop init failed: {e}; fallback center')
                lip = None
                method = 'none'

    rng = np.random.RandomState(args.subset_seed)

    for split_name, spk_list in splits.items():
        if split_name not in wanted:
            continue
        # Collect stems for speakers in this split
        stems = [stem for stem in video_map.keys() if any(stem.startswith(f'video{sp}_') for sp in spk_list)]
        # Optional subset
        if args.subset_fraction < 1.0:
            total = len(stems)
            keep = max(1, int(round(total * args.subset_fraction)))
            if keep < total:
                idx = rng.permutation(total)[:keep]
                stems = [stems[i] for i in idx]
                print(f"Split {split_name}: subset {keep}/{total} (~{keep/total*100:.1f}%)")
        items = []
        for stem in stems:
            vid_path = video_map[stem]
            m = re.match(r'video(\d+)_(\d+)', stem)
            if not m:
                continue
            speaker = m.group(1)
            clip_id = m.group(2)
            bn_path = bn_dir / f"{stem}.txt"
            rom_path = rom_dir / f"{stem}.txt"
            if not bn_path.is_file():
                continue
            bn_text, words = parse_transcript_bn(bn_path)
            rom_text = read_transliteration(rom_path) if rom_path.is_file() else ""
            # Decode video
            try:
                frames = decode_video_frames(vid_path, max_frames=args.max_frames)
            except Exception as e:
                warnings.warn(str(e))
                continue
            if len(frames) < args.min_frames:
                continue
            # Align & crop if requested
            prepared_path = None
            if args.align:
                # lipcrop path uses different API: treat separately
                if lip is not None:
                    try:
                        vnp = np.stack(frames, axis=0)  # (T,H,W,3)
                        import torch as _torch
                        vt = _torch.from_numpy(vnp)
                        lms = lip.detect_landmarks(vt, verbose=0)
                        pre_lms = lip.landmarks_interpolate(lms)
                        if pre_lms:
                            vcrop = lip.crop_patch(vnp, pre_lms)
                            grays = []
                            for f in vcrop:
                                g = Image.fromarray(f).convert('L').resize((88,88), Image.BILINEAR)
                                grays.append(np.array(g, dtype=np.uint8))
                            stack = np.stack(grays, axis=0)
                        else:
                            stack = crop_mouth_stack_video(frames, mp_mesh=mp_mesh, fan=fan, bbox_mode=args.bbox_mode)
                        method_used = 'lipcrop'
                    except Exception as e:
                        warnings.warn(f'lipcrop failed {e}; fallback bbox/center')
                        stack = crop_mouth_stack_video(frames, mp_mesh=mp_mesh, fan=fan, bbox_mode=args.bbox_mode)
                        method_used = 'fallback'
                else:
                    stack = crop_mouth_stack_video(frames, mp_mesh=mp_mesh, fan=fan, bbox_mode=args.bbox_mode)
                    method_used = 'bbox_or_center'
                digest = hash_id(str(vid_path), str(len(frames)))
                prepared_path = prepared_root / split_name / speaker / f"{stem}_{digest}.pt"
                prepared_path.parent.mkdir(parents=True, exist_ok=True)
                if (not prepared_path.exists()) or args.overwrite:
                    if torch is not None:
                        torch.save({'frames': torch.from_numpy(stack), 'meta': {'method': method_used}}, prepared_path)
                    else:
                        np.save(str(prepared_path.with_suffix('.npy')), stack)
            item = {
                'speaker': speaker,
                'clip_id': clip_id,
                'id': stem,
                'video_path': str(vid_path),
                'bn_text': bn_text,
                'rom_text': rom_text,
                'words': words,
                'T': min(len(frames), args.max_frames),
            }
            if prepared_path is not None:
                item['prepared'] = str(prepared_path)
            items.append(item)
        out_pt = indices_dir / f"{split_name}.pt"
        if torch is not None:
            torch.save(items, out_pt)
        else:
            out_pt.with_suffix('.json').write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding='utf-8')
        print(f"Saved split {split_name}: {len(items)} items -> {out_pt}")
        # Accumulate vocab texts if building across this split (train only unless vocab_all_splits)
        if not args.skip_vocab and (split_name == 'train' or args.vocab_all_splits):
            vocab_text_accum.extend([it['rom_text'] for it in items if it.get('rom_text')])

    if not args.skip_vocab and vocab_text_accum:
        vocab_dir = Path(args.vocab_dir)
        vocab_dir.mkdir(parents=True, exist_ok=True)
        vocab_list = build_char_vocab(vocab_text_accum)
        # Add CTC blank token symbol if desired (we leave blank as index 0 reserved externally typically)
        with (vocab_dir / 'char_vocab.json').open('w', encoding='utf-8') as f:
            json.dump(vocab_list, f, ensure_ascii=False, indent=2)
        char2idx = {c: i+1 for i, c in enumerate(vocab_list)}  # reserve 0 for CTC blank
        with (vocab_dir / 'char2idx.json').open('w', encoding='utf-8') as f:
            json.dump(char2idx, f, ensure_ascii=False, indent=2)
        print(f"Saved vocab: {len(vocab_list)} chars (blank assumed at index 0)")

    print("Done.")

if __name__ == '__main__':
    main()
