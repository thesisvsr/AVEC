#!/usr/bin/env python3
"""Robust LipBengal word-level inference.

Pipeline:
 1. Load model + checkpoint (explicit or auto-last from callbacks dir).
 2. Decode input video (RGB) with torchvision.
 3. Multi-scale mouth ROI detection using FAN:
       - Original resolution
       - Resize width -> 640
       - Resize width -> 320
    For each scale attempt FAN landmarks. Accept if >= 68 points.
 4. If no full 68-point detection succeeded:
       - If we ever saw > 20 landmarks (partial), build bbox from those points.
       - Else fallback to heuristic lower-face box (x: 15%-85%, y: 55%-95%).
 5. Derive mouth bounding box from landmark mouth indices (48:68), enlarge by 1.4x, clamp to frame.
 6. Crop all frames; convert to grayscale; resize to 96x96.
 7. Temporal normalize: pad or sub-sample to fixed 29 frames (match training config).
 8. Normalize intensities ((x/255 - 0.5)/0.5) and run model forward.
 9. Output Top-K predictions (default K=5) with probabilities.
10. (Optional) Save montage & GIF of processed frames and JSON output.

Usage:
  python scripts/infer_lipbengal.py --video callbacks/LipBengal/AV/test1.mp4 \
     --checkpoint callbacks/LipBengal/AV/VisualCE/checkpoints_epoch_338_step_0.ckpt \
     --save_montage --output_dir export/infer_samples

If --checkpoint is omitted and --callbacks_dir is provided, the latest valid checkpoint
(checkpoints_epoch_*_step_0.ckpt) will be chosen (skipping obviously corrupted zero/short files).

Requirements: ibug.face_alignment (FANPredictor) installed; torchvision, Pillow.
"""
from __future__ import annotations

import argparse, os, sys, json, math, glob, hashlib, traceback
from pathlib import Path
from typing import Optional, Tuple, List

import torch
import numpy as np
from PIL import Image
import torchvision

try:
    from ibug.face_alignment import FANPredictor  # type: ignore
except Exception as e:  # pragma: no cover
    FANPredictor = None  # type: ignore

# Add repo root to path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import importlib
import functions  # for find_last_checkpoint if needed

MOUTH_IDXS = list(range(48, 68))  # 20 mouth landmark indices (68-pt format)
EYE_IDXS = list(range(36, 48))    # eye + brow vicinity for orientation heuristic


def parse_args():
    ap = argparse.ArgumentParser(description="LipBengal inference")
    ap.add_argument('--video', required=True, help='Input video path')
    ap.add_argument('--checkpoint', help='Checkpoint .ckpt path (overrides auto)')
    ap.add_argument('--callbacks_dir', default='callbacks/LipBengal/AV/VisualCE', help='Directory containing checkpoints (used if --checkpoint not given)')
    ap.add_argument('--config', default='configs.LipBengal.AV.VisualCE', help='Config module for model + vocab')
    ap.add_argument('--device', default='cuda', help='Device (cuda or cpu)')
    ap.add_argument('--topk', type=int, default=5, help='Top-K predictions to display')
    ap.add_argument('--fixed_frames', type=int, default=29, help='Fixed temporal length')
    ap.add_argument('--enlarge', type=float, default=1.4, help='Enlargement factor for mouth bbox')
    ap.add_argument('--save_montage', action='store_true', help='Save montage PNG of processed frames')
    ap.add_argument('--save_gif', action='store_true', help='Save animated GIF of processed frames')
    ap.add_argument('--output_dir', default='export/infer_outputs', help='Directory to write outputs (montage/json)')
    ap.add_argument('--json', action='store_true', help='Save predictions JSON')
    ap.add_argument('--verbose', action='store_true', help='Verbose debug logging')
    return ap.parse_args()


def log(msg: str, verbose: bool):
    if verbose:
        print(msg)


def load_model(config_mod: str, checkpoint: str, device: str, verbose: bool):
    cfg = importlib.import_module(config_mod.replace('.py','').replace('/', '.'))
    model = cfg.model  # already instantiated in config
    words = cfg.training_dataset.classes  # class list
    # Load checkpoint
    ckpt = torch.load(checkpoint, map_location='cpu')
    state = ckpt.get('model_state_dict', ckpt)
    # Filter load (some keys may mismatch if architecture changed)
    current = model.state_dict()
    filtered = {k: v for k, v in state.items() if k in current and getattr(v, 'shape', None) == current[k].shape}
    missing = set(current.keys()) - set(filtered.keys())
    if verbose:
        print(f"Loaded {len(filtered)}/{len(current)} tensors from checkpoint; missing {len(missing)}")
    model.load_state_dict(filtered, strict=False)
    model.eval()
    dev = torch.device(device if device == 'cpu' or (torch.cuda.is_available() and 'cuda' in device) else 'cpu')
    model.to(dev)
    return model, words, dev


def pick_checkpoint(callbacks_dir: str, verbose: bool) -> Optional[str]:
    # Use provided functions.find_last_checkpoint to get candidate then verify load.
    cb = Path(callbacks_dir)
    if not cb.is_dir():
        return None
    candidate = functions.find_last_checkpoint(str(cb))
    if candidate is None:
        return None
    full = cb / candidate
    # Validate non-trivial size ( > 10MB )
    try:
        if full.stat().st_size < 10 * 1024 * 1024:
            if verbose:
                print(f"Checkpoint {full} too small; skipping")
            return None
    except Exception:
        return None
    # Try a light load (CPU) just to ensure not corrupted
    try:
        _ = torch.load(str(full), map_location='cpu')
        return str(full)
    except Exception as e:
        if verbose:
            print(f"Failed to load candidate {full}: {e}")
        # Fallback: search all checkpoints newest backwards
        ckpts = sorted(cb.glob('checkpoints_epoch_*_step_0.ckpt'), key=lambda p: int(p.name.split('_')[2]), reverse=True)
        for p in ckpts:
            try:
                if p.stat().st_size < 10 * 1024 * 1024:
                    continue
                torch.load(str(p), map_location='cpu')
                return str(p)
            except Exception:
                continue
    return None


def fan_landmarks(frame: np.ndarray, fan) -> Optional[np.ndarray]:
    try:
        out = fan(frame)
        if not out:
            return None
        # out may be (list, meta) or just list
        if isinstance(out, tuple):
            lm_list = out[0]
        else:
            lm_list = out
        if not lm_list:
            return None
        pts = np.array(lm_list[0], dtype=np.float32)
        return pts  # shape (68,2) expected
    except Exception:
        return None


def detect_mouth_bbox(frame: np.ndarray, fan, enlarge: float, verbose: bool) -> Tuple[int,int,int,int,str]:
    H, W = frame.shape[:2]
    attempted_partial = None  # store partial landmark set
    scales = [None, 640, 320]  # original, width=640, width=320
    for scale in scales:
        if scale is None:
            f_proc = frame
            s = 1.0
        else:
            s = scale / W
            f_proc = np.array(Image.fromarray(frame).resize((scale, int(H * s)), Image.BILINEAR))
        pts = fan_landmarks(f_proc, fan)
        if pts is None:
            continue
        if pts.shape[0] >= 68:
            # rescale back
            pts[:, 0] /= s
            pts[:, 1] /= s
            mouth = pts[MOUTH_IDXS]
            x0, y0 = mouth.min(axis=0)
            x1, y1 = mouth.max(axis=0)
            if x1 > x0 and y1 > y0:
                # Enlarge
                cx = (x0 + x1) / 2
                cy = (y0 + y1) / 2
                bw = (x1 - x0) * enlarge
                bh = (y1 - y0) * enlarge
                x0 = int(max(0, cx - bw / 2)); x1 = int(min(W, cx + bw / 2))
                y0 = int(max(0, cy - bh / 2)); y1 = int(min(H, cy + bh / 2))
                return x0, y0, x1, y1, f"fan_scale_{'orig' if scale is None else scale}"    
        elif pts.shape[0] > 20 and attempted_partial is None:
            attempted_partial = (pts, s)
    # Partial fallback
    if attempted_partial is not None:
        pts, s = attempted_partial
        pts[:, 0] /= s
        pts[:, 1] /= s
        x0, y0 = pts[:,0].min(), pts[:,1].min()
        x1, y1 = pts[:,0].max(), pts[:,1].max()
        if x1 > x0 and y1 > y0:
            cx = (x0 + x1) / 2; cy = (y0 + y1) / 2
            bw = (x1 - x0) * enlarge; bh = (y1 - y0) * enlarge
            x0 = int(max(0, cx - bw / 2)); x1 = int(min(W, cx + bw / 2))
            y0 = int(max(0, cy - bh / 2)); y1 = int(min(H, cy + bh / 2))
            return x0, y0, x1, y1, 'partial_fallback'
    # Heuristic lower-face box
    x0 = int(W * 0.15); x1 = int(W * 0.85)
    y0 = int(H * 0.55); y1 = int(H * 0.95)
    return x0, y0, x1, y1, 'heuristic_lower_face'


def attempt_full_landmarks(frame: np.ndarray, fan, verbose: bool) -> Optional[np.ndarray]:
    """Try multi-scale landmark detection returning full 68 landmarks if successful.
    Does NOT enlarge or crop. Returns landmarks in original frame coords or None."""
    H, W = frame.shape[:2]
    scales = [None, 640, 320]
    for scale in scales:
        if scale is None:
            f_proc = frame; s = 1.0
        else:
            s = scale / W
            f_proc = np.array(Image.fromarray(frame).resize((scale, int(H * s)), Image.BILINEAR))
        pts = fan_landmarks(f_proc, fan)
        if pts is not None and pts.shape[0] >= 68:
            pts[:,0] /= s; pts[:,1] /= s
            return pts
    return None


def maybe_correct_orientation(video: np.ndarray, fan, verbose: bool) -> Tuple[np.ndarray, str]:
    """Detect if video appears upside-down and rotate 180° if so.

    Heuristics:
      1. Attempt landmarks; if found, compare mean mouth y vs mean eye/brow y.
         Expect mouth center > eye center. If mouth_center_y < eye_center_y, treat as upside-down.
      2. If no landmarks found, try 180° rotation and landmarks again; if rotated succeeds and original fails, keep rotated.
    Returns (possibly-rotated-video, orientation_flag)."""
    T,H,W,C = video.shape
    mid = T // 2
    frame = video[mid]
    pts = attempt_full_landmarks(frame, fan, verbose)
    if pts is not None:
        mouth_cy = pts[MOUTH_IDXS,1].mean()
        eye_cy = pts[EYE_IDXS,1].mean()
        if mouth_cy < eye_cy:  # inverted
            if verbose:
                print(f"Orientation: upside_down detected (mouth_cy {mouth_cy:.1f} < eye_cy {eye_cy:.1f}); rotating 180°")
            video = np.ascontiguousarray(video[:, ::-1, ::-1, :])  # rotate 180
            return video, 'rotated_180'
        return video, 'upright'
    # Try rotated if initial failed
    video_rot = np.ascontiguousarray(video[:, ::-1, ::-1, :])
    frame_r = video_rot[mid]
    pts_r = attempt_full_landmarks(frame_r, fan, verbose)
    if pts_r is not None:
        mouth_cy = pts_r[MOUTH_IDXS,1].mean(); eye_cy = pts_r[EYE_IDXS,1].mean()
        if mouth_cy < eye_cy:
            # Even rotated variant still inverted per heuristic; keep rotated anyway if it produced landmarks.
            if verbose:
                print("Orientation: only rotated version produced landmarks (still flagged inverted heuristically); using rotated.")
            return video_rot, 'rotated_180'
        if verbose:
            print("Orientation: original failed; rotated version succeeded; using rotated.")
        return video_rot, 'rotated_180'
    # Both failed; leave as-is
    if verbose:
        print("Orientation: landmarks unavailable; keeping original orientation.")
    return video, 'unknown'


def build_montage(frames: np.ndarray, cols: int = 10) -> np.ndarray:
    T, H, W = frames.shape[:3]
    cols = min(cols, T)
    rows = math.ceil(T / cols)
    canvas = np.zeros((rows * H, cols * W), dtype=frames.dtype)
    for i in range(T):
        r = i // cols
        c = i % cols
        canvas[r * H:(r + 1) * H, c * W:(c + 1) * W] = frames[i]
    return canvas


def main():
    args = parse_args()
    video_path = Path(args.video)
    assert video_path.is_file(), f"Video not found: {video_path}"

    # Resolve checkpoint
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
    else:
        ckpt_auto = pick_checkpoint(args.callbacks_dir, args.verbose)
        if ckpt_auto is None:
            raise SystemExit('No valid checkpoint found.')
        ckpt_path = Path(ckpt_auto)
    assert ckpt_path.is_file(), f"Checkpoint not found: {ckpt_path}"

    model, words, device = load_model(args.config, str(ckpt_path), args.device, args.verbose)

    # Load video
    vid, _, _ = torchvision.io.read_video(str(video_path), pts_unit='sec')  # (T,H,W,C)
    if vid.numel() == 0:
        raise SystemExit('Empty video after decode')
    vid_np = vid.numpy()
    T, H, W, C = vid_np.shape
    mid = T // 2

    # Instantiate FAN
    if FANPredictor is None:
        raise SystemExit('FANPredictor not available (ibug.face_alignment not installed).')
    # FANPredictor expects a raw string ("cuda" / "cpu"), not torch.device
    fan_device = 'cuda' if 'cuda' in str(device) else 'cpu'
    fan = FANPredictor(device=fan_device)  # type: ignore

    # Orientation correction before bbox detection
    vid_np, orientation = maybe_correct_orientation(vid_np, fan, args.verbose)
    T, H, W, C = vid_np.shape  # refresh (unchanged dims)
    mid = T // 2
    # Detect mouth bbox on (possibly rotated) middle frame
    x0, y0, x1, y1, method = detect_mouth_bbox(vid_np[mid], fan, args.enlarge, args.verbose)
    if args.verbose:
        print(f"Detection method: {method} bbox=({x0},{y0},{x1},{y1})")

    # Crop + preprocess frames
    crops = []
    for f in vid_np:
        crop = f[y0:y1, x0:x1]
        im = Image.fromarray(crop).convert('L').resize((96, 96), Image.BILINEAR)
        arr = np.array(im, dtype=np.uint8)
        crops.append(arr)
    arr = np.stack(crops, axis=0)  # (T,96,96)

    # Temporal normalization
    fixed = args.fixed_frames
    if T > fixed:
        idx = torch.linspace(0, T - 1, steps=fixed).round().long().numpy()
        arr = arr[idx]
    elif T < fixed:
        pad = np.repeat(arr[-1:, :, :], fixed - T, axis=0)
        arr = np.concatenate([arr, pad], axis=0)

    # (B,C,T,H,W)
    x = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).float() / 255.0
    x = (x - 0.5) / 0.5
    x = x.to(device)

    with torch.no_grad():
        # model.encoder returns (B,T,Dim) under key [0]; we mimic training pooling.
        logits = model.encoder(x, lengths=None)[0].mean(dim=1)
        probs = torch.softmax(logits, dim=-1)
        topk = probs.topk(min(args.topk, probs.shape[-1]))

    predictions = []
    for p, i in zip(topk.values[0].tolist(), topk.indices[0].tolist()):
        predictions.append({'word': words[i], 'prob': float(p)})

    print('Top-{} Predictions:'.format(len(predictions)))
    for r, pred in enumerate(predictions, 1):
        print(f"Top{r}: {pred['word']} prob={pred['prob']:.4f}")

    # Outputs
    out_dir = Path(args.output_dir)
    if args.save_montage or args.save_gif or args.json:
        out_dir.mkdir(parents=True, exist_ok=True)
    base = video_path.stem

    if args.save_montage:
        montage = build_montage(arr)
        Image.fromarray(montage).save(out_dir / f"{base}_montage.png")
    if args.save_gif:
        try:
            import imageio
            imageio.mimsave(out_dir / f"{base}.gif", arr, fps=15)
        except Exception:
            print('GIF save skipped (imageio not installed or failed)')
    if args.json:
        meta = {
            'video': str(video_path),
            'checkpoint': str(ckpt_path),
            'detection_method': method,
            'orientation': orientation,
            'bbox': [int(x0), int(y0), int(x1), int(y1)],
            'topk': predictions,
            'num_frames_original': int(T),
            'fixed_frames': fixed,
        }
        (out_dir / f"{base}_predictions.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Inference interrupted.')
    except Exception as e:
        print('Error during inference:', e)
        if '--verbose' in sys.argv:
            traceback.print_exc()
        sys.exit(1)
