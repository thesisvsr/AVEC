#!/usr/bin/env python3
"""Preprocess LRW-B Bengali word-level dataset.

Goals (mirrors LRW-AR pipeline but adapted to flat file layout):
 1. Speaker-based split (default):
       train: p1,p2,p5,p6   val: p4   test: p3
 2. Detect & crop mouth region (MediaPipe or FAN) to 96x96 grayscale.
 3. Save per-clip tensor: datasets/LRW-B/prepared/{split}/{WORD}/{sha}.pt
       {frames: uint8[T,H,W], meta:{method, word}}
 4. Build indices: datasets/LRW-B/indices/{split}.pt  (list of {word, prepared, T})
 5. (Optional) drop clips when detection fails (no center fallback if --drop_on_fail set).

IMPORTANT: Unlike the LRW-AR script, we intentionally do NOT center-fallback when
--drop_on_fail is specified (user requested only lip-region crops). If detection fails
and fallback is disabled, the clip is skipped.

We assume file naming pattern: p{speakerIndex}_{WORD}.mp4 under datasets/LRW-B/Data.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Tuple, Optional
import argparse, hashlib, os, json, re, warnings, multiprocessing as mp_proc

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore
try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None  # type: ignore
try:
    import torchvision
except Exception:  # pragma: no cover
    torchvision = None  # type: ignore
try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None  # type: ignore
try:
    import mediapipe as mp  # type: ignore
    _MP_OK = True
except Exception:  # pragma: no cover
    mp = None  # type: ignore
    _MP_OK = False
try:
    from ibug.face_alignment import FANPredictor  # type: ignore
    _FAN_OK = True
except Exception:  # pragma: no cover
    FANPredictor = None  # type: ignore
    _FAN_OK = False


# --------------------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------------------

def sha1_list(strs: List[str]) -> str:
    h = hashlib.sha1()
    for s in strs:
        h.update(s.encode("utf-8"))
    return h.hexdigest()[:16]


def detect_bbox_fan(img_rgb, fan) -> Optional[Tuple[int, int, int, int]]:
    if fan is None:
        return None
    try:
        out = fan(img_rgb)
        if out is None:
            return None
        if isinstance(out, tuple) and len(out) >= 1:
            lms_list = out[0]
        else:
            lms_list = out
        if not lms_list:
            return None
        pts = lms_list[0]
        import numpy as _np
        pts = _np.array(pts, dtype=_np.float32)
        if pts.shape[0] < 68:
            return None
        mouth = pts[48:68, :]
        x0, y0 = mouth.min(axis=0)
        x1, y1 = mouth.max(axis=0)
        if x1 <= x0 or y1 <= y0:
            return None
        return int(x0), int(y0), int(x1), int(y1)
    except Exception:
        return None


def detect_bbox_mp(img_rgb, mp_mesh, mp_face_det=None) -> Optional[Tuple[int, int, int, int]]:
    # FaceMesh path
    if mp_mesh is not None:
        try:
            h, w, _ = img_rgb.shape
            res = mp_mesh.process(img_rgb)
            if res.multi_face_landmarks:
                lm = res.multi_face_landmarks[0].landmark
                lip_idx = [61,146,91,181,84,17,314,405,321,375,78,191,80,81,82,13,312,311,310,415]
                xs = [int(lm[i].x * w) for i in lip_idx]
                ys = [int(lm[i].y * h) for i in lip_idx]
                x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
                if x1 > x0 and y1 > y0:
                    return x0, y0, x1, y1
        except Exception:
            pass
    # Fallback: approximate mouth via face detection lower region
    if mp_face_det is not None:
        try:
            h, w, _ = img_rgb.shape
            res = mp_face_det.process(img_rgb)
            if res.detections:
                box = res.detections[0].location_data.relative_bounding_box
                x0 = int(box.xmin * w)
                y0 = int(box.ymin * h)
                bw = int(box.width * w)
                bh = int(box.height * h)
                y_m0 = y0 + int(bh * 0.55)
                y_m1 = y0 + int(bh * 0.95)
                x_m0 = x0 + int(bw * 0.15)
                x_m1 = x0 + int(bw * 0.85)
                x_m0 = max(0, x_m0); y_m0 = max(0, y_m0)
                x_m1 = min(w, x_m1); y_m1 = min(h, y_m1)
                if x_m1 > x_m0 and y_m1 > y_m0:
                    return x_m0, y_m0, x_m1, y_m1
        except Exception:
            return None
    return None


def mouth_stack_from_video(video_path: Path, out_size=96, enlarge=1.4, method="mp", fan=None, mp_mesh=None, mp_face_det=None) -> Optional[Tuple['np.ndarray', str]]:  # type: ignore
    """Return (uint8[T,H,W], method_used) or None if detection fails.
    No center fallback included (enforces user requirement)."""
    if torchvision is None:
        raise RuntimeError("torchvision not installed")
    import numpy as _np
    v, a, info = torchvision.io.read_video(str(video_path), pts_unit="sec")  # (T,H,W,C)
    if v.numel() == 0:
        return None
    v_np = v.numpy()
    T = v_np.shape[0]
    # Use middle frame for detection
    mid = T // 2
    ref = v_np[mid]
    bbox = None
    used = None
    # Try FAN first if method == fan
    if method == "fan":
        bbox = detect_bbox_fan(ref, fan)
        if bbox is not None:
            used = "bbox_fan"
        else:  # fallback to mp (still detection-based)
            bbox = detect_bbox_mp(ref, mp_mesh, mp_face_det)
            if bbox is not None:
                used = "bbox_mp"
    elif method == "mp":
        bbox = detect_bbox_mp(ref, mp_mesh, mp_face_det)
        if bbox is not None:
            used = "bbox_mp"
    else:
        raise ValueError(f"Unsupported method {method}")
    if bbox is None:
        return None  # enforce drop
    x0, y0, x1, y1 = bbox
    cx = (x0 + x1) / 2.0; cy = (y0 + y1) / 2.0
    bw = (x1 - x0) * enlarge; bh = (y1 - y0) * enlarge
    x0 = int(max(0, cx - bw / 2)); x1 = int(cx + bw / 2)
    y0 = int(max(0, cy - bh / 2)); y1 = int(cy + bh / 2)
    frames_out = []
    for i in range(T):
        frame = v_np[i][y0:y1, x0:x1]
        if Image is not None:
            im = Image.fromarray(frame).convert("L").resize((out_size, out_size), Image.BILINEAR)
            arr = np.array(im, dtype=np.uint8)
        else:
            arr = frame.mean(axis=2).astype(np.uint8)
        frames_out.append(arr)
    return np.stack(frames_out, axis=0), (used or method)


# --------------------------------------------------------------------------------------
# Multiprocessing helpers
# --------------------------------------------------------------------------------------
_GLOBAL_CTX = None


def _init_worker(worker_method: str, device: str):
    global _GLOBAL_CTX
    ctx = {"fan": None, "mp_mesh": None, "mp_face_det": None, "method": worker_method}
    if worker_method in ("mp", "fan") and _MP_OK:
        try:
            ctx["mp_mesh"] = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)
            ctx["mp_face_det"] = mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.4)
        except Exception as e:  # pragma: no cover
            warnings.warn(f"MediaPipe init failed in worker: {e}")
    if worker_method == "fan" and _FAN_OK:
        try:
            ctx["fan"] = FANPredictor(device=device)  # type: ignore
        except Exception as e:  # pragma: no cover
            warnings.warn(f"FAN init failed in worker: {e}")
    _GLOBAL_CTX = ctx


def _worker(arg):
    (vid_path_str, out_size, enlarge, method, prepared_root_str, split) = arg
    from pathlib import Path as _P
    vid = _P(vid_path_str)
    word = vid.name.split('_', 1)[1].rsplit('.mp4', 1)[0]
    h = sha1_list([str(vid)])
    out_pt = _P(prepared_root_str) / split / word / f"{h}.pt"
    out_pt.parent.mkdir(parents=True, exist_ok=True)
    try:
        ctx = _GLOBAL_CTX or {}
        arr_used = mouth_stack_from_video(
            vid,
            out_size=out_size,
            enlarge=enlarge,
            method=method,
            fan=ctx.get("fan"),
            mp_mesh=ctx.get("mp_mesh"),
            mp_face_det=ctx.get("mp_face_det"),
        )
        if arr_used is None:
            return None
        arr, used = arr_used
        if torch is None:  # pragma: no cover
            raise RuntimeError("torch required to save prepared tensor")
        torch.save({"frames": torch.from_numpy(arr), "meta": {"method": used, "word": word}}, out_pt)
        return {"word": word, "prepared": str(out_pt), "T": int(arr.shape[0]), "method": used}
    except Exception as e:  # pragma: no cover
        return {"error": f"{vid}: {e}"}


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="datasets/LRW-B/Data", help="Directory containing pX_*.mp4 files")
    ap.add_argument("--prepared_dir", default="datasets/LRW-B/prepared", help="Output directory for prepared crops")
    ap.add_argument("--indices_dir", default="datasets/LRW-B/indices", help="Output directory for indices .pt")
    ap.add_argument("--splits_record", default="datasets/LRW-B/splits.json", help="Optional JSON record of split assignment")
    ap.add_argument("--train_speakers", default="p1,p2,p5,p6", help="Comma-separated speakers for train")
    ap.add_argument("--val_speakers", default="p4", help="Comma-separated speakers for val")
    ap.add_argument("--test_speakers", default="p3", help="Comma-separated speakers for test")
    ap.add_argument("--landmarks", choices=["mp","fan"], default="mp")
    ap.add_argument("--device", default="cuda", help="Device for FAN (if used)")
    ap.add_argument("--out_size", type=int, default=96)
    ap.add_argument("--enlarge", type=float, default=1.4)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--max_clips", type=int, default=0, help="Debug limit")
    ap.add_argument("--overwrite", action="store_true", help="Recompute even if output exists (not yet used)")
    ap.add_argument("--drop_on_fail", action="store_true", help="(Always true logically here) Drop clips with failed detection")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    assert data_dir.is_dir(), f"Data dir missing: {data_dir}"

    prepared_dir = Path(args.prepared_dir); prepared_dir.mkdir(parents=True, exist_ok=True)
    indices_dir = Path(args.indices_dir); indices_dir.mkdir(parents=True, exist_ok=True)

    # Gather mp4 files
    mp4s = sorted([p for p in data_dir.glob("*.mp4")])
    spk_re = re.compile(r"^(p\d+)_")
    entries = []
    for p in mp4s:
        m = spk_re.match(p.name)
        if not m:
            continue
        spk = m.group(1)
        word = p.name.split('_',1)[1].rsplit('.mp4',1)[0]
        entries.append((spk, word, p))
    if args.max_clips > 0:
        entries = entries[:args.max_clips]
    print(f"Found {len(entries)} clips")

    train_speakers = {s.strip() for s in args.train_speakers.split(',') if s.strip()}
    val_speakers = {s.strip() for s in args.val_speakers.split(',') if s.strip()}
    test_speakers = {s.strip() for s in args.test_speakers.split(',') if s.strip()}
    assert train_speakers.isdisjoint(val_speakers)
    assert train_speakers.isdisjoint(test_speakers)
    assert val_speakers.isdisjoint(test_speakers)

    split_map = {}
    for spk, word, path in entries:
        if spk in train_speakers:
            split = "train"
        elif spk in val_speakers:
            split = "val"
        elif spk in test_speakers:
            split = "test"
        else:
            continue
        split_map.setdefault(split, []).append((word, path))

    train_words = {w for w,_ in split_map.get('train', [])}
    # Filter val/test: keep only words present in train (avoid unseen vocab issue)
    removed_val = removed_test = 0
    new_val = []
    for w,p in split_map.get('val', []):
        if w in train_words:
            new_val.append((w,p))
        else:
            removed_val += 1
    split_map['val'] = new_val
    new_test = []
    for w,p in split_map.get('test', []):
        if w in train_words:
            new_test.append((w,p))
        else:
            removed_test += 1
    split_map['test'] = new_test
    if removed_val or removed_test:
        print(f"Filtered unseen words - val removed: {removed_val}, test removed: {removed_test}")

    # Write sorted labels (train vocabulary)
    sorted_labels_path = Path("datasets/LRW-B/sorted_labels.txt")
    if torch is not None:
        labels_sorted = sorted(train_words)
        sorted_labels_path.parent.mkdir(parents=True, exist_ok=True)
        sorted_labels_path.write_text("\n".join(labels_sorted), encoding="utf-8")
        print(f"Saved labels list: {sorted_labels_path} ({len(labels_sorted)})")

    # Save split record (filenames only for reproducibility)
    record = {k: [str(p.name) for _,p in v] for k,v in split_map.items()}
    Path(args.splits_record).write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"Saved split record: {args.splits_record}")

    # Initialize global detectors for serial path if needed
    fan = None; mp_mesh = None; mp_face_det = None
    method = args.landmarks
    if method in ("mp","fan") and _MP_OK:
        try:
            mp_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)
            mp_face_det = mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.4)
            print("Initialized MediaPipe")
        except Exception as e:
            warnings.warn(f"MediaPipe init failed: {e}")
            mp_mesh = None; mp_face_det = None
    if method == "fan" and _FAN_OK:
        try:
            dev = args.device
            if dev == "cuda" and not (torch and torch.cuda.is_available()):
                warnings.warn("CUDA not available; using CPU for FAN")
                dev = "cpu"
            fan = FANPredictor(device=dev)  # type: ignore
            print("Initialized FANPredictor")
        except Exception as e:
            warnings.warn(f"FAN init failed: {e}; falling back to mp only")
            fan = None

    for split, items in split_map.items():
        print(f"Processing split {split} ({len(items)} clips)")
        method_counts = {"bbox_fan":0, "bbox_mp":0}
        out_list: List[Dict] = []
        # Build work list
        work = [(str(p), args.out_size, args.enlarge, method, str(prepared_dir), split) for (w,p) in items]
        if args.num_workers > 0:
            from tqdm import tqdm
            with mp_proc.Pool(processes=args.num_workers, initializer=_init_worker, initargs=(method, args.device)) as pool:
                for res in tqdm(pool.imap_unordered(_worker, work), total=len(work), desc=f"{split} preprocess", unit="clip"):
                    if not res:
                        continue  # dropped (detection fail)
                    if 'error' in res:
                        if args.debug:
                            print(res['error'])
                        continue
                    m = res.get('method','')
                    method_counts[m] = method_counts.get(m,0)+1
                    out_list.append({"word": res['word'], "prepared": res['prepared'], "T": res['T'], "method": m})
        else:
            from tqdm import tqdm
            for w,p in tqdm(items, desc=f"{split} preprocess", unit="clip"):
                arr_used = mouth_stack_from_video(p, out_size=args.out_size, enlarge=args.enlarge, method=method, fan=fan, mp_mesh=mp_mesh, mp_face_det=mp_face_det)
                if arr_used is None:
                    continue
                arr, used = arr_used
                method_counts[used] = method_counts.get(used,0)+1
                h = sha1_list([str(p)])
                out_pt = prepared_dir / split / w / f"{h}.pt"
                out_pt.parent.mkdir(parents=True, exist_ok=True)
                if torch is None:
                    raise RuntimeError("torch required to save prepared tensor")
                torch.save({"frames": torch.from_numpy(arr), "meta": {"method": used, "word": w}}, out_pt)
                out_list.append({"word": w, "prepared": str(out_pt), "T": int(arr.shape[0]), "method": used})
        if torch is not None:
            out_indices = indices_dir / f"{split}.pt"
            torch.save(out_list, out_indices)
            with (indices_dir / f"{split}_preview.json").open('w', encoding='utf-8') as f:
                json.dump(out_list[:5], f, ensure_ascii=False, indent=2)
            print(f"Saved indices: {out_indices} ({len(out_list)})")
        else:  # pragma: no cover
            with (indices_dir / f"{split}.json").open('w', encoding='utf-8') as f:
                json.dump(out_list, f, ensure_ascii=False, indent=2)
        print(f"Split {split} done. Method usage: {method_counts}")
    print("Done.")


if __name__ == "__main__":
    main()
