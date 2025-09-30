#!/usr/bin/env python3
"""Preprocess LRW-AR dataset (word-level Arabic lip-reading) similar in spirit to LipBengal.

Pipeline:
 1. Scan datasets/LRW-AR/{train,val,test}/*/*.mp4
 2. (Optional) Extract frames to a temporary cache OR read on-the-fly (default: on-the-fly using torchvision.io)
 3. Detect/crop mouth region (configurable methods: lipcrop, fan, mediapipe, face-det fallback, center) -> resize to 96x96 grayscale
 4. Save prepared tensor (.pt) per clip: {frames: uint8[T,H,W], meta:{method,word}}
 5. Build indices/*.pt listing: {word, prepared, T}
"""

from typing import Optional, List, Dict
from pathlib import Path

import os, sys, argparse, json, hashlib, warnings, multiprocessing as mp_proc

try:
	import torch
except Exception:
	torch = None  # type: ignore

try:
	import numpy as np
except Exception:
	np = None  # type: ignore

try:
	from PIL import Image
except Exception:
	Image = None  # type: ignore

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
	import torchvision
	_TV_OK = True
except Exception:
	torchvision = None  # type: ignore
	_TV_OK = False

try:
	from nnet.transforms import LipDetectCrop  # type: ignore
	_LIP_OK = True
except Exception:
	LipDetectCrop = None  # type: ignore
	_LIP_OK = False

def sha1_list(strs: List[str]) -> str:
	h = hashlib.sha1()
	for s in strs:
		h.update(s.encode("utf-8"))
	return h.hexdigest()[:16]

def detect_mouth_bbox_fan(img_rgb, fan) -> Optional[tuple[int,int,int,int]]:
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
        if lms_list is None or len(lms_list) == 0:
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


def detect_mouth_bbox_mp(img_rgb, mp_mesh, mp_face_det=None) -> Optional[tuple[int,int,int,int]]:
	"""Primary: FaceMesh lip landmarks. Fallback: face detection lower region."""
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
	# Fallback: generic face detection -> approximate mouth region
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
				# Lower 40% central 70% of face rectangle
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


def mouth_stack_from_video(video_path: Path, out_size=96, enlarge=1.6, mp_mesh=None, mp_face_det=None, fan=None, per_clip=True,
						   lip=None, lip_fail_fallback=True):
	"""Return uint8 (T,H,W) grayscale mouth crops from MP4 with hierarchy:
		1. LipDetectCrop warp (if lip provided)
		2. Landmark bbox (FAN / MediaPipe)
		3. Center crop fallback
	"""
	if not _TV_OK:
		raise RuntimeError("torchvision not installed; cannot read video")
	import numpy as _np
	import torchvision.io
	v, a, info = torchvision.io.read_video(str(video_path), pts_unit="sec")  # (T,H,W,C)
	if v.numel() == 0:
		return _np.zeros((1, out_size, out_size), dtype=_np.uint8), "center"
	v_np = v.numpy()  # (T,H,W,3)
	# Try lipcrop path
	if lip is not None:
		try:
			import torch as _torch
			video_t = _torch.from_numpy(v_np)
			lms = lip.detect_landmarks(video_t, verbose=0)
			pre_lms = lip.landmarks_interpolate(lms)
			if pre_lms:
				vcrop = lip.crop_patch(v_np, pre_lms)
				if vcrop is not None:
					grays = []
					for f in vcrop:
						if Image is not None:
							g = Image.fromarray(f).convert("L").resize((out_size, out_size), Image.BILINEAR)
							grays.append(_np.array(g, dtype=_np.uint8))
						else:
							grays.append(f.mean(axis=2).astype(_np.uint8))
					return _np.stack(grays, axis=0), "lipcrop"
		except Exception:
			if not lip_fail_fallback:
				raise
			# fall back to bbox/center path
	# BBox path
	T = v_np.shape[0]
	crops = []
	fixed_bbox = None
	detector_used = None
	if per_clip:
		mid = T // 2
		ref = v_np[mid]
		bbox = detect_mouth_bbox_fan(ref, fan) if fan is not None else None
		if bbox is not None:
			detector_used = "bbox_fan"
		if bbox is None:
			bbox = detect_mouth_bbox_mp(ref, mp_mesh, mp_face_det=mp_face_det)
			if bbox is not None:
				detector_used = "bbox_mp"
		fixed_bbox = bbox
	for i in range(T):
		frame = v_np[i]
		bbox = fixed_bbox
		if bbox is None:
			bfan = detect_mouth_bbox_fan(frame, fan) if fan is not None else None
			if bfan is not None:
				bbox = bfan; detector_used = "bbox_fan"
			else:
				bmp = detect_mouth_bbox_mp(frame, mp_mesh, mp_face_det=mp_face_det)
				if bmp is not None:
					bbox = bmp; detector_used = "bbox_mp"
		if bbox is None:
			H, W = frame.shape[:2]
			side = min(H, W)
			cy, cx = H // 2, W // 2
			half = side // 2
			y0, y1 = cy - half, cy + half
			x0, x1 = cx - half, cx + half
			cur_method = "center"
		else:
			x0, y0, x1, y1 = bbox
			cx = (x0 + x1) / 2.0
			cy = (y0 + y1) / 2.0
			bw = (x1 - x0) * enlarge
			bh = (y1 - y0) * enlarge
			x0 = int(max(0, cx - bw / 2))
			x1 = int(cx + bw / 2)
			y0 = int(max(0, cy - bh / 2))
			y1 = int(cy + bh / 2)
			cur_method = detector_used or "bbox"
		roi = frame[y0:y1, x0:x1]
		if Image is not None:
			im = Image.fromarray(roi).convert("L").resize((out_size, out_size), Image.BILINEAR)
			arr = _np.array(im, dtype=_np.uint8)
		else:
			arr = roi.mean(axis=2).astype(_np.uint8)
		crops.append(arr)
	if not crops:
		crops = [_np.zeros((out_size, out_size), dtype=_np.uint8)]
	method_final = detector_used if detector_used is not None else "center"
	return _np.stack(crops, axis=0), method_final


# ---- Multiprocessing worker helpers (must be top-level to be pickled) ----
_GLOBAL_WORK_CTX = None

def _init_worker(worker_method, device, use_lip):
	global _GLOBAL_WORK_CTX
	_GLOBAL_WORK_CTX = {"mp_mesh": None, "mp_face_det": None, "fan": None, "lip": None, "method": worker_method}
	if worker_method in ("mp","fan","lipcrop") and _MP_OK:
		try:
			m_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)
			m_det = mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.4)
			_GLOBAL_WORK_CTX["mp_mesh"] = m_mesh
			_GLOBAL_WORK_CTX["mp_face_det"] = m_det
		except Exception:
			pass
	if worker_method in ("fan","lipcrop") and _FAN_OK:
		try:
			fan_local = FANPredictor(device=device)  # type: ignore
			_GLOBAL_WORK_CTX["fan"] = fan_local
		except Exception:
			pass
	if use_lip and _LIP_OK:
		try:
			lip_local = LipDetectCrop(mean_face_landmarks_path="media/20words_mean_face.npy")  # type: ignore
			_GLOBAL_WORK_CTX["lip"] = lip_local
		except Exception:
			pass


def _worker_process(arg_tuple):
	(vid_path_str, out_size, enlarge, per_clip, overwrite, update_only, drop_on_fail, debug, split, prepared_dir_str, method) = arg_tuple
	from pathlib import Path as _Path
	vid = _Path(vid_path_str)
	word = vid.parent.name
	h = sha1_list([str(vid)])
	prepared_dir = _Path(prepared_dir_str)
	out_pt = prepared_dir / split / word / f"{h}.pt"
	out_pt.parent.mkdir(parents=True, exist_ok=True)
	if update_only and out_pt.exists() and not overwrite:
		try:
			meta_T = torch.load(out_pt, map_location="cpu")
			T = int(meta_T.get("frames").shape[0]) if isinstance(meta_T, dict) else 0
			method_used = meta_T.get("meta",{}).get("method","?") if isinstance(meta_T, dict) else "?"
		except Exception:
			T = 0
			method_used = "?"
		return {"word": word, "prepared": str(out_pt), "T": T, "method": method_used}
	try:
		ctx = _GLOBAL_WORK_CTX or {}
		arr, method_used = mouth_stack_from_video(
			vid,
			out_size=out_size,
			enlarge=enlarge,
			mp_mesh=ctx.get("mp_mesh"),
			mp_face_det=ctx.get("mp_face_det"),
			fan=ctx.get("fan"),
			per_clip=per_clip,
			lip=ctx.get("lip") if method == "lipcrop" else None,
		)
		if torch is None:
			raise RuntimeError("torch required to save prepared tensors")
		torch.save({"frames": torch.from_numpy(arr), "meta": {"method": method_used, "word": word}}, out_pt)
		return {"word": word, "prepared": str(out_pt), "T": arr.shape[0], "method": method_used}
	except Exception as e:
		if debug:
			return {"error": f"{vid}: {e}"}
		return None


def main():
	ap = argparse.ArgumentParser()
	ap.add_argument("--root", default="datasets/LRW-AR", help="LRW-AR dataset root (contains train/val/test)")
	ap.add_argument("--splits", default="train,val,test", help="Comma-separated splits to process")
	ap.add_argument("--prepared_dir", default="datasets/LRW-AR/prepared", help="Output dir for prepared crops")
	ap.add_argument("--indices_dir", default="datasets/LRW-AR/indices", help="Output dir for indices .pt files")
	ap.add_argument("--update_only", action="store_true", help="Skip if prepared exists")
	ap.add_argument("--overwrite", action="store_true", help="Recompute even if exists")
	ap.add_argument("--landmarks", choices=["lipcrop","mp","fan","none"], default="mp")
	ap.add_argument("--device", default="cpu", help="Device for FAN")
	ap.add_argument("--bbox_mode", choices=["per_clip","per_frame"], default="per_clip")
	ap.add_argument("--enlarge", type=float, default=1.4, help="BBox enlarge factor")
	ap.add_argument("--out_size", type=int, default=96, help="Output spatial size (H=W)")
	ap.add_argument("--num_workers", type=int, default=0, help="Multiprocessing workers (FaceMesh/FAN re-init per worker)")
	ap.add_argument("--max_clips", type=int, default=0, help="Process only first N clips (debug)")
	ap.add_argument("--debug_fail", action="store_true", help="Print detection failure details")
	ap.add_argument("--shard", type=int, default=0)
	ap.add_argument("--num_shards", type=int, default=1)
	ap.add_argument("--drop_on_fail", action="store_true", help="Drop clips that fallback to center")
	args = ap.parse_args()

	root = Path(args.root)
	assert root.is_dir(), f"Root not found: {root}"

	labels_file = root / "sorted_labels.txt"
	if labels_file.is_file():
		vocab = [l.strip() for l in labels_file.read_text(encoding="utf-8").splitlines() if l.strip()]
	else:
		vocab = []

	# Initialize alignment tools
	mp_mesh = None
	fan = None
	method = args.landmarks
	lip_obj = None
	if method == "lipcrop":
		if not _LIP_OK:
			warnings.warn("LipDetectCrop not available; falling back to mp")
			method = "mp"
		else:
			try:
				lip_obj = LipDetectCrop(mean_face_landmarks_path="media/20words_mean_face.npy")  # type: ignore
				print("Initialized LipDetectCrop")
			except Exception as e:
				warnings.warn(f"LipDetectCrop init failed ({e}); falling back to mp")
				lip_obj = None
				method = "mp"
	if method == "fan" and _FAN_OK:
		try:
			dev = args.device
			if dev == "cuda" and not (torch and torch.cuda.is_available()):
				warnings.warn("CUDA not available; using CPU for FAN")
				dev = "cpu"
			fan = FANPredictor(device=dev)  # type: ignore
			print("Initialized FANPredictor")
		except Exception as e:
			warnings.warn(f"FAN init failed ({e}); falling back to mp")
			fan = None
			method = "mp"
	mp_face_det = None
	if method in ("mp", "lipcrop", "fan") and _MP_OK:
		try:
			mp_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)
			mp_face_det = mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.4)
			print("Initialized MediaPipe FaceMesh + FaceDetection")
		except Exception as e:
			warnings.warn(f"MediaPipe init failed: {e}")
			mp_mesh = None
			mp_face_det = None
	if method == "none":
		mp_mesh = None
		fan = None

	prepared_dir = Path(args.prepared_dir)
	indices_dir = Path(args.indices_dir)
	prepared_dir.mkdir(parents=True, exist_ok=True)
	indices_dir.mkdir(parents=True, exist_ok=True)

	splits = [s.strip() for s in args.splits.split(",") if s.strip()]
	for split in splits:
		mp4_pattern = root / split
		if not mp4_pattern.is_dir():
			warnings.warn(f"Split folder missing: {mp4_pattern}")
			continue
		mp4s = list(mp4_pattern.glob("*/*.mp4"))
		if not mp4s:
			warnings.warn(f"No MP4 files for split {split}")
			continue
		mp4s = sorted(mp4s, key=lambda p: (p.parent.name, p.name))
		total = len(mp4s)
		mp4s_shard = [p for i,p in enumerate(mp4s) if i % args.num_shards == args.shard]
		print(f"Split {split}: {len(mp4s_shard)}/{total} clips for shard {args.shard}")
		if args.max_clips > 0:
			mp4s_shard = mp4s_shard[:args.max_clips]
		method_counts = {"lipcrop":0, "bbox_fan":0, "bbox_mp":0, "center":0}
		items: List[Dict] = []
		if args.num_workers > 0:
			per_clip_flag = (args.bbox_mode=="per_clip")
			worker_args = [ (str(p), args.out_size, args.enlarge, per_clip_flag, args.overwrite, args.update_only, args.drop_on_fail, args.debug_fail, split, str(prepared_dir), method) for p in mp4s_shard ]
			from tqdm import tqdm
			with mp_proc.Pool(processes=args.num_workers, initializer=_init_worker, initargs=(method, args.device, method=="lipcrop")) as pool:
				for res in tqdm(pool.imap_unordered(_worker_process, worker_args), total=len(worker_args), desc=f"{split} preprocess", unit="clip"):
					if res is None:
						continue
					if 'error' in res:
						if args.debug_fail:
							print(res['error'])
						continue
					m = res.get("method","center")
					method_counts[m] = method_counts.get(m,0)+1
					items.append({"word": res["word"], "prepared": res["prepared"], "T": res["T"]})
		else:
			from tqdm import tqdm
			for vid in tqdm(mp4s_shard, desc=f"{split} preprocess", unit="clip"):
				word = vid.parent.name
				if vocab and word not in vocab:
					pass
				h = sha1_list([str(vid)])
				out_pt = prepared_dir / split / word / f"{h}.pt"
				out_pt.parent.mkdir(parents=True, exist_ok=True)
				if args.update_only and out_pt.exists() and not args.overwrite:
					try:
						meta_T = torch.load(out_pt, map_location="cpu")
						T = int(meta_T.get("frames").shape[0]) if isinstance(meta_T, dict) else 0
					except Exception:
						T = 0
					items.append({"word": word, "prepared": str(out_pt), "T": T})
					continue
				try:
					arr, method_used = mouth_stack_from_video(
						vid,
						out_size=args.out_size,
						enlarge=args.enlarge,
						mp_mesh=mp_mesh,
						mp_face_det=mp_face_det,
						fan=fan,
						per_clip=(args.bbox_mode=="per_clip"),
						lip=lip_obj if method == "lipcrop" else None,
					)
					method_counts[method_used] = method_counts.get(method_used,0)+1
					if torch is None:
						raise RuntimeError("torch required to save prepared tensors")
					torch.save({"frames": torch.from_numpy(arr), "meta": {"method": method_used, "word": word}}, out_pt)
					items.append({"word": word, "prepared": str(out_pt), "T": arr.shape[0]})
				except Exception as e:
					if args.debug_fail:
						warnings.warn(f"Failed clip {vid}: {e}")
					continue
		if torch is not None:
			out_indices = indices_dir / f"{split}.pt"
			torch.save(items, out_indices)
			with (indices_dir / f"{split}_preview.json").open("w", encoding="utf-8") as f:
				json.dump(items[:5], f, ensure_ascii=False, indent=2)
			print(f"Saved indices: {out_indices} ({len(items)})")
		else:
			with (indices_dir / f"{split}.json").open("w", encoding="utf-8") as f:
				json.dump(items, f, ensure_ascii=False, indent=2)
		print(f"Split {split} done. Method usage: {method_counts}")
	print("Done.")


if __name__ == "__main__":
	main()

