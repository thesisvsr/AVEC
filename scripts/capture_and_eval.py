#!/usr/bin/env python3
"""Load a prerecorded clip, preprocess it like LipBengal, and run VisualCE inference."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch

# Make repo modules importable when running directly from scripts/
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    import cv2  # type: ignore
except Exception as exc:  # pragma: no cover - hard fail without camera bindings
    raise SystemExit(
        "OpenCV (cv2) is required for live capture. Install it with 'pip install opencv-python'."
    ) from exc

try:  # Optional mouth landmarking via MediaPipe
    import mediapipe as mp  # type: ignore

    _MP_OK = True
except Exception:
    mp = None  # type: ignore
    _MP_OK = False

import importlib

import functions  # noqa: E402
from nnet import transforms as vtf  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Video file preprocessing + VisualCE evaluation")
    parser.add_argument(
        "--config",
        default="configs.LipBengal.AV.VisualCE",
        help="Python module path to the config to load (default: %(default)s)",
    )
    parser.add_argument(
        "--config-file",
        default=None,
        help="Optional config file path passed to functions.load_model (defaults to module path with '.py').",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Checkpoint file to load. Provide either the filename inside the checkpoint folder or an absolute path."
             " Defaults to the most recent checkpoint in the config callback path.",
    )
    parser.add_argument(
        "--video-path",
        type=Path,
        default=Path("LipBengal/AV/test1.mp4"),
        help="Use a prerecorded video instead of the live camera (default: %(default)s).",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Display a real-time OpenCV preview window while recording.",
    )
    parser.add_argument(
        "--output-prepared",
        type=Path,
        default=None,
        help="If set, save the preprocessed grayscale tensor as a .pt file at this path.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Force inference device (e.g. 'cpu', 'cuda'). Defaults to auto-detect.",
    )
    parser.add_argument(
        "--target-frames",
        type=int,
        default=29,
        help="Number of frames expected by the model after preprocessing (default: %(default)s)",
    )
    parser.add_argument(
        "--mouth-enlarge",
        type=float,
        default=1.4,
        help="Scaling factor applied around detected lips before cropping (default: %(default)s)",
    )
    parser.add_argument(
        "--no-mediapipe",
        action="store_true",
        help="Disable MediaPipe face mesh detection even if installed.",
    )
    return parser.parse_args()


def read_video_file(path: Path, preview: bool) -> List[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video file: {path}")
    frames: List[np.ndarray] = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            frames.append(frame)
            if preview:
                cv2.imshow("Preview", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        cap.release()
        if preview:
            cv2.destroyAllWindows()
    if not frames:
        raise RuntimeError(f"No frames decoded from {path}")
    return frames


def resample_sequence(seq: Sequence[np.ndarray], target_len: int) -> List[np.ndarray]:
    if target_len <= 0:
        return list(seq)
    n = len(seq)
    if n == 0:
        return []
    if n == target_len:
        return list(seq)
    if n > target_len:
        idx = np.linspace(0, n - 1, target_len).round().astype(int)
        return [seq[i] for i in idx]
    # Pad by repeating the last frame
    result = list(seq)
    last = seq[-1]
    while len(result) < target_len:
        result.append(last.copy())
    return result


def detect_mouth_bbox_mp(img_rgb: np.ndarray, mesh) -> Optional[Tuple[int, int, int, int]]:
    if mesh is None:
        return None
    try:
        res = mesh.process(img_rgb)
    except Exception:
        return None
    if not getattr(res, "multi_face_landmarks", None):
        return None
    lm = res.multi_face_landmarks[0].landmark
    h, w, _ = img_rgb.shape
    lip_idx = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 78, 191, 80, 81, 82, 13, 312, 311, 310, 415]
    xs = [int(lm[i].x * w) for i in lip_idx]
    ys = [int(lm[i].y * h) for i in lip_idx]
    x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


def crop_center(frame: np.ndarray, out_size: int) -> np.ndarray:
    h, w, _ = frame.shape
    side = min(h, w)
    cy, cx = h // 2, w // 2
    half = side // 2
    y0 = max(0, cy - half)
    x0 = max(0, cx - half)
    y1 = min(h, y0 + side)
    x1 = min(w, x0 + side)
    roi = frame[y0:y1, x0:x1]
    gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    return cv2.resize(gray, (out_size, out_size), interpolation=cv2.INTER_AREA)


def crop_with_bbox(frame: np.ndarray, bbox: Optional[Tuple[int, int, int, int]], out_size: int, enlarge: float) -> np.ndarray:
    if bbox is None:
        return crop_center(frame, out_size)
    h, w, _ = frame.shape
    x0, y0, x1, y1 = bbox
    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0
    bw = (x1 - x0) * float(enlarge)
    bh = (y1 - y0) * float(enlarge)
    x0 = max(0, int(cx - bw / 2))
    x1 = min(w, int(cx + bw / 2))
    y0 = max(0, int(cy - bh / 2))
    y1 = min(h, int(cy + bh / 2))
    if x1 <= x0 or y1 <= y0:
        return crop_center(frame, out_size)
    roi = frame[y0:y1, x0:x1]
    if roi.size == 0:
        return crop_center(frame, out_size)
    gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
    return cv2.resize(gray, (out_size, out_size), interpolation=cv2.INTER_AREA)


def preprocess_frames(
    frames_bgr: Sequence[np.ndarray],
    target_frames: int,
    out_size: int = 88,
    enlarge: float = 1.4,
    use_mediapipe: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if not frames_bgr:
        raise ValueError("No frames captured from camera; cannot preprocess.")

    frames_rgb = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in frames_bgr]
    frames_rgb = resample_sequence(frames_rgb, target_frames)

    mesh = None
    if use_mediapipe and _MP_OK:
        try:
            mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1)
        except Exception:
            mesh = None

    try:
        base_bbox: Optional[Tuple[int, int, int, int]] = None
        if mesh is not None:
            mid = frames_rgb[len(frames_rgb) // 2]
            base_bbox = detect_mouth_bbox_mp(mid, mesh)

        crops: List[np.ndarray] = []
        for frame in frames_rgb:
            bbox = base_bbox
            if bbox is None and mesh is not None:
                bbox = detect_mouth_bbox_mp(frame, mesh)
            crop = crop_with_bbox(frame, bbox, out_size=out_size, enlarge=enlarge)
            crops.append(crop.astype(np.uint8, copy=False))
        data_uint8 = np.stack(crops, axis=0)  # (T, H, W)
    finally:
        if mesh is not None and hasattr(mesh, "close"):
            mesh.close()

    # Convert to torch tensor matching dataset expectations: (C=1, T, H, W)
    video = torch.from_numpy(data_uint8).unsqueeze(0)  # (1, T, H, W)
    video_float = video.to(dtype=torch.float32) / 255.0
    return video_float, video


def load_model_and_labels(
    config_module_name: str,
    config_file: str,
    checkpoint_arg: Optional[str],
    device_hint: Optional[str],
) -> Tuple[torch.nn.Module, List[str], Path, object]:
    config = importlib.import_module(config_module_name)

    class Args:
        pass

    args = Args()
    args.rank = 0
    args.cpu = False
    if device_hint is not None:
        args.cpu = device_hint.startswith("cpu")
    else:
        args.cpu = not torch.cuda.is_available()
    args.distributed = False
    args.parallel = False
    args.show_dict = False
    args.show_modules = False
    args.config = config
    args.config_file = config_file
    args.mode = "evaluation"
    args.batch_size_eval = None
    args.num_workers = 0
    args.world_size = 1
    args.dist_log = False
    args.load_last = checkpoint_arg is None
    args.checkpoint = None

    checkpoint_path: Optional[Path] = None
    if checkpoint_arg:
        cand = Path(checkpoint_arg)
        if cand.is_file():
            config.callback_path = str(cand.parent)
            args.checkpoint = cand.name
            args.load_last = False
            checkpoint_path = cand
        else:
            args.checkpoint = checkpoint_arg
            checkpoint_path = Path(config.callback_path) / checkpoint_arg
    model = functions.load_model(args)
    model.eval()

    if checkpoint_path is None:
        last = functions.find_last_checkpoint(config.callback_path, return_full_path=True)
        if last is None:
            raise RuntimeError(
                f"No checkpoints found under {config.callback_path}. Provide --checkpoint explicitly."
            )
        checkpoint_path = Path(last)

    if args.load_last and args.checkpoint is None:
        # load_model already loaded last; nothing more to do
        pass
    elif args.checkpoint and checkpoint_path.is_file():
        # load_model already handled this case during initialization
        pass
    elif checkpoint_path.is_file():
        model.load(str(checkpoint_path), load_optimizer=False)

    if hasattr(config, "training_dataset") and hasattr(config.training_dataset, "classes"):
        label_words = list(config.training_dataset.classes)
    elif hasattr(config, "evaluation_dataset") and hasattr(config.evaluation_dataset, "classes"):
        label_words = list(config.evaluation_dataset.classes)
    else:
        raise RuntimeError("Unable to infer class labels from the config datasets.")

    return model, label_words, checkpoint_path, config


def run_inference(
    model: torch.nn.Module,
    video: torch.Tensor,
    val_transform,
) -> torch.Tensor:
    # Ensure the clip has the same normalization pipeline as dataset evaluation
    normalized = vtf.NormalizeVideo(mean=(0.5,), std=(0.5,))(video)
    processed = val_transform(normalized)
    if not torch.is_tensor(processed):
        processed = torch.as_tensor(processed)
    if processed.dim() == 3:
        processed = processed.unsqueeze(0)
    if processed.dim() != 4:
        raise RuntimeError(f"Expected preprocessed video to have 4 dims (C,T,H,W), got {processed.shape}")
    processed = processed.unsqueeze(0)  # (1, C, T, H, W)
    device = next(model.parameters()).device
    processed = processed.to(device=device)
    param_dtype = next(model.parameters()).dtype
    if processed.dtype != param_dtype:
        processed = processed.to(dtype=param_dtype)
    with torch.no_grad():
        logits = model.forward(processed)
    return logits


def display_topk(logits: torch.Tensor, label_words: Sequence[str], k: int = 5) -> None:
    probs = torch.softmax(logits, dim=1)[0]
    topk = torch.topk(probs, k=min(k, probs.shape[0]))
    print("\nTop predictions:")
    for rank, (idx, score) in enumerate(zip(topk.indices.tolist(), topk.values.tolist()), start=1):
        word = label_words[idx] if 0 <= idx < len(label_words) else f"<unknown:{idx}>"
        print(f"  {rank}. {word:20s}  {score * 100:.2f}%")


def main() -> None:
    args = parse_args()

    if args.config_file is None:
        if args.config.endswith(".py"):
            args.config_file = args.config
        else:
            args.config_file = args.config.replace(".", "/") + ".py"

    if args.video_path is None:
        raise SystemExit("--video-path must be provided.")
    if not args.video_path.is_file():
        raise SystemExit(f"Video file not found: {args.video_path}")

    frames = read_video_file(args.video_path, preview=args.preview)
    print(f"Loaded {len(frames)} frames from {args.video_path}.")

    video_float, video_uint8 = preprocess_frames(
        frames,
        target_frames=args.target_frames,
        out_size=88,
        enlarge=args.mouth_enlarge,
        use_mediapipe=not args.no_mediapipe,
    )

    if args.output_prepared is not None:
        args.output_prepared.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"frames": video_uint8.cpu(), "meta": {"source": "video_file", "path": str(args.video_path)}},
            args.output_prepared,
        )
        print(f"Saved preprocessed tensor to {args.output_prepared}")

    model, label_words, ckpt_path, config = load_model_and_labels(
        config_module_name=args.config,
        config_file=args.config_file,
        checkpoint_arg=args.checkpoint,
        device_hint=args.device,
    )
    val_transform = config.val_video_transform

    logits = run_inference(model, video_float, val_transform)
    print(f"Loaded checkpoint: {ckpt_path}")
    display_topk(logits, label_words, k=5)


if __name__ == "__main__":
    main()
