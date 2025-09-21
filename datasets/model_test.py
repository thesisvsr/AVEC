import argparse
import importlib
import importlib.util
import os
import sys
import glob
import time
import torch
import torchvision
import sentencepiece as spm
import numpy as np

# Ensure project root on sys.path so 'configs' and 'nnet' are importable when running from datasets/
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

def find_latest_checkpoint(callback_path: str) -> str | None:
    if not os.path.isdir(callback_path):
        return None
    cands = []
    for pat in ("**/*.ckpt", "*.ckpt"):
        cands += glob.glob(os.path.join(callback_path, pat), recursive=True)
    if not cands:
        return None
    cands.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return cands[0]

def to_device(x, device):
    if isinstance(x, torch.Tensor):
        return x.to(device)
    return x

def resample_audio_if_needed(audio, sr, target_sr=16000):
    try:
        import torchaudio
        if sr != target_sr:
            audio = torchaudio.functional.resample(audio, sr, target_sr)
            sr = target_sr
    except Exception:
        # Fallback: naive numpy resample if torchaudio missing
        if sr != target_sr:
            import math
            ratio = target_sr / sr
            idx = (np.arange(int(math.ceil(audio.shape[-1] * ratio))) / ratio).astype(np.int64)
            idx = np.clip(idx, 0, audio.shape[-1] - 1)
            audio = audio[..., idx]
            sr = target_sr
        audio = torch.as_tensor(audio)
    return audio, sr

def video_center_crop(frames, crop=(88, 88)):
    H, W = frames.shape[1], frames.shape[2]
    ch, cw = crop
    top = max(0, (H - ch) // 2)
    left = max(0, (W - cw) // 2)
    return frames[:, top:top+ch, left:left+cw, :]

def rgb_to_gray(frames):
    # frames: (T, H, W, 3) -> (T, H, W, 1)
    r, g, b = frames[..., 0], frames[..., 1], frames[..., 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray[..., None]

def frame_subsample(frames, src_fps, dst_fps=25.0):
    if src_fps <= 0:
        return frames
    if abs(src_fps - dst_fps) < 1e-3:
        return frames
    ratio = src_fps / dst_fps
    idx = (np.arange(int(np.floor(len(frames) / ratio))) * ratio).astype(np.int64)
    idx = np.clip(idx, 0, len(frames) - 1)
    return frames[idx]

def ctc_greedy_decode(logits, lengths, tokenizer: spm.SentencePieceProcessor, blank_id: int):
    # logits: (B, T, V), lengths: (B,)
    with torch.no_grad():
        ids = logits.argmax(-1)  # (B, T)
    transcripts = []
    for b in range(ids.size(0)):
        T = int(lengths[b].item())
        seq = ids[b, :T].tolist()
        dedup = []
        prev = None
        for t in seq:
            if t == blank_id:
                prev = None
                continue
            if prev is None or t != prev:
                dedup.append(t)
            prev = t
        # SentencePiece decode
        text = tokenizer.decode(dedup)
        transcripts.append(text)
    return transcripts

def load_video_audio(path, force_silent: bool = False):
    # Use torchvision to read video/audio
    video, audio, info = torchvision.io.read_video(path, pts_unit="sec")
    # video: (T, H, W, C) uint8; audio: (N, K) float32
    video_fps = info.get("video_fps", 25)
    # Convert to float [0,1]
    video = video.float() / 255.0
    if video.size(-1) == 3:
        video = torch.from_numpy(rgb_to_gray(video.numpy()))
    # Subsample video to 25 fps
    video = torch.from_numpy(frame_subsample(video.numpy(), float(video_fps), 25.0))
    # Center crop to 88x88
    video = torch.from_numpy(video_center_crop(video.numpy(), (88, 88)))
    # Ensure shape (T, H, W, 1)
    if video.ndim == 3:
        video = video.unsqueeze(-1)

    # Audio handling
    if force_silent or audio.numel() == 0:
        # Empty or disabled audio; create silence roughly matching video duration (min 2048 samples)
        target_len = int(16000 * max(1.0, video.shape[0] / 25.0))
        target_len = max(target_len, 2048)
        audio = torch.zeros(1, target_len, dtype=torch.float32)
        sr = 16000
    else:
        # audio: (N, K) -> mono (1, N)
        if audio.ndim == 1:
            audio = audio.unsqueeze(0)
        if audio.size(1) > 1:
            audio = audio.mean(dim=1, keepdim=True)
        audio = audio.transpose(0, 1).contiguous()  # (K, N) -> (1, N)
        sr = int(info.get("audio_fps", 16000))
        audio, sr = resample_audio_if_needed(audio, sr, 16000)
        # Ensure a minimum length to satisfy STFT padding and roughly align with video duration
        min_len = int(sr * max(1.0, video.shape[0] / 25.0))
        min_len = max(min_len, 2048)
        cur_len = audio.shape[-1]
        if cur_len < min_len:
            pad = torch.zeros(1, min_len - cur_len, dtype=audio.dtype)
            audio = torch.cat([audio, pad], dim=-1)

    return video.contiguous(), audio.contiguous(), 25, sr

def load_from_preprocessed_pt(pt_path: str, force_silent: bool = False):
    base = pt_path[:-3] if pt_path.endswith('.pt') else pt_path
    # The dataset prepare writes: base + "_mouth.mp4" and base + ".flac"
    mouth_mp4 = base + "_mouth.mp4"
    flac = base + ".flac"
    if not os.path.isfile(mouth_mp4):
        raise FileNotFoundError(f"Mouth video not found: {mouth_mp4}")
    # Video
    video, _, info = torchvision.io.read_video(mouth_mp4, pts_unit="sec")
    video = video.float() / 255.0
    if video.size(-1) == 3:
        video = torch.from_numpy(rgb_to_gray(video.numpy()))
    # Center crop to 88x88 (mouth crops should already be close)
    video = torch.from_numpy(video_center_crop(video.numpy(), (88, 88)))
    if video.ndim == 3:
        video = video.unsqueeze(-1)

    # Audio
    if force_silent or not os.path.isfile(flac):
        target_len = int(16000 * max(1.0, video.shape[0] / 25.0))
        target_len = max(target_len, 2048)
        audio = torch.zeros(1, target_len, dtype=torch.float32)
        sr = 16000
    else:
        import torchaudio
        audio, sr = torchaudio.load(flac)
        # mono (1, N)
        if audio.ndim == 2 and audio.size(0) > 1:
            audio = audio.mean(dim=0, keepdim=True)
        elif audio.ndim == 1:
            audio = audio.unsqueeze(0)
        # Resample if needed
        if sr != 16000:
            audio, _ = resample_audio_if_needed(audio, sr, 16000)
            sr = 16000
        # min length
        min_len = int(sr * max(1.0, video.shape[0] / 25.0))
        min_len = max(min_len, 2048)
        if audio.shape[-1] < min_len:
            pad = torch.zeros(1, min_len - audio.shape[-1], dtype=audio.dtype)
            audio = torch.cat([audio, pad], dim=-1)

    return video.contiguous(), audio.contiguous(), 25, sr

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="", help="Path to input video (.mp4, etc.)")
    ap.add_argument("--pt", default="", help="Path to preprocessed .pt file (use this instead of --video)")
    ap.add_argument("--config_file", default="configs/LRS2/AV/EffConfInterCTC.py")
    ap.add_argument("--checkpoint", default="", help="Path to model checkpoint (.ckpt). If empty, auto-detect latest in callback_path.")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--tokenizer_path", default="datasets/LRS3/tokenizerbpe256.model")
    ap.add_argument("--blank_id", type=int, default=255, help="CTC blank token id (default: last id for vocab_size=256).")
    ap.add_argument("--video_only", action="store_true", help="Ignore audio; use silent audio based on video length.")
    args = ap.parse_args()

    device = torch.device(args.device)

    # Import config and get model
    cfg = None
    cfg_mod_name = args.config_file.replace(".py", "").replace("/", ".")
    try:
        cfg = importlib.import_module(cfg_mod_name)
    except ModuleNotFoundError:
        # Fallback: import by file path if a path-like string was provided
        cfg_path = args.config_file
        if not os.path.isabs(cfg_path):
            cfg_path = os.path.join(ROOT, cfg_path)
        if os.path.isfile(cfg_path):
            spec = importlib.util.spec_from_file_location("_user_cfg_module_", cfg_path)
            if spec and spec.loader:
                cfg = importlib.util.module_from_spec(spec)
                sys.modules["_user_cfg_module_"] = cfg
                spec.loader.exec_module(cfg)
        if cfg is None:
            raise
    model = cfg.model
    model = model.to(device)
    model.eval()

    # Load checkpoint
    ckpt_path = args.checkpoint
    if not ckpt_path:
        ckpt_path = find_latest_checkpoint(getattr(cfg, "callback_path", "callbacks"))
    if ckpt_path and os.path.isfile(ckpt_path):
        print(f"Loading checkpoint: {ckpt_path}")
        state = torch.load(ckpt_path, map_location=device)
        sd = state.get("model_state_dict", state)
        # strip possible .module. prefixes
        cleaned = {}
        for k, v in sd.items():
            cleaned[k.replace(".module.", ".")] = v
        missing, unexpected = model.load_state_dict(cleaned, strict=False)
        if missing:
            print(f"Warning: missing keys: {len(missing)}")
        if unexpected:
            print(f"Warning: unexpected keys: {len(unexpected)}")
    else:
        print("Note: no checkpoint provided/found. Using randomly initialized weights.")

    # Tokenizer for decoding
    sp = spm.SentencePieceProcessor()
    sp.load(args.tokenizer_path)

    # Load and preprocess video/audio
    t0 = time.time()
    if args.pt:
        video, audio, video_fps, audio_sr = load_from_preprocessed_pt(args.pt, force_silent=args.video_only)
    else:
        if not args.video:
            ap.error("Either --pt or --video must be provided")
        video, audio, video_fps, audio_sr = load_video_audio(args.video, force_silent=args.video_only)
    # Build shapes for model.encoder:
    # video: (B, C, T, H, W), audio: (B, N)
    B = 1
    T = video.shape[0]
    H, W = video.shape[1], video.shape[2]

    video_len = torch.tensor([T], dtype=torch.long)
    audio_len = torch.tensor([audio.shape[-1]], dtype=torch.long)

    video_bc_thwc = video.unsqueeze(0)  # (1, T, H, W, 1)
    video_bcthw = video_bc_thwc.permute(0, 4, 1, 2, 3).contiguous()  # (1, 1, T, H, W)
    # Ensure audio shape is (B, T) as expected by AudioPreprocessing
    audio_bt = audio  # already (1, N)

    video_bcthw = to_device(video_bcthw, device)
    audio_bt = to_device(audio_bt, device)
    video_len = to_device(video_len, device)
    audio_len = to_device(audio_len, device)

    with torch.no_grad():
        # Use the underlying encoder directly
        x, x_len, _ = model.encoder(video_bcthw, video_len, audio_bt, audio_len)  # x: (B, T', V)
        # Some models output (B, T', V), others may add a dict; assume logits in x
        if x.dim() != 3:
            raise RuntimeError(f"Unexpected model output shape: {tuple(x.shape)}")

        # Greedy CTC
        transcripts = ctc_greedy_decode(x, x_len, sp, blank_id=args.blank_id)

    dt = time.time() - t0
    print(f"Transcript: {transcripts[0]}")
    print(f"Time: {dt:.2f}s")

if __name__ == "__main__":
    main()