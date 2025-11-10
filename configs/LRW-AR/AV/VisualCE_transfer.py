import os
import sys
sys.path.append("../../")

# Transfer-learning configuration for LRW-AR (Arabish transliteration)
# ------------------------------------------------------------------
# Mirrors the LipBengal transfer setup:
#  1. Differential learning rates (encoder vs head) via lr_scale in param groups.
#  2. Initial encoder freezing for a few epochs then automatic unfreeze.
#  3. Cosine LR schedule with modest range tuned for small target fine-tune.
#  4. Label smoothing + accuracy / WER / CER metrics with Arabish transliteration.
#  5. Environment-variable overrides for quick experimentation.
#
# Usage example:
#   python main.py -c configs/LRW-AR/AV/VisualCE_transfer.py -m training \
#       --checkpoint path/to/source_pretrained.ckpt -j 2
#
# Optional environment variables:
#   LRWAR_FREEZE_EPOCHS (int, default 5)
#   LRWAR_ENCODER_LR_SCALE (float, default 0.2)
#   LRWAR_EPOCHS (int, default 600)
#   LRWAR_LR_MAX (float, default 5e-5)
#   LRWAR_LR_MIN (float, default 5e-6)
#   LRWAR_WARMUP_STEPS (int, default 100)
#
# Provide a pretrained checkpoint from (e.g.) LRS2 / LRS3 / LRS23 / multilingual
# model trained on similar architecture for optimal transfer.

import torch
import torch.nn as nn

import nnet
from nnet import transforms as vtf
from nnet import optimizers as _optim
from nnet import schedulers as _schedulers

# ---------------------------------------------------------------------------
# Data & Transforms
# ---------------------------------------------------------------------------

crop_size = (88, 88)
center_crop = (96, 96)

train_video_transform = nn.Sequential(
    vtf.RandomTemporalShift(max_shift=2),
    vtf.RandomCropVideo(crop_size),
    vtf.RandomHorizontalFlipVideo(p=0.5),
)

val_video_transform = vtf.CenterCropVideo(crop_size)

test_video_transform = vtf.CenterCropVideo(crop_size)

collate_fn = nnet.CollateFn(inputs_params=[{"axis": 0}], targets_params=[{"axis": 2}])

training_dataset = nnet.datasets.LRWAR(
    batch_size=32,
    collate_fn=collate_fn,
    mode="train",
    video_transform=train_video_transform,
    center_crop_size=center_crop,
    fixed_frames=29,
    load_audio=False,
    indices_path="datasets/LRW-AR/indices/train.pt",
    prepared_only=True,
    use_arabish=True,  # transliteration
)

evaluation_dataset = nnet.datasets.LRWAR(
    batch_size=64,
    collate_fn=collate_fn,
    mode="val",
    video_transform=val_video_transform,
    center_crop_size=center_crop,
    fixed_frames=29,
    load_audio=False,
    indices_path="datasets/LRW-AR/indices/val.pt",
    prepared_only=True,
    use_arabish=True,
)

test_dataset = nnet.datasets.LRWAR(
    batch_size=64,
    collate_fn=collate_fn,
    mode="test",
    video_transform=test_video_transform,
    center_crop_size=center_crop,
    fixed_frames=29,
    load_audio=False,
    indices_path="datasets/LRW-AR/indices/test.pt",
    prepared_only=True,
    use_arabish=True,
)

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

vocab_size = training_dataset.num_classes
model = nnet.VisualEfficientConformerCE(vocab_size=vocab_size)

# ---------------------------------------------------------------------------
# Transfer knobs (env overrides)
# ---------------------------------------------------------------------------

freeze_encoder_epochs = int(os.environ.get("LRWAR_FREEZE_EPOCHS", 5))
encoder_lr_scale = float(os.environ.get("LRWAR_ENCODER_LR_SCALE", 0.2))
_epochs = int(os.environ.get("LRWAR_EPOCHS", 600))
lr_max = float(os.environ.get("LRWAR_LR_MAX", 5e-5))
lr_min = float(os.environ.get("LRWAR_LR_MIN", 5e-6))
warmup_steps = int(os.environ.get("LRWAR_WARMUP_STEPS", 100))

# ---------------------------------------------------------------------------
# Optimizer with differential LR scaling (copied / adapted from LipBengal)
# ---------------------------------------------------------------------------

def build_param_groups_with_scaling(model: torch.nn.Module, encoder_lr_scale: float):
    decay_modules = (torch.nn.Linear, torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d)
    no_decay_modules = (torch.nn.LayerNorm, torch.nn.Embedding, torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)
    decay_params = ("weight",)
    no_decay_params = ("bias",)

    enc_decay, enc_no_decay, head_decay, head_no_decay = [], [], [], []
    for module_name, module in model.named_modules():
        for param_name, param in module.named_parameters(recurse=False):
            full_name = f"{module_name}.{param_name}" if module_name else param_name
            is_encoder = full_name.startswith("encoder.")
            target_decay = enc_decay if is_encoder else head_decay
            target_no_decay = enc_no_decay if is_encoder else head_no_decay
            placed = False
            for nd in no_decay_params:
                if param_name.endswith(nd):
                    target_no_decay.append(param)
                    placed = True
                    break
            if placed:
                continue
            for dp in decay_params:
                if param_name.endswith(dp) and isinstance(module, decay_modules):
                    target_decay.append(param)
                    placed = True
                    break
                if param_name.endswith(dp) and isinstance(module, no_decay_modules):
                    target_no_decay.append(param)
                    placed = True
                    break
            if not placed:
                target_no_decay.append(param)

    return [
        {"params": enc_decay, "weight_decay": 0.05, "lr_scale": encoder_lr_scale},
        {"params": enc_no_decay, "weight_decay": 0.0,  "lr_scale": encoder_lr_scale},
        {"params": head_decay, "weight_decay": 0.05, "lr_scale": 1.0},
        {"params": head_no_decay, "weight_decay": 0.0,  "lr_scale": 1.0},
    ]


class MultiScaleAdamW(_optim.AdamW):
    def step(self, closure=None):  # type: ignore[override]
        base_lr = self.scheduler.step()
        for group in self.param_groups:
            scale = group.get("lr_scale", 1.0)
            group['lr'] = base_lr * scale
        return super(_optim.AdamW, self).step(closure)

# Build label->word decoder (Arabish already applied inside dataset logic)
_words = training_dataset.classes
_label_to_word = nnet.ClassLabelToWordDecoder(_words)

# Scheduler (will set end_step precisely after dataset length known)
transfer_scheduler = _schedulers.CosineAnnealingScheduler(
    warmup_steps=warmup_steps,
    val_max=lr_max,
    val_min=lr_min,
    end_step=200000,  # temporary; corrected below
)

param_groups = build_param_groups_with_scaling(model, encoder_lr_scale)
optimizer = MultiScaleAdamW(
    params=param_groups,
    lr=transfer_scheduler,
    betas=(0.9, 0.98),
    eps=1e-8,
)

model.compile(
    losses=nnet.SoftmaxCrossEntropy(label_smoothing=0.1),
    metrics={
        "output": [
            nnet.CategoricalAccuracy(),
            nnet.WordErrorRate(),
            nnet.CharacterErrorRate(),
        ]
    },
    decoders={
        "output": [
            None,
            _label_to_word,
            _label_to_word,
        ]
    },
    optimizer=optimizer,
    grad_max_norm=1.0,
    ema_tau=0.999,
)

# ---------------------------------------------------------------------------
# Freezing & Unfreezing
# ---------------------------------------------------------------------------

def _freeze_backbone_keep_head(m):
    head_params = set(id(p) for p in getattr(m.encoder, 'head').parameters()) if hasattr(m.encoder, 'head') else set()
    frozen = kept = 0
    for p in m.encoder.parameters():
        if id(p) in head_params:
            p.requires_grad = True
            kept += 1
        else:
            p.requires_grad = False
            frozen += 1
    return frozen, kept

if freeze_encoder_epochs > 0:
    frozen, kept = _freeze_backbone_keep_head(model)
    if not hasattr(model, 'rank') or model.rank == 0:
        print(f"[Transfer][LRW-AR] Frozen backbone params: {frozen}, trainable head params: {kept} for first {freeze_encoder_epochs} epochs.")

_orig_on_epoch_end = model.on_epoch_end

def _transfer_on_epoch_end(evaluate, save, log_figure, callback_path, epoch, *a, **kw):
    res = _orig_on_epoch_end(evaluate, save, log_figure, callback_path, epoch, *a, **kw)
    if epoch == freeze_encoder_epochs:
        unfroze = 0
        for p in model.encoder.parameters():
            if not p.requires_grad:
                p.requires_grad = True
                unfroze += 1
        if unfroze and (not hasattr(model, 'rank') or model.rank == 0):
            print(f"[Transfer][LRW-AR] Unfroze {unfroze} backbone params at epoch {epoch}.")
    return res

model.on_epoch_end = _transfer_on_epoch_end  # type: ignore

# ---------------------------------------------------------------------------
# Optional Transfer Encoder Weight Loading (from LRS23 InterCTC SWA)
# Set LRWAR_SOURCE_CKPT env or rely on default path.
# ---------------------------------------------------------------------------
source_ckpt = os.environ.get("LRWAR_SOURCE_CKPT", "callbacks/LRS23/VO/EffConfInterCTC/checkpoints_swa-equal-90-100.ckpt")
_loaded_enc = False
try:
    ck = torch.load(source_ckpt, map_location="cpu")
    state = ck.get("model_state_dict", {})
    msd = model.state_dict()
    loadable = {}
    for k, v in state.items():
        if (k.startswith("encoder.front_end") or k.startswith("encoder.back_end")) and "head" not in k:
            if k in msd and getattr(v, 'shape', None) == getattr(msd[k], 'shape', None):
                loadable[k] = v
    if loadable:
        msd.update(loadable)
        model.load_state_dict(msd, strict=False)
        _loaded_enc = True
        if not hasattr(model, 'rank') or model.rank == 0:
            print(f"[Transfer][LRW-AR] Loaded {len(loadable)} encoder tensors from {source_ckpt}")
except Exception as e:
    if not hasattr(model, 'rank') or model.rank == 0:
        print(f"[Transfer][LRW-AR] Encoder load failed: {e}")

# ---------------------------------------------------------------------------
# Training Hyperparameters
# ---------------------------------------------------------------------------

epochs = _epochs
precision = torch.float16
accumulated_steps = 1
eval_training = False
callback_path = "callbacks/LRW-AR/AV/VisualCE"
verbose_eval = 1
eval_period_epoch = 1
log_figure_period_epoch = 1
saving_period_epoch = 1

# Adjust scheduler end_step now that dataset length is known
try:
    total_steps = epochs * len(training_dataset)
    model.optimizer.scheduler.end_step = total_steps
except Exception:
    pass
