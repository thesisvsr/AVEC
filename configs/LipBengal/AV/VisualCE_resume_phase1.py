import sys
sys.path.append("../../")

# Imports
import nnet
import torch
import torch.nn as nn
import torchvision
from nnet import transforms as vtf
from nnet import optimizers as _optim
from nnet import schedulers as _schedulers

"""Transfer-aware LipBengal Visual CE configuration

Adds:
 1. Differential learning rates (encoder vs head) via per-param-group scaling.
 2. Initial encoder freezing for a number of epochs (freeze_encoder_epochs).
 3. Tighter cosine LR schedule suitable for small target dataset transfer.
 4. Automatic unfreezing hook injected into model.on_epoch_end.

Usage (after preparing transfer checkpoint with scripts/transfer_init_lipbengal_from_lrs2.py):
    python main.py -c configs/LipBengal/AV/VisualCE.py -m training \
            --checkpoint transfer_from_LRS2_swa.ckpt -j 2

Environment variables / CLI overrides can still adjust epochs etc. Adjust
freeze_encoder_epochs / encoder_lr_scale below if needed.
"""

# Vocabulary: classes are Bengali words discovered from dataset
# We'll instantiate the dataset first to infer num_classes, then feed it to model

# Transforms
crop_size = (88, 88)
train_video_transform = nn.Sequential(
    vtf.RandomTemporalShift(max_shift=2),
    vtf.RandomCropVideo(crop_size),
    vtf.RandomHorizontalFlipVideo(p=0.5),
)
val_video_transform = vtf.CenterCropVideo(crop_size)

# Datasets
collate_fn = nnet.CollateFn(inputs_params=[{"axis": 0}], targets_params=[{"axis": 2}])
training_dataset = nnet.datasets.LipBengal(
    batch_size=32,
    collate_fn=collate_fn,
    mode="train",
    video_transform=train_video_transform,
    num_frames=None,
    fixed_frames=29,
    indices_path="datasets/LipBengal/indices/train.pt",
    subset_fraction=1.0,
    subset_seed=0,
    prepared_only=True,
)
evaluation_dataset = nnet.datasets.LipBengal(
    batch_size=64,  # larger batch for faster eval
    collate_fn=collate_fn,
    mode="val",
    video_transform=val_video_transform,
    fixed_frames=29,
    indices_path="datasets/LipBengal/indices/val.pt",
    subset_fraction=1.0,
    subset_seed=0,
    prepared_only=True,
)

###############################################################################
# Model
###############################################################################
vocab_size = training_dataset.num_classes
model = nnet.VisualEfficientConformerCE(vocab_size=vocab_size)

# --- Transfer specific knobs ---
# How many initial epochs to keep the encoder frozen (feature extractor only)
freeze_encoder_epochs = 5
# Relative LR multiplier applied to all encoder param groups (after unfreeze)
encoder_lr_scale = 0.2  # encoder learns slower than new head

###############################################################################
# Optimizer with differential LR scaling
###############################################################################

def build_param_groups_with_scaling(model: torch.nn.Module, encoder_lr_scale: float):
    """Create 4 param groups (enc/ head) x (decay / no_decay) each with lr_scale.

    We replicate logic from nnet.optimizers.get_decay_param_groups but split
    encoder vs head so we can apply different per-group scaling factors.
    The base optimizer's step() will multiply the scheduler LR by lr_scale.
    """
    decay_modules = (torch.nn.Linear, torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d)
    no_decay_modules = (torch.nn.LayerNorm, torch.nn.Embedding, torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)
    decay_params = ("weight",)
    no_decay_params = ("bias",)

    enc_decay, enc_no_decay, head_decay, head_no_decay = [], [], [], []
    for module_name, module in model.named_modules():
        for param_name, param in module.named_parameters(recurse=False):
            full_name = f"{module_name}.{param_name}" if module_name else param_name
            target_list_decay = enc_decay if full_name.startswith("encoder.") else head_decay
            target_list_no_decay = enc_no_decay if full_name.startswith("encoder.") else head_no_decay
            # Decide decay vs no-decay
            placed = False
            for nd in no_decay_params:
                if param_name.endswith(nd):
                    target_list_no_decay.append(param)
                    placed = True
                    break
            if placed:
                continue
            for dp in decay_params:
                if param_name.endswith(dp) and isinstance(module, decay_modules):
                    target_list_decay.append(param)
                    placed = True
                    break
                if param_name.endswith(dp) and isinstance(module, no_decay_modules):
                    target_list_no_decay.append(param)
                    placed = True
                    break
            if not placed:  # default: no decay
                target_list_no_decay.append(param)

    param_groups = [
        {"params": enc_decay, "weight_decay": 0.05, "lr_scale": encoder_lr_scale},
        {"params": enc_no_decay, "weight_decay": 0.0,  "lr_scale": encoder_lr_scale},
        {"params": head_decay, "weight_decay": 0.05, "lr_scale": 1.0},
        {"params": head_no_decay, "weight_decay": 0.0,  "lr_scale": 1.0},
    ]
    return param_groups


class MultiScaleAdamW(_optim.AdamW):
    """AdamW that applies per-param-group lr_scale to the shared scheduler LR."""
    def step(self, closure=None):  # type: ignore[override]
        base_lr = self.scheduler.step()
        for group in self.param_groups:
            scale = group.get("lr_scale", 1.0)
            group['lr'] = base_lr * scale
        return super(_optim.AdamW, self).step(closure)  # call grandparent AdamW

# For CE classification, we want both accuracy and WER on the decoded words.
# Build a label->word decoder from the dataset's class list.
_words = training_dataset.classes
_label_to_word = nnet.ClassLabelToWordDecoder(_words)

###############################################################################
# Compile with custom optimizer (differential LR + scaling-aware scheduler)
###############################################################################

# Cosine schedule tuned for transfer (smaller warmup & LR range)
# Total steps estimated below after dataset instantiation (recomputed later).
transfer_scheduler = _schedulers.CosineAnnealingScheduler(
    warmup_steps=100,  # quick warm start
    val_max=5e-5,
    val_min=5e-6,
    end_step=200000,   # large ceiling; will be reset precisely after dataset length known
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

def _freeze_backbone_keep_head(m):
    """Freeze everything in encoder except the final classification head module."""
    # Heuristic: encoder.head is the classification head (Linear or Sequential containing Linear)
    head_params = set(id(p) for p in m.encoder.head.parameters()) if hasattr(m.encoder, 'head') else set()
    frozen, kept = 0, 0
    for p in m.encoder.parameters():
        if id(p) in head_params:
            p.requires_grad = True
            kept += 1
        else:
            p.requires_grad = False
            frozen += 1
    return frozen, kept

# Freeze encoder backbone (keep head trainable) initially
if freeze_encoder_epochs > 0:
    frozen, kept = _freeze_backbone_keep_head(model)
    if hasattr(model, 'rank') and model.rank == 0:
        print(f"[Transfer] Frozen backbone params: {frozen}, trainable head params: {kept} for first {freeze_encoder_epochs} epochs.")

# Inject hook to unfreeze & optionally adjust lr scaling mid-training
_orig_on_epoch_end = model.on_epoch_end
def _transfer_on_epoch_end(evaluate, save, log_figure, callback_path, epoch, *a, **kw):
    res = _orig_on_epoch_end(evaluate, save, log_figure, callback_path, epoch, *a, **kw)
    # epoch argument is 1-based in on_epoch_end (passed as epoch+1 in fit)
    if epoch == freeze_encoder_epochs:
        unfroze = 0
        for p in model.encoder.parameters():
            if not p.requires_grad:
                p.requires_grad = True
                unfroze += 1
        if unfroze and (not hasattr(model, 'rank') or model.rank == 0):
            print(f"[Transfer] Unfroze {unfroze} backbone params at epoch {epoch} (encoder now fully trainable).")
    return res
model.on_epoch_end = _transfer_on_epoch_end  # type: ignore

# Training
# Reduced total epochs for transfer fine-tuning
epochs = 250
precision = torch.float16
accumulated_steps = 1
eval_training = False
callback_path = "callbacks/LipBengal/AV/VisualCE"

# Show a few qualitative samples after each epoch
verbose_eval = 1
eval_period_epoch = 1
log_figure_period_epoch = 1
# Ensure checkpoints are saved every epoch (CLI can also override)
saving_period_epoch = 1

# Cap the number of validation steps per epoch to shorten eval time
# eval_steps = 50

# Tighter LR schedule tuned to this run's step budget
# Rationale: avoid LR growing too high relative to total steps (observed collapse).
try:
    total_steps = epochs * len(training_dataset)
    # Rebuild scheduler with correct end_step reflecting dataset size
    model.optimizer.scheduler.end_step = total_steps
except Exception:
    pass

