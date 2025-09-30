import sys
sys.path.append("../../")

# Imports
import nnet
import torch
import torch.nn as nn
import torchvision
from nnet import transforms as vtf

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

# Model
vocab_size = training_dataset.num_classes
model = nnet.VisualEfficientConformerCE(vocab_size=vocab_size)

# For CE classification, we want both accuracy and WER on the decoded words.
# Build a label->word decoder from the dataset's class list.
_words = training_dataset.classes
_label_to_word = nnet.ClassLabelToWordDecoder(_words)

model.compile(
    losses=nnet.SoftmaxCrossEntropy(label_smoothing=0.1),
    metrics={
        "output": [
            nnet.CategoricalAccuracy(),           # Top-1 classification accuracy
            nnet.WordErrorRate(),                 # WER over words
            nnet.CharacterErrorRate(),            # CER over characters of the word
        ]
    },
    decoders={
        "output": [
            None,                                 # No decoder for accuracy (uses argmax internally)
            _label_to_word,                       # Map class ids/logits to word strings for WER/CER
            _label_to_word,
        ]
    },
    optimizer="AdamW",
    grad_max_norm=1.0,
    ema_tau=0.999,
)

# Training
# Extend to 200 total epochs so that resuming with --load_last from an epoch 100
# checkpoint continues with epochs 101..200 (100 additional epochs).
epochs = 1000
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
    # Conservative ceiling; anneal within actual step horizon
    model.optimizer.scheduler = nnet.schedulers.CosineAnnealingScheduler(
        warmup_steps=200,
        val_max=1e-4,
        val_min=1e-5,
        end_step=total_steps,
    )
except Exception:
    # Fallback silently if any object missing during import-time dry runs
    pass

