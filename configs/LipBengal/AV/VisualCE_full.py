import sys
sys.path.append("../../")

# Imports
import nnet
import torch
import torch.nn as nn
import torchvision
from nnet import transforms as vtf

# Transforms
crop_size = (88, 88)
# Use video-aware transforms that operate on (C, T, H, W)
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
    subset_fraction=1.0,  # full train set
    subset_seed=0,
    prepared_only=True,
)
validation_dataset = nnet.datasets.LipBengal(
    batch_size=64,
    collate_fn=collate_fn,
    mode="val",
    video_transform=val_video_transform,
    fixed_frames=29,
    indices_path="datasets/LipBengal/indices/val.pt",
    subset_fraction=1.0,
    subset_seed=0,
    prepared_only=True,
)
test_dataset = nnet.datasets.LipBengal(
    batch_size=64,
    collate_fn=collate_fn,
    mode="test",
    video_transform=val_video_transform,
    fixed_frames=29,
    indices_path="datasets/LipBengal/indices/test.pt",
    subset_fraction=1.0,
    subset_seed=0,
    prepared_only=True,
)

# Model
vocab_size = training_dataset.num_classes
model = nnet.VisualEfficientConformerCE(vocab_size=vocab_size)

# Label->word decoder for WER/CER
_words = training_dataset.classes
_label_to_word = nnet.ClassLabelToWordDecoder(_words)

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
    optimizer="AdamW",
    grad_max_norm=1.0,
    ema_tau=0.999,
)

# Increase preview samples printed after each eval pass
model.preview_samples_per_epoch = 8

# Train/eval settings
epochs = 50
precision = torch.float16
accumulated_steps = 1
eval_training = False
callback_path = "callbacks/LipBengal/AV/VisualCE_full"

# Verbosity and logging
verbose_eval = 1
eval_period_epoch = 1
log_figure_period_epoch = 1

# Full evaluation (no cap on steps)
eval_steps = None

# Evaluate on both val and test each time
evaluation_dataset = [validation_dataset, test_dataset]

# Early stopping: monitor word error rate (WER) on primary eval dataset (val)
early_stopping_metric = "wer"   # from nnet.metrics.WordErrorRate
early_stopping_mode = "min"
early_stopping_patience = 5
early_stopping_min_delta = 0.0
