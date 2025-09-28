import sys
sys.path.append("../../")

# Imports
import nnet
import torch
import torch.nn as nn
from nnet import transforms as vtf

# Transforms
crop_size = (88, 88)
train_video_transform = nn.Sequential(
    vtf.RandomTemporalShift(max_shift=2),
    vtf.RandomCropVideo(crop_size),
    vtf.RandomHorizontalFlipVideo(p=0.5),
)
val_video_transform = vtf.CenterCropVideo(crop_size)

# Datasets (use indices produced by preprocess_lipbengal.py; labels are Banglish)
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
    prepared_only=False,
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
    prepared_only=False,
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

# Preview samples after each eval pass
model.preview_samples_per_epoch = 4

# Train/eval settings
epochs = 50
precision = torch.float16
accumulated_steps = 1
eval_training = False
callback_path = "callbacks/LipBengal/AV/VisualCE_banglish_train50"

# Logging cadence
verbose_eval = 1
eval_period_epoch = 1
log_figure_period_epoch = 1

# Limit validation steps per epoch to shorten wall-clock time
eval_steps = 100

# Evaluate only on validation set (faster; early stopping uses this)
evaluation_dataset = validation_dataset

# Early stopping on validation WER
early_stopping_metric = "wer"
early_stopping_mode = "min"
early_stopping_patience = 5
early_stopping_min_delta = 0.0
