import sys
sys.path.append("../../")

import nnet
import torch
import torch.nn as nn
from nnet import transforms as vtf

# Training hyper-parameters
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
    fixed_frames=30,
    load_audio=False,
    indices_path="datasets/LRW-AR/indices/train.pt",  # will fallback to raw if missing
    prepared_only=True,
    use_arabish=True,
)

evaluation_dataset = nnet.datasets.LRWAR(
    batch_size=64,
    collate_fn=collate_fn,
    mode="val",
    video_transform=val_video_transform,
    center_crop_size=center_crop,
    fixed_frames=30,
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
    fixed_frames=30,
    load_audio=False,
    indices_path="datasets/LRW-AR/indices/test.pt",
    prepared_only=True,
    use_arabish=True,
)

vocab_size = training_dataset.num_classes
model = nnet.VisualEfficientConformerCE(vocab_size=vocab_size)

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

epochs = 400
precision = torch.float16
accumulated_steps = 1
eval_training = False
callback_path = "callbacks/LRW-AR/AV/VisualCE"
verbose_eval = 1
eval_period_epoch = 1
log_figure_period_epoch = 1
saving_period_epoch = 1

try:
    total_steps = epochs * len(training_dataset)
    model.optimizer.scheduler = nnet.schedulers.CosineAnnealingScheduler(
        warmup_steps=200,
        val_max=1e-4,
        val_min=1e-5,
        end_step=total_steps,
    )
except Exception:
    pass
