import sys
sys.path.append("../../")

# Imports
import nnet
import torch
import torch.nn as nn
import torchvision

# Vocabulary: classes are Bengali words discovered from dataset
# We'll instantiate the dataset first to infer num_classes, then feed it to model

# Transforms
crop_size = (88, 88)
train_video_transform = nn.Sequential(
    torchvision.transforms.RandomCrop(crop_size),
    torchvision.transforms.RandomHorizontalFlip()
)
val_video_transform = torchvision.transforms.CenterCrop(crop_size)

# Datasets
collate_fn = nnet.CollateFn(inputs_params=[{"axis": 0}], targets_params=[{"axis": 2}])
training_dataset = nnet.datasets.LipBengal(
    batch_size=32,
    collate_fn=collate_fn,
    mode="train",
    video_transform=train_video_transform,
    num_frames=None,
)
evaluation_dataset = nnet.datasets.LipBengal(
    batch_size=32,
    collate_fn=collate_fn,
    mode="val",
    video_transform=val_video_transform,
)

# Model
vocab_size = training_dataset.num_classes
model = nnet.VisualEfficientConformerCE(vocab_size=vocab_size)
model.compile(
    losses=nnet.SoftmaxCrossEntropy(),
    metrics=nnet.CategoricalAccuracy(),
)

# Training
epochs = 30
precision = torch.float16
accumulated_steps = 1
eval_training = False
callback_path = "callbacks/LipBengal/AV/VisualCE"
