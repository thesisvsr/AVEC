import sys
sys.path.append("../../")

import nnet
import torch
import torch.nn as nn
import torchvision

vocab_size = 256
v_interctc_blocks = [3, 6]
a_interctc_blocks = [8, 11]
f_interctc_blocks = [2]
loss_weights={
    "v_ctc_2": 0.5 / 3,
    "v_ctc_5": 0.5 / 3,
    "a_ctc_7": 0.0,
    "a_ctc_10": 0.0,
    "f_ctc_1": 0.5 / 3,
    "outputs": 0.5
}

tokenizer_path = "datasets/LRS3/tokenizerbpe256.model"

batch_size = 2
accumulated_steps = 32
eval_training = False
precision = torch.float32
recompute_metrics = True
callback_path = "callbacks/LRS2/AV/EffConfAdaptiveCTC_Mid"

model = nnet.AudioVisualEfficientConformerInterCTC(
    vocab_size=vocab_size,
    v_interctc_blocks=v_interctc_blocks,
    a_interctc_blocks=a_interctc_blocks,
    f_interctc_blocks=f_interctc_blocks,
)

adaptive_decoder = nnet.AdaptiveCTCDecoder(
    tokenizer_path=tokenizer_path,
    entropy_low=0.6,
    entropy_high=1.2,
    small_beam=8,
    large_beam=32,
    exclude_blank=True,
    ngram_path=None,
    neural_config_path=None,
    neural_checkpoint=None,
)

model.compile(
    losses={
        "outputs": nnet.CTCLoss(zero_infinity=True, assert_shorter=False)
    },
    decoders={
        "outputs": adaptive_decoder
    },
    metrics={"outputs": [nnet.WordErrorRate(), nnet.CharacterErrorRate()]},
    loss_weights=loss_weights,
)

align = False
video_max_length = 150
collate_fn = nnet.CollateFn(
    inputs_params=[{"axis": 0, "padding": True}, {"axis": 3}, {"axis": 1, "padding": True}, {"axis": 4}],
    targets_params=(
        {"axis": 2, "padding": True},
        {"axis": 5}
    ),
)
crop_size = (88, 88)
training_video_transform = nn.Sequential(
    torchvision.transforms.RandomCrop(crop_size),
    torchvision.transforms.RandomHorizontalFlip(),
    nnet.Permute(dims=(2, 3, 0, 1)),
    nnet.TimeMaskSecond(T_second=0.4, num_mask_second=1.0, fps=25.0, mean_frame=True),
    nnet.Permute(dims=(2, 3, 0, 1)),
)
evaluation_video_transform = torchvision.transforms.CenterCrop(crop_size)

training_dataset = nnet.datasets.LRS(
    batch_size=batch_size,
    collate_fn=collate_fn,
    version="LRS2",
    mode="pretrain+train+val",
    video_max_length=video_max_length,
    video_transform=training_video_transform,
    align=align,
    subset_fraction=0.3,
)

evaluation_dataset = [
    nnet.datasets.LRS(
        batch_size=batch_size,
        collate_fn=collate_fn,
        version="LRS2",
        mode="test",
        video_transform=evaluation_video_transform,
        align=align,
        subset_fraction=1.0,
    )
]
