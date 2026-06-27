#!/bin/bash
#
# Evaluate the LATEST checkpoint on LipBengal test dataset
#

set -e

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# Activate virtual environment
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Find latest checkpoint automatically
LATEST_CKPT=$(ls -t callbacks/LipBengal/AV/VisualCE/checkpoints_epoch_*.ckpt | head -n 1)

if [ -z "$LATEST_CKPT" ]; then
    echo "Error: No checkpoint found."
    exit 1
fi

echo "========================================"
echo "Evaluating Latest Checkpoint"
echo "========================================"
echo "Checkpoint: $LATEST_CKPT"
echo "Dataset: LipBengal Test Set"
echo ""

# Create a python script on the fly to run evaluation
# We reuse the logic from eval_epoch177.py but make it dynamic
cat << EOF > eval_latest_temp.py
import torch
import importlib
import functions
import nnet
import torch.nn.functional as F
import random
from pathlib import Path
import sys

# Config
checkpoint_path = "$LATEST_CKPT"
config_path = 'configs.LipBengal.AV.VisualCE'

print(f"Loading config: {config_path}")
config = importlib.import_module(config_path)

# Collate
collate_fn = nnet.CollateFn(inputs_params=[{"axis": 0}], targets_params=[{"axis": 2}])

# Dataset
print("Loading LipBengal test dataset...")
dataset = nnet.datasets.LipBengal(
    batch_size=64,
    collate_fn=collate_fn,
    mode='test',
    video_transform=config.val_video_transform,
    fixed_frames=29,
    indices_path='datasets/LipBengal/indices/test.pt',
    prepared_only=True
)
print(f"Test dataset loaded: {len(dataset)} samples")

loader = torch.utils.data.DataLoader(
    dataset=dataset,
    batch_size=dataset.batch_size,
    shuffle=False,
    num_workers=4,
    collate_fn=dataset.collate_fn,
    pin_memory=True
)

# Args
class Args: pass
args = Args()
args.rank = 0
args.cpu = False
args.distributed = False
args.parallel = False
args.show_dict = False
args.show_modules = False
args.config_file = 'configs/LipBengal/AV/VisualCE.py'
args.config = config
args.load_last = False
args.checkpoint = None
args.mode = 'evaluation'
args.batch_size_eval = None
args.num_workers = 4
args.world_size = 1
args.dist_log = False

print(f"Loading model from {checkpoint_path}...")
model = functions.load_model(args)
model.load(checkpoint_path, load_optimizer=False)
model.eval()

label_to_word = {i: w for i, w in enumerate(dataset.classes)}

records = []
print("Starting evaluation...")
with torch.no_grad():
    for batch_idx, batch in enumerate(loader):
        vids = batch['inputs'].to(model.device)
        labels = batch['targets']
        
        out = model.forward(vids)
        logp = F.log_softmax(out, dim=1)
        
        # Top-k
        k = 10
        topk = torch.topk(logp, k=k, dim=1)
        indices = topk.indices.cpu()
        values = topk.values.cpu()
        
        for i in range(labels.size(0)):
            label = int(labels[i])
            preds = indices[i].tolist()
            pred_words = [label_to_word[p] for p in preds]
            
            top1_correct = preds[0] == label
            top10_correct = label in preds
            
            records.append({
                'top1_correct': top1_correct,
                'top10_correct': top10_correct,
                'true_word': label_to_word[label],
                'pred_word': pred_words[0]
            })
            
        if (batch_idx + 1) % 20 == 0:
            print(f"Processed {len(records)} samples...")

# Metrics
top1 = sum(1 for r in records if r['top1_correct']) / len(records) * 100
top10 = sum(1 for r in records if r['top10_correct']) / len(records) * 100

wer_metric = nnet.WordErrorRate()
cer_metric = nnet.CharacterErrorRate()
true_words = [r['true_word'] for r in records]
pred_words = [r['pred_word'] for r in records]

wer = wer_metric(true_words, pred_words).item()
cer = cer_metric(true_words, pred_words).item()

print(f"\n{'='*60}")
print(f"RESULTS for {checkpoint_path}")
print(f"{'='*60}")
print(f"Top-1 Accuracy:  {top1:.2f}%")
print(f"Top-10 Accuracy: {top10:.2f}%")
print(f"WER:             {wer:.2f}%")
print(f"CER:             {cer:.2f}%")
print(f"{'='*60}")

EOF

# Run the python script
python3 eval_latest_temp.py 2>&1 | tee logs/eval_latest.log

# Cleanup
rm eval_latest_temp.py







