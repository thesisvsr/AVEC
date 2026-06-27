#!/usr/bin/env python3
"""
Comprehensive Top-K Accuracy Evaluation Script
Evaluates a checkpoint with Top-1, Top-3, Top-5, Top-10, Top-20 accuracy + WER/CER
"""

import torch
import importlib
import functions
import nnet
import torch.nn.functional as F
import sys
import os
from pathlib import Path

def main():
    # Find latest checkpoint
    ckpt_dir = Path('callbacks/LipBengal/AV/VisualCE')
    ckpts = sorted(ckpt_dir.glob('checkpoints_epoch_*.ckpt'), 
                   key=lambda x: os.path.getmtime(x), reverse=True)
    
    if not ckpts:
        print("Error: No checkpoints found!")
        sys.exit(1)
    
    checkpoint_path = str(ckpts[0])
    
    # Extract epoch number from checkpoint name
    epoch_num = checkpoint_path.split('epoch_')[1].split('_')[0]
    
    print("=" * 80)
    print(f"  COMPREHENSIVE TOP-K EVALUATION - EPOCH {epoch_num}")
    print("=" * 80)
    print(f"\nCheckpoint: {checkpoint_path}")
    
    # Config
    config_path = 'configs.LipBengal.AV.VisualCE'
    print(f"Config: {config_path}")
    config = importlib.import_module(config_path)

    # Collate
    collate_fn = nnet.CollateFn(inputs_params=[{"axis": 0}], targets_params=[{"axis": 2}])

    # Dataset
    print("\nLoading LipBengal test dataset...")
    dataset = nnet.datasets.LipBengal(
        batch_size=64,
        collate_fn=collate_fn,
        mode='test',
        video_transform=config.val_video_transform,
        fixed_frames=29,
        indices_path='datasets/LipBengal/indices/test.pt',
        prepared_only=True
    )
    print(f"Test samples: {len(dataset):,}")

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

    print(f"\nLoading model...")
    model = functions.load_model(args)
    model.load(checkpoint_path, load_optimizer=False)
    model.eval()

    label_to_word = {i: w for i, w in enumerate(dataset.classes)}

    # Storage for results
    all_labels = []
    all_top1_preds = []
    all_top3_preds = []
    all_top5_preds = []
    all_top10_preds = []
    all_top20_preds = []
    
    print("\nRunning evaluation...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            vids = batch['inputs'].to(model.device)
            labels = batch['targets']
            
            out = model.forward(vids)
            logp = F.log_softmax(out, dim=1)
            
            # Top-20 (we'll extract smaller k from this)
            topk = torch.topk(logp, k=20, dim=1)
            indices = topk.indices.cpu()
            
            for i in range(labels.size(0)):
                label = int(labels[i])
                preds = indices[i].tolist()
                
                all_labels.append(label)
                all_top1_preds.append(preds[:1])
                all_top3_preds.append(preds[:3])
                all_top5_preds.append(preds[:5])
                all_top10_preds.append(preds[:10])
                all_top20_preds.append(preds[:20])
                
            if (batch_idx + 1) % 50 == 0:
                print(f"  Processed {len(all_labels):,} / {len(dataset):,} samples...")

    # Calculate Top-K accuracies
    def calc_topk_acc(labels, topk_preds):
        correct = sum(1 for l, preds in zip(labels, topk_preds) if l in preds)
        return correct / len(labels) * 100

    top1_acc = calc_topk_acc(all_labels, all_top1_preds)
    top3_acc = calc_topk_acc(all_labels, all_top3_preds)
    top5_acc = calc_topk_acc(all_labels, all_top5_preds)
    top10_acc = calc_topk_acc(all_labels, all_top10_preds)
    top20_acc = calc_topk_acc(all_labels, all_top20_preds)

    # Calculate WER and CER using top-1 predictions
    true_words = [label_to_word[l] for l in all_labels]
    pred_words = [label_to_word[p[0]] for p in all_top1_preds]
    
    wer_metric = nnet.WordErrorRate()
    cer_metric = nnet.CharacterErrorRate()
    
    wer = wer_metric(true_words, pred_words).item()
    cer = cer_metric(true_words, pred_words).item()

    # Print results
    print("\n" + "=" * 80)
    print(f"  RESULTS FOR EPOCH {epoch_num}")
    print("=" * 80)
    print(f"\nCheckpoint: {checkpoint_path}")
    print(f"Test Samples: {len(all_labels):,}")
    
    print(f"\n{'─' * 40}")
    print("TOP-K ACCURACY")
    print(f"{'─' * 40}")
    print(f"  Top-1:  {top1_acc:6.2f}%")
    print(f"  Top-3:  {top3_acc:6.2f}%  (+{top3_acc - top1_acc:.2f}% from Top-1)")
    print(f"  Top-5:  {top5_acc:6.2f}%  (+{top5_acc - top1_acc:.2f}% from Top-1)")
    print(f"  Top-10: {top10_acc:6.2f}%  (+{top10_acc - top1_acc:.2f}% from Top-1)")
    print(f"  Top-20: {top20_acc:6.2f}%  (+{top20_acc - top1_acc:.2f}% from Top-1)")
    
    print(f"\n{'─' * 40}")
    print("ERROR RATES")
    print(f"{'─' * 40}")
    print(f"  WER (Word Error Rate):      {wer:6.2f}%")
    print(f"  CER (Character Error Rate): {cer:6.2f}%")
    
    print(f"\n{'─' * 40}")
    print("COMPARISON WITH EPOCH 177")
    print(f"{'─' * 40}")
    # Epoch 177 baseline values from the user's file
    ep177_top1 = 40.76
    ep177_top3 = 58.25
    ep177_top5 = 64.32
    ep177_top10 = 70.80
    ep177_top20 = 75.22
    
    print(f"  Metric  | Epoch 177 | Epoch {epoch_num:>3} |   Change")
    print(f"  --------|-----------|-----------|----------")
    print(f"  Top-1   |   {ep177_top1:5.2f}%  |   {top1_acc:5.2f}%  | {top1_acc - ep177_top1:+6.2f}%")
    print(f"  Top-3   |   {ep177_top3:5.2f}%  |   {top3_acc:5.2f}%  | {top3_acc - ep177_top3:+6.2f}%")
    print(f"  Top-5   |   {ep177_top5:5.2f}%  |   {top5_acc:5.2f}%  | {top5_acc - ep177_top5:+6.2f}%")
    print(f"  Top-10  |   {ep177_top10:5.2f}%  |   {top10_acc:5.2f}%  | {top10_acc - ep177_top10:+6.2f}%")
    print(f"  Top-20  |   {ep177_top20:5.2f}%  |   {top20_acc:5.2f}%  | {top20_acc - ep177_top20:+6.2f}%")
    
    print("\n" + "=" * 80)
    
    # Save results to file
    output_file = f"VisualCE_epoch{epoch_num}_topk_results.txt"
    with open(output_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write(f"  VISUALCE EPOCH {epoch_num} LIPBENGAL TOP-K ACCURACY RESULTS\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Checkpoint: {checkpoint_path}\n")
        f.write(f"Epoch: {epoch_num}\n")
        f.write(f"Test Samples: {len(all_labels):,}\n\n")
        f.write("Top-K Accuracy:\n")
        f.write(f"  Top-1:  {top1_acc:.2f}%\n")
        f.write(f"  Top-3:  {top3_acc:.2f}%  (+{top3_acc - top1_acc:.2f}% improvement)\n")
        f.write(f"  Top-5:  {top5_acc:.2f}%  (+{top5_acc - top1_acc:.2f}% improvement)\n")
        f.write(f"  Top-10: {top10_acc:.2f}%  (+{top10_acc - top1_acc:.2f}% improvement)\n")
        f.write(f"  Top-20: {top20_acc:.2f}%  (+{top20_acc - top1_acc:.2f}% improvement)\n\n")
        f.write("Error Rates:\n")
        f.write(f"  WER: {wer:.2f}%\n")
        f.write(f"  CER: {cer:.2f}%\n\n")
        f.write("=" * 80 + "\n")
        f.write("COMPARISON WITH EPOCH 177\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Model             | Top-1  | Top-3  | Top-5  | Top-10 | Top-20\n")
        f.write(f"-" * 70 + "\n")
        f.write(f"VisualCE (ep177)  | {ep177_top1:.2f}% | {ep177_top3:.2f}% | {ep177_top5:.2f}% | {ep177_top10:.2f}% | {ep177_top20:.2f}%\n")
        f.write(f"VisualCE (ep{epoch_num})  | {top1_acc:.2f}% | {top3_acc:.2f}% | {top5_acc:.2f}% | {top10_acc:.2f}% | {top20_acc:.2f}%\n")
        f.write(f"Change            | {top1_acc - ep177_top1:+.2f}% | {top3_acc - ep177_top3:+.2f}% | {top5_acc - ep177_top5:+.2f}% | {top10_acc - ep177_top10:+.2f}% | {top20_acc - ep177_top20:+.2f}%\n\n")
        f.write("=" * 80 + "\n")
    
    print(f"\nResults saved to: {output_file}")

if __name__ == "__main__":
    main()





