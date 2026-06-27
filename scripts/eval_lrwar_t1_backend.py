#!/usr/bin/env python3
"""
Comprehensive Evaluation for LRW-AR T1_backend (Transfer Backend Ablation)
Metrics: Top-1, Top-3, Top-5, Top-10, WER, CER, Parameters
"""

import torch
import importlib
import functions
import nnet
import torch.nn.functional as F
import sys
import os
from pathlib import Path

def count_parameters(model):
    """Count total and trainable parameters"""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def main():
    # Checkpoint path - use latest available (epoch 99)
    checkpoint_path = "callbacks/LRW-AR/AV/ablations/T1_backend/checkpoints_epoch_99_step_49379.ckpt"
    config_path = 'configs.LRW-AR.AV.ablations.T1_backend'
    
    print("=" * 80)
    print("  LRW-AR T1_BACKEND (TRANSFER BACKEND) EVALUATION")
    print("=" * 80)
    print(f"\nCheckpoint: {checkpoint_path}")
    print(f"Config: {config_path}")
    
    # Load config
    config = importlib.import_module(config_path)

    # Collate
    collate_fn = nnet.CollateFn(inputs_params=[{"axis": 0}], targets_params=[{"axis": 2}])

    # Use validation dataset (standard for LRW-AR evaluation)
    print("\nLoading LRW-AR validation dataset...")
    dataset = nnet.datasets.LRWAR(
        batch_size=64,
        collate_fn=collate_fn,
        mode='val',
        video_transform=config.val_video_transform,
        center_crop_size=(96, 96),
        fixed_frames=29,
        load_audio=False,
        indices_path='datasets/LRW-AR/indices/val.pt',
        prepared_only=True,
        use_arabish=True,
    )
    print(f"Validation samples: {len(dataset):,}")
    print(f"Number of classes: {dataset.num_classes}")

    loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=dataset.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=dataset.collate_fn,
        pin_memory=True
    )

    # Args for model loading
    class Args: pass
    args = Args()
    args.rank = 0
    args.cpu = False
    args.distributed = False
    args.parallel = False
    args.show_dict = False
    args.show_modules = False
    args.config_file = 'configs/LRW-AR/AV/ablations/T1_backend.py'
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

    # Count parameters
    total_params, trainable_params = count_parameters(model)
    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")

    label_to_word = {i: w for i, w in enumerate(dataset.classes)}

    # Storage for results
    all_labels = []
    all_top1_preds = []
    all_top3_preds = []
    all_top5_preds = []
    all_top10_preds = []
    
    print("\nRunning evaluation...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            vids = batch['inputs'].to(model.device)
            labels = batch['targets']
            
            out = model.forward(vids)
            logp = F.log_softmax(out, dim=1)
            
            # Top-10
            topk = torch.topk(logp, k=10, dim=1)
            indices = topk.indices.cpu()
            
            for i in range(labels.size(0)):
                label = int(labels[i])
                preds = indices[i].tolist()
                
                all_labels.append(label)
                all_top1_preds.append(preds[:1])
                all_top3_preds.append(preds[:3])
                all_top5_preds.append(preds[:5])
                all_top10_preds.append(preds[:10])
                
            if (batch_idx + 1) % 20 == 0:
                print(f"  Processed {len(all_labels):,} / {len(dataset):,} samples...")

    # Calculate Top-K accuracies
    def calc_topk_acc(labels, topk_preds):
        correct = sum(1 for l, preds in zip(labels, topk_preds) if l in preds)
        return correct / len(labels) * 100

    top1_acc = calc_topk_acc(all_labels, all_top1_preds)
    top3_acc = calc_topk_acc(all_labels, all_top3_preds)
    top5_acc = calc_topk_acc(all_labels, all_top5_preds)
    top10_acc = calc_topk_acc(all_labels, all_top10_preds)

    # Calculate WER and CER using top-1 predictions
    true_words = [label_to_word[l] for l in all_labels]
    pred_words = [label_to_word[p[0]] for p in all_top1_preds]
    
    wer_metric = nnet.WordErrorRate()
    cer_metric = nnet.CharacterErrorRate()
    
    wer = wer_metric(true_words, pred_words).item()
    cer = cer_metric(true_words, pred_words).item()

    # Print results
    print("\n" + "=" * 80)
    print("  RESULTS: LRW-AR T1_BACKEND (TRANSFER BACKEND ABLATION)")
    print("=" * 80)
    
    print(f"\n{'─' * 40}")
    print("MODEL INFORMATION")
    print(f"{'─' * 40}")
    print(f"  Checkpoint:           {checkpoint_path}")
    print(f"  Total Parameters:     {total_params:,}")
    print(f"  Trainable Parameters: {trainable_params:,}")
    print(f"  Validation Samples:   {len(all_labels):,}")
    print(f"  Number of Classes:    {dataset.num_classes}")
    
    print(f"\n{'─' * 40}")
    print("TOP-K ACCURACY")
    print(f"{'─' * 40}")
    print(f"  Top-1:   {top1_acc:6.2f}%")
    print(f"  Top-3:   {top3_acc:6.2f}%  (+{top3_acc - top1_acc:.2f}% from Top-1)")
    print(f"  Top-5:   {top5_acc:6.2f}%  (+{top5_acc - top1_acc:.2f}% from Top-1)")
    print(f"  Top-10:  {top10_acc:6.2f}%  (+{top10_acc - top1_acc:.2f}% from Top-1)")
    
    print(f"\n{'─' * 40}")
    print("ERROR RATES")
    print(f"{'─' * 40}")
    print(f"  WER (Word Error Rate):      {wer:6.2f}%")
    print(f"  CER (Character Error Rate): {cer:6.2f}%")
    
    print("\n" + "=" * 80)
    
    # Save results to file
    output_file = "LRWAR_T1_backend_evaluation_results.txt"
    with open(output_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("  LRW-AR T1_BACKEND (TRANSFER BACKEND ABLATION) EVALUATION RESULTS\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("MODEL INFORMATION\n")
        f.write("-" * 40 + "\n")
        f.write(f"Checkpoint: {checkpoint_path}\n")
        f.write(f"Config: {config_path}\n")
        f.write(f"Total Parameters: {total_params:,}\n")
        f.write(f"Trainable Parameters: {trainable_params:,}\n")
        f.write(f"Validation Samples: {len(all_labels):,}\n")
        f.write(f"Number of Classes: {dataset.num_classes}\n\n")
        
        f.write("TOP-K ACCURACY\n")
        f.write("-" * 40 + "\n")
        f.write(f"Top-1:  {top1_acc:.2f}%\n")
        f.write(f"Top-3:  {top3_acc:.2f}%  (+{top3_acc - top1_acc:.2f}% improvement)\n")
        f.write(f"Top-5:  {top5_acc:.2f}%  (+{top5_acc - top1_acc:.2f}% improvement)\n")
        f.write(f"Top-10: {top10_acc:.2f}%  (+{top10_acc - top1_acc:.2f}% improvement)\n\n")
        
        f.write("ERROR RATES\n")
        f.write("-" * 40 + "\n")
        f.write(f"WER (Word Error Rate):      {wer:.2f}%\n")
        f.write(f"CER (Character Error Rate): {cer:.2f}%\n\n")
        
        f.write("=" * 80 + "\n")
        f.write("ABLATION DETAILS\n")
        f.write("=" * 80 + "\n")
        f.write("- Transfer Mode: backend (pretrained Conformer backend only)\n")
        f.write("- Pretrained: Conformer layers from LRW English model\n")
        f.write("- Frontend: Randomly initialized (3D CNN + ResNet)\n")
        f.write("- Freeze Epochs: 0\n")
        f.write("- Encoder LR Mult: 0.2\n")
        f.write("- Head LR Mult: 1.0\n")
        f.write("- Training Epochs: 100\n")
        f.write("=" * 80 + "\n")
    
    print(f"\nResults saved to: {output_file}")

if __name__ == "__main__":
    main()





