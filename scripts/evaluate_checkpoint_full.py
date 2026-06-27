#!/usr/bin/env python3
"""
Comprehensive checkpoint evaluation: Accuracy, WER, CER, and Top-K analysis
"""

import os
import sys
import argparse
import importlib
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project modules
import functions
import nnet.metrics as metrics


def evaluate_checkpoint(config_file, checkpoint_path, k_values=[1, 3, 5, 10, 20]):
    """
    Comprehensive evaluation of a checkpoint
    
    Args:
        config_file: Path to config file
        checkpoint_path: Path to checkpoint
        k_values: List of K values for top-k analysis
        
    Returns:
        Dictionary with all metrics
    """
    print("="*80)
    print("COMPREHENSIVE CHECKPOINT EVALUATION")
    print("="*80)
    print(f"Config: {config_file}")
    print(f"Checkpoint: {checkpoint_path}")
    print("="*80)
    
    # Create args namespace
    class Args:
        pass
    
    args = Args()
    args.mode = 'evaluation'
    args.config_file = config_file
    args.checkpoint = checkpoint_path
    args.load_last = False
    args.distributed = False
    args.parallel = False
    args.show_dict = False
    args.show_modules = False
    args.batch_size_eval = None
    args.num_workers = 4
    args.world_size = 1
    args.dist_log = False
    args.cpu = False
    args.rank = 0
    
    # Load config
    print("\nLoading configuration...")
    spec = importlib.util.spec_from_file_location("config", config_file)
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    args.config = config
    
    # Load model
    print("Loading model...")
    model = functions.load_model(args)
    model.load(checkpoint_path, load_optimizer=False)
    model.eval()
    
    print(f"Model name: {model.name}")
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number Parameters: {num_params:,}")
    
    # Get evaluation dataset
    dataset = config.evaluation_dataset
    dataset_name = getattr(dataset, 'name', 'Unknown')
    num_samples = len(dataset.dataset) if hasattr(dataset, 'dataset') else len(dataset)
    print(f"\nEvaluation Dataset: {dataset_name}, {num_samples} samples")
    print(f"Batch size: {dataset.batch_size}, Number of batches: {len(dataset)}")
    
    # Get label-to-word mapping
    label_to_word = {i: w for i, w in enumerate(dataset.classes)}
    
    # Initialize metrics
    wer_metric = metrics.WordErrorRate()
    cer_metric = metrics.CharacterErrorRate()
    
    # Storage for results
    total_samples = 0
    correct_at_k = {k: 0 for k in k_values}
    all_true_words = []
    all_pred_words = []
    
    print(f"\n{'='*80}")
    print(f"Running evaluation with Top-K analysis (k={k_values})...")
    print(f"{'='*80}\n")
    
    # Evaluate
    with torch.no_grad():
        for batch in tqdm(dataset, desc="Evaluating"):
            # Unpack batch - handle both dict and tuple formats
            if isinstance(batch, dict):
                inputs = batch["inputs"]
                targets = batch["targets"]
            elif isinstance(batch, (tuple, list)):
                inputs = batch[0]
                targets = batch[1]
            else:
                raise ValueError(f"Unexpected batch type: {type(batch)}")
            
            # Move to device
            if isinstance(inputs, dict):
                inputs = {key: val.to(model.device) if torch.is_tensor(val) else val 
                         for key, val in inputs.items()}
            elif torch.is_tensor(inputs):
                inputs = inputs.to(model.device)
            
            # Forward pass
            outputs = model(inputs)
            
            # Get logits (handle different output formats)
            if isinstance(outputs, dict):
                logits = outputs['output'] if 'output' in outputs else list(outputs.values())[0]
            elif isinstance(outputs, (list, tuple)):
                logits = outputs[0]
            else:
                logits = outputs
            
            # Get true labels
            if isinstance(targets, dict):
                labels = targets['output'] if 'output' in targets else list(targets.values())[0]
            elif isinstance(targets, (list, tuple)):
                labels = targets[0]
            else:
                labels = targets
            
            # Get top-k predictions
            batch_size = labels.size(0)
            max_k = max(k_values)
            
            # Get top-k predictions
            topk = torch.topk(logits, k=max_k, dim=1)
            topk_indices = topk.indices.cpu()
            
            # Process each sample in batch
            for i in range(batch_size):
                label = int(labels[i])
                preds = topk_indices[i].tolist()
                
                # Check correctness at different K values
                for k in k_values:
                    if label in preds[:k]:
                        correct_at_k[k] += 1
                
                # Store predictions for WER/CER
                true_word = label_to_word[label]
                pred_word = label_to_word[preds[0]]  # Top-1 prediction
                
                all_true_words.append(true_word)
                all_pred_words.append(pred_word)
                
                total_samples += 1
    
    # Calculate metrics
    print(f"\n{'='*80}")
    print("EVALUATION RESULTS")
    print(f"{'='*80}")
    print(f"Total Samples Evaluated: {total_samples:,}")
    print()
    
    # Top-K Accuracies
    print("Top-K Accuracy:")
    topk_results = {}
    for k in k_values:
        acc = (correct_at_k[k] / total_samples) * 100
        topk_results[f"top{k}"] = acc
        improvement = acc - topk_results["top1"] if k > 1 else 0
        improvement_str = f"  (+{improvement:.2f}%)" if k > 1 else ""
        print(f"  Top-{k:2d}: {acc:6.2f}%{improvement_str}")
    
    print()
    
    # WER and CER
    wer = wer_metric(all_true_words, all_pred_words).item()
    cer = cer_metric(all_true_words, all_pred_words).item()
    
    print(f"Word Error Rate (WER): {wer:.2f}%")
    print(f"Char Error Rate (CER): {cer:.2f}%")
    
    print(f"{'='*80}")
    
    # Return all metrics
    results = {
        'total_samples': total_samples,
        'topk_accuracy': topk_results,
        'wer': wer,
        'cer': cer,
        'checkpoint': checkpoint_path,
        'config': config_file
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Comprehensive checkpoint evaluation')
    parser.add_argument('-c', '--config', required=True, help='Path to config file')
    parser.add_argument('-i', '--checkpoint', required=True, help='Path to checkpoint file')
    parser.add_argument('--k_values', nargs='+', type=int, default=[1, 3, 5, 10, 20],
                        help='K values for top-k analysis')
    parser.add_argument('--output', help='Optional: Save results to JSON file')
    
    args = parser.parse_args()
    
    results = evaluate_checkpoint(args.config, args.checkpoint, args.k_values)
    
    # Optionally save to JSON
    if args.output:
        import json
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Results saved to: {args.output}")
    
    return results


if __name__ == '__main__':
    main()

