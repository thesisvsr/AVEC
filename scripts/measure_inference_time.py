#!/usr/bin/env python3
"""
Inference Time Measurement Script
Measures average inference time per sample and per batch for each dataset.
"""

import os
import sys
import argparse
import importlib
import torch
import time
import json
from pathlib import Path
import numpy as np
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project modules
import functions


def measure_inference_time(model, dataset, num_batches=50, warmup_batches=5):
    """
    Measure inference time statistics
    
    Args:
        model: The model to evaluate
        dataset: The dataset to run inference on
        num_batches: Number of batches to measure (default: 50)
        warmup_batches: Number of warmup batches to skip (default: 5)
    
    Returns:
        dict: Dictionary containing timing statistics
    """
    model.eval()
    
    batch_times = []
    sample_times = []
    total_samples = 0
    
    print(f"\nWarming up with {warmup_batches} batches...")
    with torch.no_grad():
        # Warmup
        for i, batch in enumerate(dataset):
            if i >= warmup_batches:
                break
            
            inputs = batch["inputs"]
            
            # Move to device
            if isinstance(inputs, dict):
                inputs = {key: val.to(model.device) if torch.is_tensor(val) else val 
                         for key, val in inputs.items()}
            elif torch.is_tensor(inputs):
                inputs = inputs.to(model.device)
            
            # Warmup forward pass
            _ = model(inputs)
            
            # Synchronize GPU
            if torch.cuda.is_available():
                torch.cuda.synchronize()
    
    print(f"Measuring inference time over {num_batches} batches...")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataset, total=num_batches, desc="Measuring")):
            if batch_idx >= num_batches + warmup_batches:
                break
            if batch_idx < warmup_batches:
                continue
            
            inputs = batch["inputs"]
            
            # Get batch size
            if isinstance(inputs, dict):
                first_key = list(inputs.keys())[0]
                batch_size = inputs[first_key].shape[0] if torch.is_tensor(inputs[first_key]) else len(inputs[first_key])
            elif torch.is_tensor(inputs):
                batch_size = inputs.shape[0]
            else:
                batch_size = 1
            
            # Move to device
            if isinstance(inputs, dict):
                inputs = {key: val.to(model.device) if torch.is_tensor(val) else val 
                         for key, val in inputs.items()}
            elif torch.is_tensor(inputs):
                inputs = inputs.to(model.device)
            
            # Measure time
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            start_time = time.perf_counter()
            _ = model(inputs)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            end_time = time.perf_counter()
            
            batch_time = (end_time - start_time) * 1000  # Convert to ms
            sample_time = batch_time / batch_size
            
            batch_times.append(batch_time)
            sample_times.append(sample_time)
            total_samples += batch_size
    
    # Calculate statistics
    results = {
        'num_batches_measured': len(batch_times),
        'total_samples': total_samples,
        'batch_time_ms': {
            'mean': float(np.mean(batch_times)),
            'std': float(np.std(batch_times)),
            'min': float(np.min(batch_times)),
            'max': float(np.max(batch_times)),
            'median': float(np.median(batch_times)),
        },
        'sample_time_ms': {
            'mean': float(np.mean(sample_times)),
            'std': float(np.std(sample_times)),
            'min': float(np.min(sample_times)),
            'max': float(np.max(sample_times)),
            'median': float(np.median(sample_times)),
        },
        'throughput': {
            'samples_per_second': float(1000.0 / np.mean(sample_times)),
            'batches_per_second': float(1000.0 / np.mean(batch_times)),
        }
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Measure Inference Time for Lip Reading Models')
    parser.add_argument('-c', '--config_file', type=str, required=True,
                       help='Configuration file path')
    parser.add_argument('--num_batches', type=int, default=50,
                       help='Number of batches to measure (default: 50)')
    parser.add_argument('--warmup_batches', type=int, default=5,
                       help='Number of warmup batches (default: 5)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for results')
    parser.add_argument('--cpu', action='store_true',
                       help='Use CPU instead of GPU')
    
    args = parser.parse_args()
    
    # Load configuration
    print(f"\nLoading configuration from: {args.config_file}")
    args.config = importlib.import_module(args.config_file.replace(".py", "").replace("/", "."))
    
    # Setup
    args.mode = "evaluation"
    args.num_workers = 4
    args.distributed = False
    args.parallel = False
    args.rank = 0
    args.world_size = 1
    args.num_gpus = 1
    args.show_dict = False
    args.show_modules = False
    args.load_last = True
    args.checkpoint = None
    
    # Load model
    print("\nLoading model...")
    model = functions.load_model(args)
    
    # Load dataset
    print("\nLoading evaluation dataset...")
    _, dataset_eval = functions.load_datasets(args)
    
    if dataset_eval is None:
        print("Error: No evaluation dataset found in configuration!")
        return
    
    # Handle multiple eval datasets
    if isinstance(dataset_eval, list):
        dataset_eval = dataset_eval[0]
        print(f"Using first evaluation dataset: {dataset_eval.dataset.__class__.__name__}")
    
    # Get dataset info
    callback_path = Path(args.config.callback_path) if isinstance(args.config.callback_path, str) else args.config.callback_path
    dataset_name = ""
    if "LRW-AR" in str(callback_path):
        dataset_name = "LRW-AR"
    elif "LipBengal" in str(callback_path):
        dataset_name = "LipBengal"
    
    experiment_name = callback_path.name
    
    # Setup output directory
    if args.output_dir is None:
        args.output_dir = Path("inference_time_results")
    else:
        args.output_dir = Path(args.output_dir)
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Print info
    print("\n" + "="*70)
    print(f"INFERENCE TIME MEASUREMENT")
    print("="*70)
    print(f"Dataset:     {dataset_name}")
    print(f"Experiment:  {experiment_name}")
    print(f"Device:      {model.device}")
    print(f"Batches:     {args.num_batches}")
    print(f"Warmup:      {args.warmup_batches}")
    print("="*70)
    
    # Measure inference time
    results = measure_inference_time(
        model, 
        dataset_eval, 
        num_batches=args.num_batches,
        warmup_batches=args.warmup_batches
    )
    
    # Add metadata
    results['dataset_name'] = dataset_name
    results['experiment_name'] = experiment_name
    results['device'] = str(model.device)
    results['model_name'] = model.name if hasattr(model, 'name') else 'Unknown'
    
    # Print results
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"\nBatch Statistics:")
    print(f"  Mean:   {results['batch_time_ms']['mean']:.2f} ms")
    print(f"  Std:    {results['batch_time_ms']['std']:.2f} ms")
    print(f"  Median: {results['batch_time_ms']['median']:.2f} ms")
    print(f"  Min:    {results['batch_time_ms']['min']:.2f} ms")
    print(f"  Max:    {results['batch_time_ms']['max']:.2f} ms")
    
    print(f"\nPer-Sample Statistics:")
    print(f"  Mean:   {results['sample_time_ms']['mean']:.2f} ms")
    print(f"  Std:    {results['sample_time_ms']['std']:.2f} ms")
    print(f"  Median: {results['sample_time_ms']['median']:.2f} ms")
    print(f"  Min:    {results['sample_time_ms']['min']:.2f} ms")
    print(f"  Max:    {results['sample_time_ms']['max']:.2f} ms")
    
    print(f"\nThroughput:")
    print(f"  Samples/sec: {results['throughput']['samples_per_second']:.2f}")
    print(f"  Batches/sec: {results['throughput']['batches_per_second']:.2f}")
    
    print(f"\nTotal Samples Processed: {results['total_samples']}")
    print("="*70)
    
    # Save results
    output_file = args.output_dir / f"inference_time_{dataset_name}_{experiment_name}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_file}")


if __name__ == "__main__":
    main()

