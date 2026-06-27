#!/usr/bin/env python3
"""
Top-K Analysis Script
Evaluates models with top-k accuracy metrics to understand prediction confidence
and potential improvements beyond top-1 accuracy.
"""

import os
import sys
import argparse
import importlib
import torch
import json
from pathlib import Path
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import project modules
import functions
import nnet.metrics as metrics
import nnet.decoders as decoders


class TopKAnalyzer:
    """Analyzes model predictions with various top-k values"""
    
    def __init__(self, model, dataset, k_values=[1, 3, 5, 10]):
        self.model = model
        self.dataset = dataset
        self.k_values = sorted(k_values)
        self.results = {k: [] for k in k_values}
        self.predictions = []
        self.ground_truths = []
        
    def evaluate(self):
        """Evaluate model with different top-k metrics"""
        self.model.eval()
        
        print(f"\nRunning Top-K Analysis (k={self.k_values})...")
        print("="*70)
        
        total_samples = 0
        correct_at_k = {k: 0 for k in self.k_values}
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.dataset, desc="Evaluating")):
                # Unpack batch
                inputs = batch["inputs"]
                targets = batch["targets"]
                
                # Move to device - handle different input types
                if isinstance(inputs, dict):
                    inputs = {key: val.to(self.model.device) if torch.is_tensor(val) else val 
                             for key, val in inputs.items()}
                elif torch.is_tensor(inputs):
                    inputs = inputs.to(self.model.device)
                
                if isinstance(targets, dict):
                    targets = {key: val.to(self.model.device) if torch.is_tensor(val) else val 
                              for key, val in targets.items()}
                elif torch.is_tensor(targets):
                    targets = targets.to(self.model.device)
                
                # Forward pass
                outputs = self.model(inputs)
                
                # Get logits (handle different output formats)
                if isinstance(outputs, dict):
                    logits = outputs['output'] if 'output' in outputs else list(outputs.values())[0]
                elif isinstance(outputs, (list, tuple)):
                    logits = outputs[0]
                else:
                    logits = outputs
                
                # Get true labels
                if isinstance(targets, dict):
                    true_labels = targets['output'] if 'output' in targets else list(targets.values())[0]
                elif isinstance(targets, (list, tuple)):
                    true_labels = targets[0]
                else:
                    true_labels = targets
                
                # Calculate top-k accuracy for each k
                batch_size = logits.shape[0]
                total_samples += batch_size
                
                # Get probabilities
                probs = torch.softmax(logits, dim=-1)
                
                for k in self.k_values:
                    # Get top-k predictions
                    topk_probs, topk_indices = torch.topk(probs, k=min(k, probs.shape[-1]), dim=-1)
                    
                    # Check if true label is in top-k
                    true_in_topk = (topk_indices == true_labels.unsqueeze(-1)).any(dim=-1)
                    correct_at_k[k] += true_in_topk.sum().item()
                
                # Store predictions for detailed analysis (first batch only to save memory)
                if batch_idx == 0:
                    for i in range(min(batch_size, 10)):  # Store first 10 samples
                        top10_probs, top10_indices = torch.topk(probs[i], k=min(10, probs.shape[-1]))
                        self.predictions.append({
                            'true_label': true_labels[i].item(),
                            'top10_indices': top10_indices.cpu().tolist(),
                            'top10_probs': top10_probs.cpu().tolist()
                        })
        
        # Calculate accuracies
        for k in self.k_values:
            accuracy = 100.0 * correct_at_k[k] / total_samples
            self.results[k] = accuracy
        
        return self.results
    
    def print_results(self):
        """Print top-k analysis results"""
        print("\n" + "="*70)
        print("TOP-K ANALYSIS RESULTS")
        print("="*70)
        
        for k in self.k_values:
            acc = self.results[k]
            improvement = acc - self.results[1] if k > 1 else 0.0
            print(f"Top-{k:2d} Accuracy: {acc:6.2f}%  (+{improvement:5.2f}% vs Top-1)")
        
        print("="*70)
        
        # Show example predictions
        if self.predictions:
            print("\nExample Predictions (first 5 samples):")
            print("-"*70)
            for i, pred in enumerate(self.predictions[:5]):
                print(f"\nSample {i+1}:")
                print(f"  True Label: {pred['true_label']}")
                print(f"  Top-5 Predictions:")
                for rank, (idx, prob) in enumerate(zip(pred['top10_indices'][:5], 
                                                       pred['top10_probs'][:5]), 1):
                    marker = "✓" if idx == pred['true_label'] else " "
                    print(f"    {marker} Rank {rank}: Label {idx:3d} (prob={prob:.4f})")
    
    def save_results(self, output_path):
        """Save results to JSON file"""
        results_dict = {
            'k_values': self.k_values,
            'accuracies': {f'top{k}': self.results[k] for k in self.k_values},
            'improvements': {f'top{k}_improvement': self.results[k] - self.results[1] 
                           for k in self.k_values if k > 1},
            'example_predictions': self.predictions[:20]  # Save first 20
        }
        
        with open(output_path, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        print(f"\n✓ Results saved to: {output_path}")
    
    def plot_results(self, output_path, dataset_name=""):
        """Plot top-k accuracy curve"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        k_vals = self.k_values
        accs = [self.results[k] for k in k_vals]
        
        # Plot line
        ax.plot(k_vals, accs, 'o-', linewidth=2, markersize=8, color='steelblue')
        
        # Add value labels
        for k, acc in zip(k_vals, accs):
            ax.text(k, acc + 1, f'{acc:.1f}%', ha='center', va='bottom', 
                   fontsize=10, fontweight='bold')
        
        ax.set_xlabel('K (Top-K)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
        ax.set_title(f'Top-K Accuracy Analysis{" - " + dataset_name if dataset_name else ""}', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 100])
        ax.set_xticks(k_vals)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Plot saved to: {output_path}")
        plt.close()


def main():
    parser = argparse.ArgumentParser(description='Top-K Analysis for Lip Reading Models')
    parser.add_argument('-c', '--config_file', type=str, required=True,
                       help='Configuration file path')
    parser.add_argument('-i', '--checkpoint', type=str, default=None,
                       help='Checkpoint file name (default: best checkpoint)')
    parser.add_argument('--k_values', type=int, nargs='+', default=[1, 3, 5, 10],
                       help='K values to evaluate (default: 1 3 5 10)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for results (default: same as callback_path)')
    parser.add_argument('--cpu', action='store_true',
                       help='Use CPU instead of GPU')
    parser.add_argument('--load_last', action='store_true',
                       help='Load last checkpoint instead of best')
    
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
    
    # Setup output directory
    if args.output_dir is None:
        callback_path = Path(args.config.callback_path) if isinstance(args.config.callback_path, str) else args.config.callback_path
        args.output_dir = callback_path / "topk_analysis"
    else:
        args.output_dir = Path(args.output_dir)
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine dataset name for labeling
    dataset_name = ""
    callback_path = Path(args.config.callback_path) if isinstance(args.config.callback_path, str) else args.config.callback_path
    
    if "LRW-AR" in str(callback_path):
        dataset_name = "LRW-AR"
    elif "LipBengal" in str(callback_path):
        dataset_name = "LipBengal"
    
    experiment_name = callback_path.name
    
    # Run Top-K Analysis
    print("\n" + "="*70)
    print(f"TOP-K ANALYSIS: {experiment_name}")
    print("="*70)
    
    analyzer = TopKAnalyzer(model, dataset_eval, k_values=args.k_values)
    results = analyzer.evaluate()
    analyzer.print_results()
    
    # Save results
    output_json = args.output_dir / f"topk_results_{experiment_name}.json"
    analyzer.save_results(output_json)
    
    # Plot results
    output_plot = args.output_dir / f"topk_plot_{experiment_name}.png"
    analyzer.plot_results(output_plot, dataset_name=f"{dataset_name} - {experiment_name}")
    
    print("\n" + "="*70)
    print("TOP-K ANALYSIS COMPLETE!")
    print("="*70)


if __name__ == "__main__":
    main()

