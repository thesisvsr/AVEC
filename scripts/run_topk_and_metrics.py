#!/usr/bin/env python3
"""
Top-K Analysis Script with WER and CER Metrics
Evaluates models with top-k accuracy, WER, and CER.
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

class TopKMetricsAnalyzer:
    """Analyzes model predictions with top-k values, WER, and CER"""
    
    def __init__(self, model, dataset_loader, k_values=[1, 3, 5, 10]):
        self.model = model
        self.dataset_loader = dataset_loader
        self.k_values = sorted(k_values)
        self.results = {k: [] for k in k_values}
        self.predictions = []
        
        # Get class-to-word mapping
        if hasattr(dataset_loader.dataset, 'classes'):
            self.classes = dataset_loader.dataset.classes
        else:
            print("Warning: Dataset classes not found. WER/CER will use indices.")
            self.classes = None
            
        self.wer_metric = metrics.WordErrorRate()
        self.cer_metric = metrics.CharacterErrorRate()
        
    def evaluate(self):
        """Evaluate model with different top-k metrics and WER/CER"""
        self.model.eval()
        
        print(f"\nRunning Comprehensive Analysis (Top-K={self.k_values}, WER, CER)...")
        print("="*70)
        
        total_samples = 0
        correct_at_k = {k: 0 for k in self.k_values}
        
        all_true_words = []
        all_pred_words = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(self.dataset_loader, desc="Evaluating")):
                # Unpack batch
                inputs = batch["inputs"]
                targets = batch["targets"]
                
                # Move to device
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
                
                # Get logits
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
                
                batch_size = logits.shape[0]
                total_samples += batch_size
                
                # Get probabilities
                probs = torch.softmax(logits, dim=-1)
                
                # Top-1 predictions for WER/CER
                _, top1_indices = torch.topk(probs, k=1, dim=-1)
                top1_indices = top1_indices.squeeze(-1)
                
                # Collect words for WER/CER
                if self.classes:
                    for i in range(batch_size):
                        true_idx = true_labels[i].item()
                        pred_idx = top1_indices[i].item()
                        
                        if true_idx < len(self.classes):
                            all_true_words.append(self.classes[true_idx])
                        else:
                            all_true_words.append(str(true_idx))
                            
                        if pred_idx < len(self.classes):
                            all_pred_words.append(self.classes[pred_idx])
                        else:
                            all_pred_words.append(str(pred_idx))
                
                # Calculate top-k accuracy
                for k in self.k_values:
                    # Get top-k predictions
                    topk_probs, topk_indices = torch.topk(probs, k=min(k, probs.shape[-1]), dim=-1)
                    
                    # Check if true label is in top-k
                    true_in_topk = (topk_indices == true_labels.unsqueeze(-1)).any(dim=-1)
                    correct_at_k[k] += true_in_topk.sum().item()
                
                # Store predictions (first batch only)
                if batch_idx == 0 and self.classes:
                    for i in range(min(batch_size, 10)):
                        top10_probs, top10_indices = torch.topk(probs[i], k=min(10, probs.shape[-1]))
                        self.predictions.append({
                            'true_label': self.classes[true_labels[i].item()],
                            'top10_words': [self.classes[idx.item()] for idx in top10_indices],
                            'top10_probs': top10_probs.cpu().tolist()
                        })
        
        # Calculate final metrics
        results = {}
        
        # Accuracies
        for k in self.k_values:
            accuracy = 100.0 * correct_at_k[k] / total_samples
            results[f'top{k}_acc'] = accuracy
            
        # WER and CER
        if all_true_words and all_pred_words:
            wer = self.wer_metric(all_true_words, all_pred_words).item()
            cer = self.cer_metric(all_true_words, all_pred_words).item()
            results['wer'] = wer
            results['cer'] = cer
        else:
            results['wer'] = 0.0
            results['cer'] = 0.0
            
        self.final_results = results
        return results
    
    def print_results(self):
        """Print analysis results"""
        print("\n" + "="*70)
        print("COMPREHENSIVE EVALUATION RESULTS")
        print("="*70)
        
        # Print Accuracies
        for k in self.k_values:
            acc = self.final_results[f'top{k}_acc']
            improvement = acc - self.final_results['top1_acc'] if k > 1 else 0.0
            print(f"Top-{k:2d} Accuracy: {acc:6.2f}%  (+{improvement:5.2f}% vs Top-1)")
            
        print("-" * 70)
        print(f"WER:             {self.final_results['wer']:6.2f}%")
        print(f"CER:             {self.final_results['cer']:6.2f}%")
        print("="*70)
        
        # Show example predictions
        if self.predictions:
            print("\nExample Predictions (first 5 samples):")
            print("-"*70)
            for i, pred in enumerate(self.predictions[:5]):
                print(f"\nSample {i+1}:")
                print(f"  True: {pred['true_label']}")
                print(f"  Pred: {pred['top10_words'][0]} (prob={pred['top10_probs'][0]:.4f})")
                print(f"  Top-5 Candidates:")
                for rank, (word, prob) in enumerate(zip(pred['top10_words'][:5], 
                                                       pred['top10_probs'][:5]), 1):
                    marker = "✓" if word == pred['true_label'] else " "
                    print(f"    {marker} {rank}. {word:<20} ({prob:.4f})")
    
    def save_results(self, output_path):
        """Save results to JSON file"""
        results_dict = {
            'metrics': self.final_results,
            'k_values': self.k_values,
            'example_predictions': self.predictions
        }
        
        with open(output_path, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        print(f"\n✓ Results saved to: {output_path}")
    
    def plot_results(self, output_path, dataset_name=""):
        """Plot top-k accuracy curve"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        k_vals = self.k_values
        accs = [self.final_results[f'top{k}_acc'] for k in k_vals]
        
        # Plot line
        ax.plot(k_vals, accs, 'o-', linewidth=2, markersize=8, color='steelblue')
        
        # Add value labels
        for k, acc in zip(k_vals, accs):
            ax.text(k, acc + 1, f'{acc:.1f}%', ha='center', va='bottom', 
                   fontsize=10, fontweight='bold')
        
        # Add metrics box
        metrics_text = (
            f"Top-1 Acc: {self.final_results['top1_acc']:.2f}%\n"
            f"WER:       {self.final_results['wer']:.2f}%\n"
            f"CER:       {self.final_results['cer']:.2f}%"
        )
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.3)
        ax.text(0.95, 0.05, metrics_text, transform=ax.transAxes, fontsize=11,
                verticalalignment='bottom', horizontalalignment='right', bbox=props)
        
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
    parser = argparse.ArgumentParser(description='Comprehensive Evaluation for Lip Reading')
    parser.add_argument('-c', '--config_file', type=str, required=True,
                       help='Configuration file path')
    parser.add_argument('-i', '--checkpoint', type=str, default=None,
                       help='Checkpoint file name (default: best checkpoint)')
    parser.add_argument('--k_values', type=int, nargs='+', default=[1, 2, 3, 4, 5, 10],
                       help='K values to evaluate')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Load configuration
    print(f"\nLoading configuration from: {args.config_file}")
    try:
        args.config = importlib.import_module(args.config_file.replace(".py", "").replace("/", "."))
    except ImportError as e:
        print(f"Error loading config: {e}")
        sys.exit(1)
    
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
    args.cpu = False
    args.load_last = False
    
    # Load model
    print("\nLoading model...")
    model = functions.load_model(args)
    
    # Load checkpoint
    if args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        # If full path provided, use it directly, else join with callback path
        if os.path.exists(args.checkpoint):
            ckpt_path = args.checkpoint
        else:
            ckpt_path = os.path.join(args.config.callback_path, args.checkpoint)
            
        model.load(ckpt_path, load_optimizer=False)
    
    # Load dataset
    print("\nLoading evaluation dataset...")
    _, dataset_eval = functions.load_datasets(args)
    
    if dataset_eval is None:
        print("Error: No evaluation dataset found!")
        return
    
    # Handle multiple eval datasets
    if isinstance(dataset_eval, list):
        dataset_eval = dataset_eval[0]
        print(f"Using first evaluation dataset: {dataset_eval.dataset.__class__.__name__}")
    
    # Setup output directory
    if args.output_dir is None:
        args.output_dir = Path("topk_analysis_results") / "LipBengal_Latest"
    else:
        args.output_dir = Path(args.output_dir)
    
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run Analysis
    analyzer = TopKMetricsAnalyzer(model, dataset_eval, k_values=args.k_values)
    analyzer.evaluate()
    analyzer.print_results()
    
    # Save results
    output_json = args.output_dir / "results.json"
    analyzer.save_results(output_json)
    
    # Plot results
    output_plot = args.output_dir / "topk_plot.png"
    analyzer.plot_results(output_plot, dataset_name="LipBengal VisualCE (Epoch 232)")
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print(f"Results saved to: {args.output_dir}")
    print("="*70)

if __name__ == "__main__":
    main()

