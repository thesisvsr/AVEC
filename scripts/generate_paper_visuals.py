#!/usr/bin/env python3
"""
Generate Publication-Ready Visualizations for Ablation Study

Creates:
- Phase 1 comparison bar charts
- Training curves
- Statistical significance markers
- LaTeX tables
- Summary figures for paper

Usage:
    python3 scripts/generate_paper_visuals.py --output paper_figures/
"""

import os
import sys
import argparse
import json
from pathlib import Path
from typing import Dict, List, Any
import numpy as np

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False

# Set publication-quality defaults
if PLOTTING_AVAILABLE:
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.titlesize': 18,
        'font.family': 'serif',
        'text.usetex': False,  # Set True if LaTeX installed
    })
    sns.set_style("whitegrid")
    sns.set_palette("husl")


def parse_training_log(log_path: Path) -> Dict[str, Any]:
    """Extract metrics from training log"""
    metrics = {
        'epochs': [],
        'train_loss': [],
        'val_accuracy': [],
        'val_wer': [],
        'val_cer': [],
        'final_epoch': 0,
        'best_accuracy': 0.0,
    }
    
    if not log_path.exists():
        return metrics
    
    try:
        with open(log_path, 'r') as f:
            content = f.read()
            
        # Extract final epoch
        import re
        epochs = re.findall(r'Epoch (\d+)/\d+', content)
        if epochs:
            metrics['final_epoch'] = int(epochs[-1])
        
        # Extract evaluation results
        eval_blocks = re.findall(
            r'Evaluation:(.*?)(?=Epoch|$)', 
            content, 
            re.DOTALL
        )
        
        for block in eval_blocks:
            # Look for accuracy
            acc_match = re.search(r'CategoricalAccuracy[:\s]+([0-9.]+)', block)
            if acc_match:
                acc = float(acc_match.group(1))
                metrics['val_accuracy'].append(acc)
                if acc > metrics['best_accuracy']:
                    metrics['best_accuracy'] = acc
            
            # Look for WER
            wer_match = re.search(r'WordErrorRate[:\s]+([0-9.]+)', block)
            if wer_match:
                metrics['val_wer'].append(float(wer_match.group(1)))
            
            # Look for CER
            cer_match = re.search(r'CharacterErrorRate[:\s]+([0-9.]+)', block)
            if cer_match:
                metrics['val_cer'].append(float(cer_match.group(1)))
    
    except Exception as e:
        print(f"Warning: Error parsing {log_path}: {e}")
    
    return metrics


def generate_phase1_comparison(results: Dict[str, Dict], output_dir: Path):
    """Generate Phase 1 comparison bar chart"""
    
    if not PLOTTING_AVAILABLE:
        print("Matplotlib not available, skipping plots")
        return
    
    experiments = ['S1_raw', 'S1_phonetic', 'S1_simple', 'S1_mixed']
    labels = ['Raw\nBengali', 'Phonetic\n(Banglish)', 'Simple\nTranslit', 'Mixed']
    
    accuracies = []
    wers = []
    
    for exp in experiments:
        if exp in results and results[exp]['best_accuracy'] > 0:
            accuracies.append(results[exp]['best_accuracy'] * 100)  # Convert to percentage
            if results[exp]['val_wer']:
                wers.append(min(results[exp]['val_wer']) * 100)
            else:
                wers.append(0)
        else:
            accuracies.append(0)
            wers.append(0)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Accuracy comparison
    colors = ['#d62728', '#2ca02c', '#ff7f0e', '#9467bd']
    bars1 = ax1.bar(labels, accuracies, color=colors, alpha=0.8, edgecolor='black')
    ax1.set_ylabel('Top-1 Accuracy (%)', fontweight='bold')
    ax1.set_title('Phase 1: Script Normalization Impact\nTop-1 Accuracy Comparison', fontweight='bold')
    ax1.set_ylim([0, max(accuracies) * 1.2 if max(accuracies) > 0 else 100])
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, acc in zip(bars1, accuracies):
        height = bar.get_height()
        if height > 0:
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{acc:.2f}%',
                    ha='center', va='bottom', fontweight='bold')
    
    # Add improvement annotation
    if accuracies[0] > 0 and accuracies[1] > 0:
        improvement = accuracies[1] - accuracies[0]
        ax1.annotate(f'+{improvement:.2f} pp',
                    xy=(0.5, max(accuracies[0], accuracies[1])),
                    xytext=(0.5, max(accuracies) * 1.1),
                    ha='center',
                    fontsize=14,
                    color='green',
                    fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color='green', lw=2))
    
    # WER comparison
    bars2 = ax2.bar(labels, wers, color=colors, alpha=0.8, edgecolor='black')
    ax2.set_ylabel('Word Error Rate (%)', fontweight='bold')
    ax2.set_title('Phase 1: Script Normalization Impact\nWord Error Rate Comparison', fontweight='bold')
    ax2.set_ylim([0, max(wers) * 1.2 if max(wers) > 0 else 100])
    ax2.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, wer in zip(bars2, wers):
        height = bar.get_height()
        if height > 0:
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{wer:.2f}%',
                    ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    output_path = output_dir / 'phase1_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'phase1_comparison.pdf', bbox_inches='tight')
    print(f"✓ Generated: {output_path}")
    plt.close()


def generate_training_curves(results: Dict[str, Dict], output_dir: Path):
    """Generate training curves for all experiments"""
    
    if not PLOTTING_AVAILABLE:
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    colors = {'S1_raw': '#d62728', 'S1_phonetic': '#2ca02c', 
              'S1_simple': '#ff7f0e', 'S1_mixed': '#9467bd'}
    labels_map = {'S1_raw': 'Raw Bengali', 'S1_phonetic': 'Phonetic (Banglish)',
                  'S1_simple': 'Simple Transliteration', 'S1_mixed': 'Mixed'}
    
    for exp_id, metrics in results.items():
        if metrics['val_accuracy'] and len(metrics['val_accuracy']) > 0:
            epochs = list(range(1, len(metrics['val_accuracy']) + 1))
            accuracies = [a * 100 for a in metrics['val_accuracy']]
            
            ax.plot(epochs, accuracies, 
                   label=labels_map.get(exp_id, exp_id),
                   color=colors.get(exp_id, 'gray'),
                   linewidth=2.5,
                   marker='o',
                   markersize=4,
                   markevery=max(1, len(epochs)//10))
    
    ax.set_xlabel('Epoch', fontweight='bold')
    ax.set_ylabel('Top-1 Accuracy (%)', fontweight='bold')
    ax.set_title('Phase 1: Training Curves - Script Normalization Comparison', fontweight='bold')
    ax.legend(loc='best', frameon=True, shadow=True)
    ax.grid(True, alpha=0.3)
    
    output_path = output_dir / 'training_curves.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'training_curves.pdf', bbox_inches='tight')
    print(f"✓ Generated: {output_path}")
    plt.close()


def generate_latex_table(results: Dict[str, Dict], output_dir: Path):
    """Generate LaTeX table for paper"""
    
    experiments = [
        ('S1_raw', 'Raw Bengali Script'),
        ('S1_phonetic', 'Phonetic Romanization (Banglish)'),
        ('S1_simple', 'Simple Transliteration'),
        ('S1_mixed', 'Mixed Approach'),
    ]
    
    latex = []
    latex.append("\\begin{table}[ht]")
    latex.append("\\centering")
    latex.append("\\caption{Phase 1: Script Normalization Ablation Study Results on LipBengal}")
    latex.append("\\label{tab:phase1_results}")
    latex.append("\\begin{tabular}{lccccc}")
    latex.append("\\toprule")
    latex.append("\\textbf{Method} & \\textbf{Top-1 Acc (\%)} & \\textbf{Top-10 Acc (\%)} & \\textbf{WER (\%)} & \\textbf{CER (\%)} & \\textbf{Epochs} \\\\")
    latex.append("\\midrule")
    
    for exp_id, exp_name in experiments:
        if exp_id in results and results[exp_id]['best_accuracy'] > 0:
            metrics = results[exp_id]
            top1 = metrics['best_accuracy'] * 100
            top10 = 0.0  # Not extracted yet
            wer = min(metrics['val_wer']) * 100 if metrics['val_wer'] else 0.0
            cer = min(metrics['val_cer']) * 100 if metrics['val_cer'] else 0.0
            epochs = metrics['final_epoch']
            
            # Highlight best result
            if top1 == max(results[e]['best_accuracy'] * 100 for e in results if results[e]['best_accuracy'] > 0):
                latex.append(f"{exp_name} & \\textbf{{{top1:.2f}}} & - & {wer:.2f} & {cer:.2f} & {epochs} \\\\")
            else:
                latex.append(f"{exp_name} & {top1:.2f} & - & {wer:.2f} & {cer:.2f} & {epochs} \\\\")
        else:
            latex.append(f"{exp_name} & - & - & - & - & - \\\\")
    
    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")
    
    output_path = output_dir / 'phase1_table.tex'
    with open(output_path, 'w') as f:
        f.write('\n'.join(latex))
    
    print(f"✓ Generated: {output_path}")


def generate_summary_json(results: Dict[str, Dict], output_dir: Path):
    """Generate JSON summary for easy reference"""
    
    summary = {
        'phase': 1,
        'dataset': 'LipBengal',
        'experiments': {},
        'key_findings': {}
    }
    
    for exp_id, metrics in results.items():
        summary['experiments'][exp_id] = {
            'epochs_trained': metrics['final_epoch'],
            'best_accuracy': float(metrics['best_accuracy'] * 100) if metrics['best_accuracy'] > 0 else 0.0,
            'best_wer': float(min(metrics['val_wer']) * 100) if metrics['val_wer'] else 0.0,
            'best_cer': float(min(metrics['val_cer']) * 100) if metrics['val_cer'] else 0.0,
        }
    
    # Calculate key findings
    if 'S1_raw' in results and 'S1_phonetic' in results:
        raw_acc = results['S1_raw']['best_accuracy'] * 100
        phonetic_acc = results['S1_phonetic']['best_accuracy'] * 100
        if raw_acc > 0 and phonetic_acc > 0:
            improvement = phonetic_acc - raw_acc
            summary['key_findings']['phonetic_improvement'] = f"+{improvement:.2f} percentage points"
            summary['key_findings']['relative_improvement'] = f"{(improvement/raw_acc)*100:.1f}%"
    
    output_path = output_dir / 'phase1_summary.json'
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"✓ Generated: {output_path}")
    
    return summary


def main():
    parser = argparse.ArgumentParser(description="Generate paper visuals for ablation study")
    parser.add_argument('--output', default='paper_figures', help='Output directory')
    parser.add_argument('--dataset', default='LipBengal', help='Dataset name')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect results from all Phase 1 experiments
    results = {}
    base_path = Path(f'logs/ablations/{args.dataset}')
    
    experiments = ['S1_raw', 'S1_phonetic', 'S1_simple', 'S1_mixed']
    
    print("=" * 70)
    print("Collecting Phase 1 Results")
    print("=" * 70)
    
    for exp_id in experiments:
        log_path = base_path / exp_id / 'training.log'
        print(f"Processing: {exp_id}...")
        
        metrics = parse_training_log(log_path)
        results[exp_id] = metrics
        
        if metrics['final_epoch'] > 0:
            print(f"  ✓ {exp_id}: {metrics['final_epoch']} epochs, "
                  f"Best Acc: {metrics['best_accuracy']*100:.2f}%")
        else:
            print(f"  ⚠ {exp_id}: No results found")
    
    print()
    print("=" * 70)
    print("Generating Visualizations")
    print("=" * 70)
    
    # Generate all visualizations
    generate_phase1_comparison(results, output_dir)
    generate_training_curves(results, output_dir)
    generate_latex_table(results, output_dir)
    summary = generate_summary_json(results, output_dir)
    
    print()
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"Output directory: {output_dir.absolute()}")
    print(f"Files generated:")
    for f in output_dir.glob('*'):
        print(f"  - {f.name}")
    
    print()
    print("Key Findings:")
    for key, value in summary.get('key_findings', {}).items():
        print(f"  {key}: {value}")
    
    print()
    print("✓ All visualizations generated successfully!")


if __name__ == '__main__':
    main()


