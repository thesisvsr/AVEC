#!/usr/bin/env python3
"""
Compile Phase 1 Results from TensorBoard Logs and Generate Paper Visuals

Extracts metrics from TensorBoard event files and generates publication-ready figures.
"""

import os
import sys
from pathlib import Path
import json
import numpy as np

try:
    from tensorboard.backend.event_processing import event_accumulator
    TB_AVAILABLE = True
except ImportError:
    TB_AVAILABLE = False
    print("Warning: tensorboard not installed. Install with: pip install tensorboard")

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOT_AVAILABLE = True
    
    # Publication settings
    plt.rcParams.update({
        'font.size': 13,
        'axes.labelsize': 15,
        'axes.titlesize': 17,
        'xtick.labelsize': 13,
        'ytick.labelsize': 13,
        'legend.fontsize': 13,
        'figure.titlesize': 19,
        'font.family': 'sans-serif',
    })
    sns.set_style("whitegrid")
    sns.set_palette("husl")
except ImportError:
    PLOT_AVAILABLE = False


def extract_tensorboard_metrics(log_dir):
    """Extract metrics from TensorBoard event file"""
    if not TB_AVAILABLE:
        return {}
    
    try:
        ea = event_accumulator.EventAccumulator(log_dir)
        ea.Reload()
        
        metrics = {
            'train_loss': [],
            'val_accuracy': [],
            'val_wer': [],
            'val_cer': [],
            'epochs': []
        }
        
        # Extract training loss
        if 'Training-step/loss' in ea.Tags()['scalars']:
            for event in ea.Scalars('Training-step/loss'):
                metrics['train_loss'].append((event.step, event.value))
        
        # Extract validation metrics (use epoch-level metrics)
        if 'Evaluation-epoch/0/acc' in ea.Tags()['scalars']:
            for event in ea.Scalars('Evaluation-epoch/0/acc'):
                metrics['val_accuracy'].append((event.step, event.value))
        
        if 'Evaluation-epoch/0/wer' in ea.Tags()['scalars']:
            for event in ea.Scalars('Evaluation-epoch/0/wer'):
                metrics['val_wer'].append((event.step, event.value))
        
        if 'Evaluation-epoch/0/cer' in ea.Tags()['scalars']:
            for event in ea.Scalars('Evaluation-epoch/0/cer'):
                metrics['val_cer'].append((event.step, event.value))
        
        # Get best metrics
        if metrics['val_accuracy']:
            best_acc_idx = max(range(len(metrics['val_accuracy'])), 
                              key=lambda i: metrics['val_accuracy'][i][1])
            best_acc = metrics['val_accuracy'][best_acc_idx][1]
            best_epoch = metrics['val_accuracy'][best_acc_idx][0]
        else:
            best_acc = 0.0
            best_epoch = 0
        
        best_wer = min([v for _, v in metrics['val_wer']]) if metrics['val_wer'] else 0.0
        best_cer = min([v for _, v in metrics['val_cer']]) if metrics['val_cer'] else 0.0
        
        return {
            'best_accuracy': best_acc,  # Already in percentage
            'best_wer': best_wer,  # Already in percentage
            'best_cer': best_cer,  # Already in percentage
            'final_epoch': int(max([s for s, _ in metrics['val_accuracy']])) if metrics['val_accuracy'] else 0,
            'best_epoch': int(best_epoch),
            'train_loss_history': metrics['train_loss'],
            'val_accuracy_history': metrics['val_accuracy'],
            'val_wer_history': metrics['val_wer'],
            'val_cer_history': metrics['val_cer'],
        }
    
    except Exception as e:
        print(f"Error extracting metrics: {e}")
        return {}


def generate_phase1_comparison_plot(results, output_dir):
    """Generate publication-ready comparison plot"""
    if not PLOT_AVAILABLE:
        return
    
    experiments = ['S1_raw', 'S1_phonetic', 'S1_simple', 'S1_mixed']
    labels = ['Raw\nBengali', 'Phonetic\nRomanization\n(Banglish)', 
              'Simple\nTransliteration', 'Mixed\nApproach']
    colors = ['#d62728', '#2ca02c', '#ff7f0e', '#9467bd']
    
    # Collect data
    accuracies = []
    wers = []
    
    for exp in experiments:
        if exp in results and 'best_accuracy' in results[exp]:
            accuracies.append(results[exp]['best_accuracy'])
            wers.append(results[exp]['best_wer'])
        else:
            accuracies.append(0)
            wers.append(0)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Top-1 Accuracy
    bars1 = ax1.bar(labels, accuracies, color=colors, alpha=0.85, edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('Top-1 Accuracy (%)', fontweight='bold', fontsize=16)
    ax1.set_title('Script Normalization Impact on Accuracy', fontweight='bold', fontsize=18, pad=20)
    ax1.set_ylim([0, max(accuracies) * 1.25 if max(accuracies) > 0 else 100])
    ax1.grid(axis='y', alpha=0.4, linestyle='--')
    ax1.set_axisbelow(True)
    
    # Add value labels
    for bar, acc in zip(bars1, accuracies):
        height = bar.get_height()
        if height > 0:
            ax1.text(bar.get_x() + bar.get_width()/2., height + max(accuracies)*0.02,
                    f'{acc:.2f}%',
                    ha='center', va='bottom', fontweight='bold', fontsize=14)
    
    # Add improvement annotation if we have both raw and phonetic
    if accuracies[0] > 0 and accuracies[1] > 0:
        improvement = accuracies[1] - accuracies[0]
        y_pos = max(accuracies[0], accuracies[1]) + max(accuracies) * 0.15
        ax1.annotate(f'+{improvement:.2f} pp\nimprovement',
                    xy=(0.5, (accuracies[0] + accuracies[1])/2),
                    xytext=(0.5, y_pos),
                    ha='center',
                    fontsize=16,
                    color='darkgreen',
                    fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.7),
                    arrowprops=dict(arrowstyle='->', color='darkgreen', lw=3))
    
    # Word Error Rate
    bars2 = ax2.bar(labels, wers, color=colors, alpha=0.85, edgecolor='black', linewidth=1.5)
    ax2.set_ylabel('Word Error Rate (%)', fontweight='bold', fontsize=16)
    ax2.set_title('Script Normalization Impact on WER', fontweight='bold', fontsize=18, pad=20)
    ax2.set_ylim([0, max(wers) * 1.25 if max(wers) > 0 else 100])
    ax2.grid(axis='y', alpha=0.4, linestyle='--')
    ax2.set_axisbelow(True)
    
    # Add value labels
    for bar, wer in zip(bars2, wers):
        height = bar.get_height()
        if height > 0:
            ax2.text(bar.get_x() + bar.get_width()/2., height + max(wers)*0.02,
                    f'{wer:.2f}%',
                    ha='center', va='bottom', fontweight='bold', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'phase1_comparison.png', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'phase1_comparison.pdf', bbox_inches='tight')
    print(f"✓ Generated: phase1_comparison.png/pdf")
    plt.close()


def generate_summary_table(results, output_dir):
    """Generate results summary table"""
    
    # Markdown table
    md_lines = []
    md_lines.append("# Phase 1: Script Normalization Ablation Results")
    md_lines.append("")
    md_lines.append("## LipBengal Dataset (100 epochs)")
    md_lines.append("")
    md_lines.append("| Experiment | Method | Top-1 Acc (%) | WER (%) | CER (%) | Epochs Trained | Best Epoch |")
    md_lines.append("|------------|--------|---------------|---------|---------|----------------|------------|")
    
    experiments = [
        ('S1_raw', 'Raw Bengali Script'),
        ('S1_phonetic', 'Phonetic Romanization (Banglish)'),
        ('S1_simple', 'Simple Transliteration'),
        ('S1_mixed', 'Mixed Approach'),
    ]
    
    best_acc = 0
    for exp_id, exp_name in experiments:
        if exp_id in results and 'best_accuracy' in results[exp_id]:
            best_acc = max(best_acc, results[exp_id]['best_accuracy'])
    
    for exp_id, exp_name in experiments:
        if exp_id in results and 'best_accuracy' in results[exp_id]:
            r = results[exp_id]
            acc_str = f"**{r['best_accuracy']:.2f}**" if r['best_accuracy'] == best_acc else f"{r['best_accuracy']:.2f}"
            md_lines.append(f"| {exp_id} | {exp_name} | {acc_str} | {r['best_wer']:.2f} | {r['best_cer']:.2f} | {r['final_epoch']} | {r['best_epoch']} |")
        else:
            md_lines.append(f"| {exp_id} | {exp_name} | - | - | - | - | - |")
    
    md_lines.append("")
    md_lines.append("## Key Findings")
    md_lines.append("")
    
    if 'S1_raw' in results and 'S1_phonetic' in results:
        raw_acc = results['S1_raw']['best_accuracy']
        phon_acc = results['S1_phonetic']['best_accuracy']
        if raw_acc > 0 and phon_acc > 0:
            improvement = phon_acc - raw_acc
            md_lines.append(f"- **Script Normalization Impact**: Phonetic romanization improved accuracy by **+{improvement:.2f} percentage points** ({raw_acc:.2f}% → {phon_acc:.2f}%)")
            md_lines.append(f"- **Relative Improvement**: {(improvement/raw_acc)*100:.1f}% relative improvement")
    
    output_path = output_dir / 'phase1_results.md'
    with open(output_path, 'w') as f:
        f.write('\n'.join(md_lines))
    
    print(f"✓ Generated: phase1_results.md")
    
    # Also save as JSON
    json_data = {
        'phase': 1,
        'dataset': 'LipBengal',
        'target_epochs': 100,
        'experiments': results
    }
    
    with open(output_dir / 'phase1_results.json', 'w') as f:
        json.dump(json_data, f, indent=2)
    
    print(f"✓ Generated: phase1_results.json")


def main():
    print("=" * 80)
    print("Phase 1 Results Compilation - Script Normalization Ablation")
    print("=" * 80)
    print()
    
    # Setup
    base_dir = Path('callbacks/LipBengal/AV/ablations')
    output_dir = Path('paper_figures')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    experiments = ['S1_raw', 'S1_phonetic', 'S1_simple', 'S1_mixed']
    results = {}
    
    # Extract metrics from each experiment
    print("Extracting metrics from TensorBoard logs...")
    print()
    
    for exp_id in experiments:
        exp_dir = base_dir / exp_id / 'logs'
        print(f"Processing {exp_id}...")
        
        if exp_dir.exists():
            metrics = extract_tensorboard_metrics(str(exp_dir))
            if metrics:
                results[exp_id] = metrics
                print(f"  ✓ Epochs: {metrics['final_epoch']}, "
                      f"Best Acc: {metrics['best_accuracy']:.2f}% (epoch {metrics['best_epoch']})")
            else:
                print(f"  ⚠ No metrics found")
        else:
            print(f"  ⚠ Log directory not found: {exp_dir}")
    
    print()
    
    if not results:
        print("❌ No results found. Experiments may still be running.")
        print("   Run this script again after experiments complete.")
        return
    
    # Generate visualizations
    print("=" * 80)
    print("Generating Visualizations...")
    print("=" * 80)
    print()
    
    generate_phase1_comparison_plot(results, output_dir)
    generate_summary_table(results, output_dir)
    
    print()
    print("=" * 80)
    print("✓ Results Compiled Successfully!")
    print("=" * 80)
    print()
    print(f"Output directory: {output_dir.absolute()}")
    print(f"Files generated:")
    for f in sorted(output_dir.glob('phase1_*')):
        print(f"  - {f.name}")
    
    print()
    print("Use these files in your paper!")


if __name__ == '__main__':
    main()

