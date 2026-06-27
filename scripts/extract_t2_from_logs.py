#!/usr/bin/env python3
"""Extract T2 experiment metrics from training log files"""

import re
from pathlib import Path

def extract_metrics_from_log(log_file):
    """Extract all evaluation metrics from a training log file"""
    metrics = {
        'epoch': [],
        'eval_acc': [],
        'eval_wer': [],
        'eval_cer': []
    }
    
    current_epoch = None
    
    try:
        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                # Track current epoch
                epoch_match = re.search(r'Epoch (\d+)/\d+:', line)
                if epoch_match:
                    current_epoch = int(epoch_match.group(1))
                
                # Look for evaluation metrics in different formats
                # Format 1: "eval loss: X eval acc: Y eval wer: Z eval cer: W"
                if 'eval acc:' in line or 'Evaluation' in line:
                    try:
                        acc_match = re.search(r'eval acc:\s*([\d.]+)|CategoricalAccuracy[:\s]+([\d.]+)', line, re.IGNORECASE)
                        wer_match = re.search(r'eval wer:\s*([\d.]+)|WordErrorRate[:\s]+([\d.]+)', line, re.IGNORECASE)
                        cer_match = re.search(r'eval cer:\s*([\d.]+)|CharacterErrorRate[:\s]+([\d.]+)', line, re.IGNORECASE)
                        
                        if acc_match:
                            acc = float(acc_match.group(1) or acc_match.group(2))
                            wer = float(wer_match.group(1) or wer_match.group(2)) if wer_match else None
                            cer = float(cer_match.group(1) or cer_match.group(2)) if cer_match else None
                            
                            if current_epoch is not None:
                                metrics['epoch'].append(current_epoch)
                                metrics['eval_acc'].append(acc)
                                metrics['eval_wer'].append(wer)
                                metrics['eval_cer'].append(cer)
                    except Exception as e:
                        pass
    except Exception as e:
        print(f"Error reading {log_file}: {e}")
    
    return metrics

def find_latest_log(log_dir):
    """Find the most recent log file in directory"""
    log_dir = Path(log_dir)
    if not log_dir.exists():
        return None
    
    log_files = list(log_dir.glob('*.log'))
    if not log_files:
        return None
    
    return max(log_files, key=lambda p: p.stat().st_mtime)

def main():
    base_path = Path('/home/thesis/Thesis/AVEC')
    
    experiments = {
        'T2_freeze_0ep': 'logs/ablations/LipBengal/T2_freeze_0ep',
        'T2_freeze_10ep': 'logs/ablations/LipBengal/T2_freeze_10ep',
    }
    
    print("=" * 80)
    print("Extracting T2 Experiment Results from Training Logs")
    print("=" * 80)
    print()
    
    results = {}
    
    for exp_name, log_dir_rel in experiments.items():
        log_dir = base_path / log_dir_rel
        print(f"Processing: {exp_name}")
        
        latest_log = find_latest_log(log_dir)
        if not latest_log:
            print(f"  ✗ No log files found in {log_dir}")
            print()
            continue
        
        print(f"  Reading: {latest_log.name}")
        metrics = extract_metrics_from_log(latest_log)
        
        if not metrics['epoch']:
            print(f"  ⚠️  No evaluation metrics found in log file")
            print()
            continue
        
        # Get best metrics
        best_acc_idx = metrics['eval_acc'].index(max(metrics['eval_acc']))
        best_acc = metrics['eval_acc'][best_acc_idx]
        
        # Get best WER and CER (minimum values)
        wer_values = [w for w in metrics['eval_wer'] if w is not None]
        cer_values = [c for c in metrics['eval_cer'] if c is not None]
        
        best_wer = min(wer_values) if wer_values else None
        best_cer = min(cer_values) if cer_values else None
        
        # Get final metrics
        final_acc = metrics['eval_acc'][-1]
        final_wer = metrics['eval_wer'][-1] if metrics['eval_wer'][-1] is not None else best_wer
        final_cer = metrics['eval_cer'][-1] if metrics['eval_cer'][-1] is not None else best_cer
        
        results[exp_name] = {
            'best_accuracy': best_acc,
            'best_wer': best_wer,
            'best_cer': best_cer,
            'final_accuracy': final_acc,
            'final_wer': final_wer,
            'final_cer': final_cer,
            'epochs_trained': len(metrics['epoch']),
            'best_epoch': metrics['epoch'][best_acc_idx]
        }
        
        print(f"  ✓ Epochs trained: {results[exp_name]['epochs_trained']}")
        print(f"  ✓ Best Accuracy: {best_acc:.2f}% (epoch {results[exp_name]['best_epoch']})")
        if best_wer:
            print(f"  ✓ Best WER: {best_wer:.2f}%")
        if best_cer:
            print(f"  ✓ Best CER: {best_cer:.2f}%")
        print(f"  • Final Accuracy: {final_acc:.2f}%")
        if final_wer:
            print(f"  • Final WER: {final_wer:.2f}%")
        if final_cer:
            print(f"  • Final CER: {final_cer:.2f}%")
        print()
    
    # Print CSV format
    if results:
        print("=" * 80)
        print("CSV Format (for lipbengal_results.csv)")
        print("=" * 80)
        print("Experiment,Accuracy,WER,CER")
        for exp_name in ['T2_freeze_0ep', 'T2_freeze_10ep']:
            if exp_name in results:
                r = results[exp_name]
                wer = r['best_wer'] if r['best_wer'] else 'N/A'
                cer = r['best_cer'] if r['best_cer'] else 'N/A'
                print(f"{exp_name},{r['best_accuracy']},{wer},{cer}")
        print()
        
        # Performance range analysis
        if 'T2_freeze_0ep' in results and 'T2_freeze_10ep' in results:
            acc_0ep = results['T2_freeze_0ep']['best_accuracy']
            acc_10ep = results['T2_freeze_10ep']['best_accuracy']
            perf_range = abs(acc_0ep - acc_10ep)
            
            print("=" * 80)
            print("PERFORMANCE RANGE ANALYSIS")
            print("=" * 80)
            print(f"T2_freeze_0ep (0 epochs):   {acc_0ep:.2f}%")
            print(f"T2_freeze_10ep (10 epochs): {acc_10ep:.2f}%")
            print(f"Performance Range:          {perf_range:.2f}%")
            print()
            print("For comparison:")
            print("LRW-AR Performance Range:   5.22% (72.38% - 67.16%)")
            print()

if __name__ == '__main__':
    main()




