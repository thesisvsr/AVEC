#!/usr/bin/env python3
"""Extract results from T2_freeze_0ep and T2_freeze_10ep experiments"""

import sys
from pathlib import Path

try:
    from tensorboard.backend.event_processing import event_accumulator
    TB_AVAILABLE = True
except ImportError:
    print("TensorBoard not available. Please install: pip install tensorboard")
    TB_AVAILABLE = False
    sys.exit(1)


def extract_tensorboard_metrics(log_dir):
    """Extract metrics from TensorBoard event file"""
    try:
        ea = event_accumulator.EventAccumulator(log_dir)
        ea.Reload()
        
        metrics = {
            'val_accuracy': [],
            'val_wer': [],
            'val_cer': [],
        }
        
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
        
        # Get final metrics (last epoch)
        final_acc = metrics['val_accuracy'][-1][1] if metrics['val_accuracy'] else 0.0
        final_wer = metrics['val_wer'][-1][1] if metrics['val_wer'] else 0.0
        final_cer = metrics['val_cer'][-1][1] if metrics['val_cer'] else 0.0
        
        return {
            'best_accuracy': best_acc,
            'best_wer': best_wer,
            'best_cer': best_cer,
            'final_accuracy': final_acc,
            'final_wer': final_wer,
            'final_cer': final_cer,
            'final_epoch': int(max([s for s, _ in metrics['val_accuracy']])) if metrics['val_accuracy'] else 0,
            'best_epoch': int(best_epoch),
        }
    
    except Exception as e:
        print(f"Error extracting metrics: {e}")
        return {}


def main():
    experiments = [
        ('T2_freeze_0ep', 'callbacks/LipBengal/AV/ablations/T2_freeze_0ep/logs'),
        ('T2_freeze_10ep', 'callbacks/LipBengal/AV/ablations/T2_freeze_10ep/logs'),
    ]
    
    print("=" * 80)
    print("LipBengal T2 Experiment Results")
    print("=" * 80)
    print()
    
    results = {}
    
    for exp_name, log_dir in experiments:
        log_path = Path(log_dir)
        if not log_path.exists():
            print(f"⚠️  {exp_name}: Log directory not found")
            continue
        
        print(f"Extracting: {exp_name}")
        metrics = extract_tensorboard_metrics(str(log_path))
        
        if metrics:
            results[exp_name] = metrics
            print(f"  ✓ Epochs trained: {metrics['final_epoch']}")
            print(f"  ✓ Best Accuracy: {metrics['best_accuracy']:.2f}% (epoch {metrics['best_epoch']})")
            print(f"  ✓ Best WER: {metrics['best_wer']:.2f}%")
            print(f"  ✓ Best CER: {metrics['best_cer']:.2f}%")
            print(f"  • Final Accuracy: {metrics['final_accuracy']:.2f}%")
            print(f"  • Final WER: {metrics['final_wer']:.2f}%")
            print(f"  • Final CER: {metrics['final_cer']:.2f}%")
        else:
            print(f"  ✗ Failed to extract metrics")
        print()
    
    # Summary
    print("=" * 80)
    print("SUMMARY - CSV Format for lipbengal_results.csv")
    print("=" * 80)
    print()
    print("Experiment,Accuracy,WER,CER")
    
    for exp_name in ['T2_freeze_0ep', 'T2_freeze_10ep']:
        if exp_name in results:
            m = results[exp_name]
            print(f"{exp_name},{m['best_accuracy']},{m['best_wer']},{m['best_cer']}")
    
    print()
    
    # Calculate performance range
    if 'T2_freeze_0ep' in results and 'T2_freeze_10ep' in results:
        acc_0ep = results['T2_freeze_0ep']['best_accuracy']
        acc_10ep = results['T2_freeze_10ep']['best_accuracy']
        perf_range = abs(acc_0ep - acc_10ep)
        
        print("=" * 80)
        print("PERFORMANCE RANGE ANALYSIS")
        print("=" * 80)
        print()
        print(f"T2_freeze_0ep (0 epochs):   {acc_0ep:.2f}%")
        print(f"T2_freeze_10ep (10 epochs): {acc_10ep:.2f}%")
        print(f"Performance Range:          {perf_range:.2f}%")
        print()
        print("For comparison:")
        print("LRW-AR Performance Range:   5.22% (72.38% - 67.16%)")
        print()


if __name__ == "__main__":
    main()




