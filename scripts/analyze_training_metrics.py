#!/usr/bin/env python3
"""
Analyze training log to determine if validation metrics are improving.
Exit codes:
  0 = Continue training (metrics improving)
  1 = Stop training (no improvement)
  2 = Insufficient data
"""

import re
import sys
import argparse

def extract_metrics(log_file):
    """Extract validation metrics from training log."""
    with open(log_file, 'r') as f:
        log_content = f.read()
    
    epoch_metrics = {}
    
    # Try multiple patterns to capture validation metrics
    patterns = [
        r'Eval.*?Epoch[:\s]+(\d+).*?WER[:\s]+([0-9.]+).*?Accuracy[:\s]+([0-9.]+)',
        r'Epoch[:\s]+(\d+).*?eval.*?wer[:\s]+([0-9.]+).*?acc[:\s]+([0-9.]+)',
        r'epoch[:\s]+(\d+).*?validation.*?wer[:\s]*=?\s*([0-9.]+).*?accuracy[:\s]*=?\s*([0-9.]+)',
        # Pattern for output/accuracy format
        r'Epoch[:\s]+(\d+).*?output/accuracy[:\s]+([0-9.]+)',
        # Pattern for metrics dict format
        r'Epoch[:\s]+(\d+).*?metrics.*?accuracy.*?([0-9.]+)',
    ]
    
    for pattern in patterns:
        for match in re.finditer(pattern, log_content, re.IGNORECASE):
            epoch = int(match.group(1))
            if len(match.groups()) >= 3:
                wer = float(match.group(2))
                acc = float(match.group(3))
            else:
                # Only accuracy found
                acc = float(match.group(2))
                wer = None
            
            if epoch not in epoch_metrics:
                epoch_metrics[epoch] = {}
            if len(match.groups()) >= 3:
                epoch_metrics[epoch]['wer'] = wer
            epoch_metrics[epoch]['acc'] = acc
    
    return epoch_metrics

def analyze_improvement(epoch_metrics, baseline_epoch=177):
    """Analyze if metrics are improving."""
    
    print("📊 Extracted Metrics:")
    print("-" * 70)
    
    sorted_epochs = sorted(epoch_metrics.keys())
    
    # Show last 10 epochs
    for ep in sorted_epochs[-10:]:
        m = epoch_metrics[ep]
        wer_str = f"WER={m.get('wer', 'N/A'):.4f}" if m.get('wer') is not None else "WER=N/A"
        acc_str = f"Accuracy={m.get('acc', 'N/A'):.4f}" if m.get('acc') is not None else "Accuracy=N/A"
        print(f"  Epoch {ep:3d}: {wer_str:20s} {acc_str}")
    
    if len(sorted_epochs) < 2:
        print("\n⚠️  Not enough metrics to determine trend")
        print("Decision: STOP (insufficient data)")
        return 2
    
    # Check if improving (focus on epochs >= baseline)
    relevant_epochs = [e for e in sorted_epochs if e >= baseline_epoch]
    
    if len(relevant_epochs) < 2:
        print("\n⚠️  Not enough recent metrics to determine trend")
        print("Decision: STOP (insufficient recent data)")
        return 2
    
    # Get baseline (epoch closest to baseline_epoch)
    baseline_ep = min(relevant_epochs, key=lambda x: abs(x - baseline_epoch))
    baseline = epoch_metrics[baseline_ep]
    
    # Get latest epoch
    latest_ep = relevant_epochs[-1]
    latest = epoch_metrics[latest_ep]
    
    print(f"\n📈 Trend Analysis:")
    print("-" * 70)
    print(f"  Baseline (Epoch {baseline_ep}):", end="")
    if baseline.get('wer') is not None:
        print(f" WER={baseline['wer']:.4f},", end="")
    print(f" Acc={baseline.get('acc', 0):.4f}")
    
    print(f"  Latest   (Epoch {latest_ep}):", end="")
    if latest.get('wer') is not None:
        print(f" WER={latest['wer']:.4f},", end="")
    print(f" Acc={latest.get('acc', 0):.4f}")
    
    # Calculate changes
    has_wer = baseline.get('wer') is not None and latest.get('wer') is not None
    
    if has_wer:
        wer_change = baseline['wer'] - latest['wer']  # Positive = improvement
        print(f"\n  WER Change: {wer_change:+.4f} ({'✓ better' if wer_change > 0 else '✗ worse'})")
    
    acc_change = latest.get('acc', 0) - baseline.get('acc', 0)  # Positive = improvement
    print(f"  Acc Change: {acc_change:+.4f} ({'✓ better' if acc_change > 0 else '✗ worse'})")
    
    # Decision criteria
    THRESHOLD_WER = -0.02  # Allow small WER increase (2%)
    THRESHOLD_ACC = 0.005  # Require 0.5% accuracy improvement
    
    # Check improvement
    wer_improving = (not has_wer) or (wer_change > THRESHOLD_WER)
    acc_improving = acc_change > THRESHOLD_ACC
    
    is_improving = wer_improving or acc_improving
    
    print("\n" + "=" * 70)
    if is_improving:
        print("✅ Decision: CONTINUE training")
        print("   Reason: Metrics show improvement or acceptable performance")
        if acc_improving:
            print(f"   - Accuracy improved by {acc_change*100:.2f}%")
        if has_wer and wer_change > 0:
            print(f"   - WER improved by {wer_change*100:.2f}%")
        print("=" * 70)
        return 0  # Continue
    else:
        print("🛑 Decision: STOP training")
        print("   Reason: No significant improvement detected")
        if not acc_improving:
            print(f"   - Accuracy change ({acc_change*100:.2f}%) below threshold ({THRESHOLD_ACC*100:.2f}%)")
        if has_wer and wer_change <= THRESHOLD_WER:
            print(f"   - WER not improving (change: {wer_change*100:.2f}%)")
        print("=" * 70)
        return 1  # Stop

def main():
    parser = argparse.ArgumentParser(description='Analyze training metrics')
    parser.add_argument('log_file', help='Path to training log file')
    parser.add_argument('--baseline-epoch', type=int, default=177, 
                        help='Baseline epoch for comparison')
    args = parser.parse_args()
    
    epoch_metrics = extract_metrics(args.log_file)
    exit_code = analyze_improvement(epoch_metrics, args.baseline_epoch)
    sys.exit(exit_code)

if __name__ == '__main__':
    main()








