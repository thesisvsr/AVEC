#!/usr/bin/env python3
"""
Monitor the training progress for resume from epoch 177.
Shows current epoch, loss trends, and estimated time remaining.
"""

import sys
import re
import time
from pathlib import Path
import glob

def get_latest_log():
    """Get the most recent resume training log file."""
    log_dir = Path("logs/resume_training")
    if not log_dir.exists():
        return None
    
    log_files = list(log_dir.glob("resume_epoch177_*.log"))
    if not log_files:
        return None
    
    # Sort by modification time, latest first
    log_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return log_files[0]

def parse_training_progress(log_file):
    """Extract training progress from log file."""
    with open(log_file, 'r') as f:
        content = f.read()
    
    # Find current epoch and step
    epoch_pattern = r'Epoch\s+(\d+)/(\d+)'
    step_pattern = r'step:\s*(\d+)'
    loss_pattern = r'loss:\s*([0-9.]+)'
    
    epochs = list(re.finditer(epoch_pattern, content))
    steps = list(re.finditer(step_pattern, content))
    losses = list(re.finditer(loss_pattern, content))
    
    if not epochs:
        return None
    
    # Get latest values
    latest_epoch_match = epochs[-1]
    current_epoch = int(latest_epoch_match.group(1))
    total_epochs = int(latest_epoch_match.group(2))
    
    latest_step = int(steps[-1].group(1)) if steps else None
    
    # Get recent losses (last 20)
    recent_losses = [float(m.group(1)) for m in losses[-20:]]
    avg_recent_loss = sum(recent_losses) / len(recent_losses) if recent_losses else None
    
    # Check for evaluation metrics
    eval_pattern = r'Eval.*?Epoch[:\s]+(\d+).*?output/accuracy[:\s]+([0-9.]+)'
    eval_matches = list(re.finditer(eval_pattern, content, re.IGNORECASE))
    
    eval_results = []
    for match in eval_matches:
        ep = int(match.group(1))
        acc = float(match.group(2))
        eval_results.append((ep, acc))
    
    return {
        'current_epoch': current_epoch,
        'total_epochs': total_epochs,
        'latest_step': latest_step,
        'avg_recent_loss': avg_recent_loss,
        'eval_results': eval_results
    }

def format_progress(info):
    """Format progress information for display."""
    if not info:
        return "No training progress found yet"
    
    lines = []
    lines.append("=" * 70)
    lines.append("  TRAINING PROGRESS MONITOR")
    lines.append("=" * 70)
    
    # Current status
    progress_pct = (info['current_epoch'] / info['total_epochs']) * 100
    lines.append(f"Epoch: {info['current_epoch']}/{info['total_epochs']} ({progress_pct:.1f}%)")
    
    if info['latest_step']:
        lines.append(f"Step: {info['latest_step']}")
    
    if info['avg_recent_loss']:
        lines.append(f"Average Recent Loss: {info['avg_recent_loss']:.4f}")
    
    # Evaluation results
    if info['eval_results']:
        lines.append("\n" + "-" * 70)
        lines.append("Validation Results:")
        lines.append("-" * 70)
        for ep, acc in info['eval_results'][-5:]:  # Show last 5
            lines.append(f"  Epoch {ep:3d}: Accuracy = {acc:.4f} ({acc*100:.2f}%)")
    
    lines.append("=" * 70)
    
    return "\n".join(lines)

def main():
    """Main monitoring loop."""
    print("🔍 Monitoring resume training from epoch 177...")
    print("   Press Ctrl+C to exit")
    print()
    
    try:
        while True:
            log_file = get_latest_log()
            
            if not log_file:
                print("⏳ Waiting for training to start...")
                time.sleep(10)
                continue
            
            info = parse_training_progress(log_file)
            
            # Clear screen (simple version)
            print("\033[2J\033[H")  # ANSI escape codes
            
            print(format_progress(info))
            print(f"\nLog file: {log_file}")
            print(f"Last updated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
            print("\nRefreshing every 30 seconds... (Ctrl+C to exit)")
            
            time.sleep(30)
            
    except KeyboardInterrupt:
        print("\n\n✋ Monitoring stopped by user")
        sys.exit(0)

if __name__ == '__main__':
    main()







