#!/usr/bin/env python3
"""
Monitor dual-GPU distributed training progress.
Shows per-GPU utilization, throughput, and training metrics.
"""

import sys
import re
import time
import subprocess
from pathlib import Path

def get_gpu_stats():
    """Get current GPU utilization stats."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            stats = []
            for line in lines:
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 5:
                    stats.append({
                        'index': int(parts[0]),
                        'util': int(parts[1]),
                        'mem_used': int(parts[2]),
                        'mem_total': int(parts[3]),
                        'temp': int(parts[4])
                    })
            return stats
    except Exception:
        pass
    return []

def get_latest_log():
    """Get the most recent dual-GPU training log file."""
    log_dir = Path("logs/dual_gpu_training")
    if not log_dir.exists():
        return None
    
    log_files = list(log_dir.glob("dual_gpu_epoch188_*.log"))
    if not log_files:
        return None
    
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
    lr_pattern = r'lr:\s*([0-9.e+-]+)'
    speed_pattern = r'(\d+\.\d+)it/s'
    
    epochs = list(re.finditer(epoch_pattern, content))
    steps = list(re.finditer(step_pattern, content))
    losses = list(re.finditer(loss_pattern, content))
    lrs = list(re.finditer(lr_pattern, content))
    speeds = list(re.finditer(speed_pattern, content))
    
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
    
    # Get recent learning rate
    latest_lr = float(lrs[-1].group(1)) if lrs else None
    
    # Get recent speed (iterations per second)
    recent_speeds = [float(m.group(1)) for m in speeds[-10:]]
    avg_speed = sum(recent_speeds) / len(recent_speeds) if recent_speeds else None
    
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
        'latest_lr': latest_lr,
        'avg_speed': avg_speed,
        'eval_results': eval_results
    }

def format_progress(info, gpu_stats):
    """Format progress information for display."""
    lines = []
    lines.append("=" * 80)
    lines.append("  DUAL-GPU DISTRIBUTED TRAINING MONITOR")
    lines.append("=" * 80)
    
    # GPU Status
    if gpu_stats:
        lines.append("\n📊 GPU Status:")
        lines.append("-" * 80)
        for gpu in gpu_stats:
            mem_pct = (gpu['mem_used'] / gpu['mem_total']) * 100
            bar_len = 30
            filled = int(bar_len * gpu['util'] / 100)
            util_bar = '█' * filled + '░' * (bar_len - filled)
            lines.append(
                f"  GPU {gpu['index']}: {util_bar} {gpu['util']:3d}% | "
                f"Mem: {gpu['mem_used']:5d}/{gpu['mem_total']:5d}MB ({mem_pct:5.1f}%) | "
                f"Temp: {gpu['temp']:2d}°C"
            )
    
    if not info:
        lines.append("\n⏳ Waiting for training metrics...")
        lines.append("=" * 80)
        return "\n".join(lines)
    
    # Training Progress
    lines.append("\n📈 Training Progress:")
    lines.append("-" * 80)
    
    progress_pct = (info['current_epoch'] / info['total_epochs']) * 100
    epochs_remaining = info['total_epochs'] - info['current_epoch']
    
    lines.append(f"  Epoch: {info['current_epoch']}/{info['total_epochs']} ({progress_pct:.1f}%)")
    lines.append(f"  Epochs Remaining: {epochs_remaining}")
    
    if info['latest_step']:
        lines.append(f"  Step: {info['latest_step']:,}")
    
    if info['avg_recent_loss']:
        lines.append(f"  Average Recent Loss: {info['avg_recent_loss']:.4f}")
    
    if info['latest_lr']:
        lines.append(f"  Learning Rate: {info['latest_lr']:.2e}")
    
    if info['avg_speed']:
        lines.append(f"  Speed: {info['avg_speed']:.2f} it/s")
        # Estimate time per epoch
        # Assuming ~1371 iterations per epoch (from single GPU logs)
        time_per_epoch_min = (1371 / info['avg_speed']) / 60
        time_remaining_hrs = (epochs_remaining * time_per_epoch_min) / 60
        lines.append(f"  Est. Time/Epoch: {time_per_epoch_min:.1f} min")
        lines.append(f"  Est. Time Remaining: {time_remaining_hrs:.1f} hours")
    
    # Evaluation Results
    if info['eval_results']:
        lines.append("\n📊 Validation Results:")
        lines.append("-" * 80)
        for ep, acc in info['eval_results'][-5:]:  # Show last 5
            lines.append(f"  Epoch {ep:3d}: Accuracy = {acc:.4f} ({acc*100:.2f}%)")
    
    lines.append("=" * 80)
    
    return "\n".join(lines)

def main():
    """Main monitoring loop."""
    print("🔍 Monitoring dual-GPU distributed training...")
    print("   Press Ctrl+C to exit")
    print()
    
    try:
        while True:
            log_file = get_latest_log()
            gpu_stats = get_gpu_stats()
            
            if log_file:
                info = parse_training_progress(log_file)
            else:
                info = None
            
            # Clear screen
            print("\033[2J\033[H")  # ANSI escape codes
            
            print(format_progress(info, gpu_stats))
            
            if log_file:
                print(f"\nLog file: {log_file}")
            print(f"Last updated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
            print("\nRefreshing every 5 seconds... (Ctrl+C to exit)")
            
            time.sleep(5)
            
    except KeyboardInterrupt:
        print("\n\n✋ Monitoring stopped by user")
        sys.exit(0)

if __name__ == '__main__':
    main()







