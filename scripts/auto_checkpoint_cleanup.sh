#!/bin/bash
# Auto-cleanup old checkpoints during training
# Keeps only the latest 2 checkpoints to save disk space

CHECKPOINT_DIR="callbacks/LipBengal/AV/best_config_early_stopping"

while true; do
  sleep 600  # Check every 10 minutes
  
  cd "/home/thesis/Thesis/AVEC/$CHECKPOINT_DIR"
  
  # Count checkpoints
  COUNT=$(ls -1 checkpoints_*.ckpt 2>/dev/null | wc -l)
  
  if [ $COUNT -gt 2 ]; then
    echo "[$(date)] Found $COUNT checkpoints, cleaning up old ones..."
    # Keep only the latest 2
    ls -t checkpoints_*.ckpt | tail -n +3 | xargs rm -f
    NEW_COUNT=$(ls -1 checkpoints_*.ckpt 2>/dev/null | wc -l)
    echo "[$(date)] Cleanup complete. Remaining: $NEW_COUNT checkpoints"
  fi
done
