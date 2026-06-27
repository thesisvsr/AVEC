#!/bin/bash
#
# Monitor S1_simple completion and auto-start S1_mixed
#

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

source .venv/bin/activate

TARGET_EPOCHS=100
CHECK_INTERVAL=300  # Check every 5 minutes

echo "================================================"
echo "Auto-starter for S1_mixed"
echo "================================================"
echo "Waiting for S1_simple to reach epoch $TARGET_EPOCHS"
echo "Check interval: ${CHECK_INTERVAL}s"
echo "================================================"
echo ""

while true; do
    # Check if S1_simple has checkpoint >= 100 epochs
    latest=$(ls -v callbacks/LipBengal/AV/ablations/S1_simple/checkpoints_epoch_*.ckpt 2>/dev/null | tail -n 1)
    
    if [ -n "$latest" ]; then
        epoch=$(echo "$latest" | grep -oP 'epoch_\K\d+')
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] S1_simple at epoch $epoch/$TARGET_EPOCHS"
        
        if [ "$epoch" -ge "$TARGET_EPOCHS" ]; then
            echo ""
            echo "✓ S1_simple completed!"
            echo ""
            
            # Wait a bit for process cleanup
            sleep 10
            
            # Start S1_mixed
            echo "🚀 Starting S1_mixed..."
            nohup python3 main.py \
                --config_file configs/LipBengal/AV/ablations/S1_mixed.py \
                --mode training \
                -j 4 \
                > logs/s1_mixed_training.log 2>&1 &
            
            PID=$!
            echo "✓ S1_mixed started (PID: $PID)"
            echo ""
            echo "Monitor with: tail -f logs/s1_mixed_training.log"
            
            exit 0
        fi
    else
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] S1_simple: No checkpoints yet"
    fi
    
    sleep $CHECK_INTERVAL
done


