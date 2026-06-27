#!/bin/bash
# Seamless continuation script - monitors training and auto-continues

CALLBACK_DIR="/home/thesis/Thesis/AVEC/callbacks/LipBengal/AV/best_config_early_stopping"
CONFIG_PATH="configs/LipBengal/AV/best_config_early_stopping.py"
LOG_BASE="/home/thesis/Thesis/AVEC/logs"

cd /home/thesis/Thesis/AVEC
source .venv/bin/activate 2>/dev/null || true

echo "=== Training Continuation Monitor Started ==="
echo "Time: $(date)"
echo "Will automatically continue training at epoch 150 if early stopping hasn't triggered"
echo ""

while true; do
    # Check if training process is running
    if ! pgrep -f "main.py.*best_config_early_stopping" > /dev/null; then
        echo "[$(date)] Training process not running - checking last checkpoint..."
        
        # Get latest checkpoint
        LATEST_CKPT=$(ls -t "$CALLBACK_DIR"/checkpoints_epoch_*.ckpt 2>/dev/null | head -1)
        
        if [ -n "$LATEST_CKPT" ]; then
            LATEST_EPOCH=$(echo "$LATEST_CKPT" | grep -oP 'epoch_\K\d+')
            echo "[$(date)] Latest checkpoint: Epoch $LATEST_EPOCH"
            
            # Check if we should continue (epoch < 600 and no early stop file)
            if [ "$LATEST_EPOCH" -lt 600 ] && [ ! -f "$CALLBACK_DIR/.early_stopped" ]; then
                echo "[$(date)] Auto-continuing training from epoch $LATEST_EPOCH..."
                
                TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
                LOG_FILE="$LOG_BASE/training_continued_${TIMESTAMP}.log"
                
                CUDA_VISIBLE_DEVICES=0 python3 /home/thesis/Thesis/AVEC/main.py \
                    --config_file "$CONFIG_PATH" \
                    --mode training \
                    -j 4 \
                    --checkpoint "$LATEST_CKPT" \
                    > "$LOG_FILE" 2>&1 &
                
                TRAINING_PID=$!
                echo "[$(date)] Training restarted with PID: $TRAINING_PID"
                echo "[$(date)] Log: $LOG_FILE"
                sleep 60  # Wait before next check
            else
                echo "[$(date)] Training complete or early stopped. Exiting monitor."
                break
            fi
        else
            echo "[$(date)] No checkpoint found. Waiting..."
            sleep 300
        fi
    else
        # Training is running, just wait
        sleep 300  # Check every 5 minutes
    fi
done

echo "[$(date)] Monitor exiting."

