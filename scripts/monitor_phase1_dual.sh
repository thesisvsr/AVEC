#!/bin/bash
#
# Monitor Phase 1 dual-GPU progress
#

cd "$(dirname "$0")/.."

while true; do
    clear
    echo "==================================================================="
    echo "         PHASE 1: DUAL-GPU PROGRESS MONITOR"
    echo "==================================================================="
    date
    echo ""
    
    # GPU Status
    echo "📊 GPU STATUS:"
    nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits | \
        awk -F',' '{printf "  GPU %s: %s%% util | %s/%s MB | %s°C\n", $1, $3, $4, $5, $6}'
    echo ""
    
    # LipBengal Progress
    echo "🔷 LipBengal (GPU 0):"
    for exp in S1_raw S1_simple S1_mixed; do
        dir="callbacks/LipBengal/AV/ablations/$exp"
        if [ -d "$dir" ]; then
            last=$(ls -t "$dir"/checkpoints_epoch_*.ckpt 2>/dev/null | head -1 | grep -oP 'epoch_\K[0-9]+' || echo "0")
            if [ "$last" = "0" ]; then
                status="❌ Not Started"
            elif [ "$last" -ge 100 ]; then
                status="✅ Complete ($last/100)"
            else
                pct=$((last * 100 / 100))
                bar=$(printf "%${pct}s" | tr ' ' '█')
                status="🔄 Running [$bar] $last/100 (${pct}%)"
            fi
            printf "  %-12s: %s\n" "$exp" "$status"
        else
            printf "  %-12s: ❌ Not Started\n" "$exp"
        fi
    done
    echo ""
    
    # LRW-AR Progress
    echo "🔶 LRW-AR (GPU 1):"
    for exp in S1_raw S1_simple S1_mixed; do
        dir="callbacks/LRW-AR/AV/ablations/$exp"
        if [ -d "$dir" ]; then
            last=$(ls -t "$dir"/checkpoints_epoch_*.ckpt 2>/dev/null | head -1 | grep -oP 'epoch_\K[0-9]+' || echo "0")
            if [ "$last" = "0" ]; then
                status="❌ Not Started"
            elif [ "$last" -ge 100 ]; then
                status="✅ Complete ($last/100)"
            else
                pct=$((last * 100 / 100))
                bar=$(printf "%${pct}s" | tr ' ' '█')
                status="🔄 Running [$bar] $last/100 (${pct}%)"
            fi
            printf "  %-12s: %s\n" "$exp" "$status"
        else
            printf "  %-12s: ❌ Not Started\n" "$exp"
        fi
    done
    echo ""
    
    # Running Processes
    echo "🔧 ACTIVE TRAINING PROCESSES:"
    ps aux | grep "python3 main.py.*ablations" | grep -v grep | \
        awk '{printf "  PID %s: GPU %s | Config: %s\n", $2, $13, $15}' | \
        head -5 || echo "  No training processes running"
    echo ""
    
    # Recent Log Activity
    echo "📝 RECENT ACTIVITY (last 60 seconds):"
    find logs/ablations -name "training_dual_gpu_*.log" -mmin -1 -exec basename {} \; 2>/dev/null | \
        sed 's/^/  /' | head -5 || echo "  No recent log updates"
    echo ""
    
    echo "==================================================================="
    echo "Refreshing every 30 seconds... (Ctrl+C to stop)"
    echo "==================================================================="
    
    sleep 30
done

