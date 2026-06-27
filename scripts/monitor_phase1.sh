#!/bin/bash
#
# Real-time Phase 1 Monitoring Dashboard
# Continuously tracks experiments and regenerates visualizations
#

cd "$(dirname "$0")/.."
source .venv/bin/activate 2>/dev/null

clear
echo "=================================================================="
echo "   Phase 1 Monitoring Dashboard - Script Normalization Study"
echo "=================================================================="
echo ""

while true; do
    # Clear previous content (keep header)
    tput cup 5 0
    tput ed
    
    echo "Current Time: $(date '+%Y-%m-%d %H:%M:%S')"
    echo ""
    echo "------------------------------------------------------------------"
    echo "EXPERIMENT STATUS"
    echo "------------------------------------------------------------------"
    
    for exp in S1_raw S1_phonetic S1_simple S1_mixed; do
        printf "%-15s : " "$exp"
        
        ckpt_dir="callbacks/LipBengal/AV/ablations/$exp"
        if [ -d "$ckpt_dir" ]; then
            latest=$(ls -t "$ckpt_dir"/checkpoints_epoch_*.ckpt 2>/dev/null | head -1)
            if [ -n "$latest" ]; then
                epoch=$(echo "$latest" | grep -oP 'epoch_\K\d+')
                if [ "$epoch" -ge 100 ]; then
                    echo "✓ COMPLETE (${epoch}/100 epochs)"
                else
                    # Check if training is active
                    if ps aux | grep -q "[p]ython.*${exp}"; then
                        echo "⚙ RUNNING (${epoch}/100 epochs)"
                    else
                        echo "⏸ PAUSED (${epoch}/100 epochs)"
                    fi
                fi
            else
                if ps aux | grep -q "[p]ython.*${exp}"; then
                    echo "⚙ STARTING..."
                else
                    echo "⏳ PENDING"
                fi
            fi
        else
            echo "⏳ PENDING"
        fi
    done
    
    echo ""
    echo "------------------------------------------------------------------"
    echo "CURRENT METRICS (from TensorBoard)"
    echo "------------------------------------------------------------------"
    
    python3 << 'PYEOF' 2>/dev/null
import sys
from pathlib import Path
try:
    from tensorboard.backend.event_processing import event_accumulator
    
    experiments = {
        'S1_raw': 'Raw Bengali',
        'S1_phonetic': 'Phonetic', 
        'S1_simple': 'Simple Translit',
        'S1_mixed': 'Mixed'
    }
    
    print(f"{'Experiment':<15} {'Epochs':<8} {'Acc (%)':<10} {'WER (%)':<10} {'CER (%)':<10}")
    print("-" * 55)
    
    for exp_id, exp_name in experiments.items():
        log_dir = f'callbacks/LipBengal/AV/ablations/{exp_id}/logs'
        if Path(log_dir).exists():
            try:
                ea = event_accumulator.EventAccumulator(log_dir)
                ea.Reload()
                
                if 'Evaluation-epoch/0/acc' in ea.Tags()['scalars']:
                    acc_events = ea.Scalars('Evaluation-epoch/0/acc')
                    wer_events = ea.Scalars('Evaluation-epoch/0/wer')
                    cer_events = ea.Scalars('Evaluation-epoch/0/cer')
                    
                    if acc_events:
                        last_acc = acc_events[-1].value
                        last_wer = wer_events[-1].value if wer_events else 0
                        last_cer = cer_events[-1].value if cer_events else 0
                        epochs = len(acc_events)
                        
                        print(f"{exp_name:<15} {epochs:<8} {last_acc:<10.2f} {last_wer:<10.2f} {last_cer:<10.2f}")
                    else:
                        print(f"{exp_name:<15} {'0':<8} {'-':<10} {'-':<10} {'-':<10}")
            except:
                print(f"{exp_name:<15} {'-':<8} {'-':<10} {'-':<10} {'-':<10}")
        else:
            print(f"{exp_name:<15} {'-':<8} {'-':<10} {'-':<10} {'-':<10}")
            
except ImportError:
    print("TensorBoard not available")
PYEOF
    
    echo ""
    echo "------------------------------------------------------------------"
    echo "SYSTEM RESOURCES"
    echo "------------------------------------------------------------------"
    
    # GPU usage
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | head -1 | \
            awk -F', ' '{printf "GPU %s: %s%% util, %s/%s MB memory\n", $1, $3, $4, $5}'
    fi
    
    # Disk usage
    df -h . | tail -1 | awk '{printf "Disk: %s used, %s available (%s used)\n", $3, $4, $5}'
    
    echo ""
    echo "------------------------------------------------------------------"
    echo "CONTROLS"
    echo "------------------------------------------------------------------"
    echo "Press Ctrl+C to stop monitoring"
    echo "Run: python3 scripts/compile_phase1_results.py  (to regenerate visuals)"
    echo "Log: tail -f logs/phase1_batch_100ep.log  (to view training details)"
    
    # Auto-regenerate visualizations every 10 minutes
    if [ $(($(date +%s) % 600)) -lt 30 ]; then
        echo ""
        echo "⟳ Auto-regenerating visualizations..."
        python3 scripts/compile_phase1_results.py > /dev/null 2>&1 &
    fi
    
    sleep 30
done


