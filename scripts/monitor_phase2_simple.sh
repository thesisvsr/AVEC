#!/bin/bash
# Simple Phase 2 monitoring script

cd "$(dirname "$0")/.."

echo "═══════════════════════════════════════════════════════════════════"
echo "              Phase 2 Progress Monitor"
echo "═══════════════════════════════════════════════════════════════════"
echo ""

# Check active processes
echo "Active Training:"
ps aux | grep "main.py" | grep -v grep | grep -E "T[123]_" | while read line; do
    CONFIG=$(echo "$line" | grep -oP 'configs/[^ ]+\.py')
    if [ -n "$CONFIG" ]; then
        DATASET=$(echo "$CONFIG" | cut -d'/' -f2)
        EXP=$(echo "$CONFIG" | grep -oP 'ablations/\K[^.]+')
        GPU=$(echo "$line" | grep -oP 'CUDA_VISIBLE_DEVICES=\K\d+' || echo "?")
        echo "  • $DATASET / $EXP (GPU $GPU)"
    fi
done

echo ""
echo "─────────────────────────────────────────────────────────────────────"
echo "Experiment Status:"
echo "─────────────────────────────────────────────────────────────────────"

for dataset in LipBengal LRW-AR; do
    echo ""
    echo "$dataset:"
    for exp in T1_scratch T1_frontend T1_backend T2_freeze_0ep T2_freeze_3ep T2_freeze_10ep T3_lr_0_1 T3_lr_0_5 T3_lr_1_0; do
        CKPT_DIR="callbacks/$dataset/AV/ablations/$exp"
        if [ -d "$CKPT_DIR" ]; then
            LATEST=$(ls -v "$CKPT_DIR"/checkpoints_epoch_*.ckpt 2>/dev/null | tail -1)
            if [ -n "$LATEST" ]; then
                EPOCH=$(echo "$LATEST" | grep -oP 'epoch_\K\d+')
                if [ "$EPOCH" -ge 100 ]; then
                    echo "  ✅ $exp: COMPLETE ($EPOCH epochs)"
                else
                    # Check if currently training
                    if ps aux | grep -v grep | grep "main.py" | grep -q "$exp"; then
                        echo "  🔄 $exp: TRAINING (Epoch $EPOCH/100)"
                    else
                        echo "  ⏸️  $exp: PAUSED (Epoch $EPOCH/100)"
                    fi
                fi
            else
                echo "  ⏳ $exp: Not started"
            fi
        else
            echo "  ⏳ $exp: Not started"
        fi
    done
done

echo ""
echo "─────────────────────────────────────────────────────────────────────"
echo "GPU Status:"
echo "─────────────────────────────────────────────────────────────────────"
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader | while IFS=',' read gpu name util mem_used mem_total; do
    echo "  GPU $gpu: $util ($(echo $mem_used | xargs) / $(echo $mem_total | xargs))"
done

echo ""
echo "─────────────────────────────────────────────────────────────────────"
echo "Storage:"
echo "─────────────────────────────────────────────────────────────────────"
df -h . | tail -1 | awk '{print "  Used: "$3" / "$2" | Free: "$4" ("$5" used)"}'

echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo ""
echo "Commands:"
echo "  Monitor logs:  tail -f logs/ablations/DATASET/EXPERIMENT/training_*.log"
echo "  TensorBoard:   http://localhost:6006 (LipBengal) | http://localhost:6007 (LRW-AR)"
echo "  Watch GPUs:    watch -n 2 nvidia-smi"
echo ""

