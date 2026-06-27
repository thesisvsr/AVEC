#!/bin/bash
################################################################################
# Quick Top-K Analysis Test - Run on best performing experiments
################################################################################

PYTHON="/home/thesis/Thesis/AVEC/.venv/bin/python3"
PROJECT_DIR="/home/thesis/Thesis/AVEC"
cd "$PROJECT_DIR" || exit 1

echo "========================================================================"
echo "                TOP-K ANALYSIS - QUICK TEST"
echo "========================================================================"
echo ""
echo "Testing on best experiments from each dataset..."
echo ""

# Test on LRW-AR best (T3_lr_1_0)
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  📊 LRW-AR: T3_lr_1_0 (Best: 80.14% Accuracy)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

$PYTHON scripts/topk_analysis.py \
    -c configs/LRW-AR/AV/ablations/T3_lr_1_0.py \
    --load_last \
    --k_values 1 3 5 10 \
    --output_dir topk_analysis_results/LRW-AR_T3_lr_1_0

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  📊 LipBengal: T3_lr_1_0 (Best: 35.77% Accuracy)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

$PYTHON scripts/topk_analysis.py \
    -c configs/LipBengal/AV/ablations/T3_lr_1_0.py \
    --load_last \
    --k_values 1 3 5 10 \
    --output_dir topk_analysis_results/LipBengal_T3_lr_1_0

echo ""
echo "========================================================================"
echo "                    QUICK TEST COMPLETE!"
echo "========================================================================"
echo ""
echo "Results saved in: $PROJECT_DIR/topk_analysis_results/"
echo ""
echo "To run on ALL experiments, use:"
echo "  bash scripts/run_topk_analysis_all.sh"
echo ""



