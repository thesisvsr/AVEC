#!/bin/bash
#
# Batch Ablation Experiment Launcher
#
# Usage:
#   ./scripts/batch_ablation.sh phase1-lipbengal
#   ./scripts/batch_ablation.sh phase2-lrwar --gpus 0,1 --parallel
#   ./scripts/batch_ablation.sh all --dry-run
#

set -e  # Exit on error

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    source .venv/bin/activate
    echo "✓ Virtual environment activated"
fi

# Default options
GPUS="0"
PARALLEL=""
DRY_RUN=""
WORKERS="4"

# Parse command line arguments
PRESET=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --gpus)
            GPUS="$2"
            shift 2
            ;;
        --parallel)
            PARALLEL="--parallel"
            shift
            ;;
        --dry-run)
            DRY_RUN="--dry-run"
            shift
            ;;
        --workers)
            WORKERS="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 <preset> [options]"
            echo ""
            echo "Presets:"
            echo "  phase1-lipbengal    Phase 1 experiments on LipBengal"
            echo "  phase1-lrwar        Phase 1 experiments on LRW-AR"
            echo "  phase2-lipbengal    Phase 2 experiments on LipBengal"
            echo "  phase2-lrwar        Phase 2 experiments on LRW-AR"
            echo "  phase3-lipbengal    Phase 3 experiments on LipBengal"
            echo "  phase3-lrwar        Phase 3 experiments on LRW-AR"
            echo "  all-lipbengal       All phases on LipBengal"
            echo "  all-lrwar           All phases on LRW-AR"
            echo "  all                 All experiments on both datasets"
            echo ""
            echo "Options:"
            echo "  --gpus GPUS         Comma-separated GPU IDs (default: 0)"
            echo "  --parallel          Run experiments in parallel across GPUs"
            echo "  --dry-run           Generate configs only, don't train"
            echo "  --workers N         Number of data loading workers (default: 4)"
            echo "  --help              Show this help message"
            exit 0
            ;;
        *)
            if [ -z "$PRESET" ]; then
                PRESET="$1"
            else
                echo "Error: Unknown argument $1"
                exit 1
            fi
            shift
            ;;
    esac
done

if [ -z "$PRESET" ]; then
    echo "Error: No preset specified"
    echo "Run with --help for usage information"
    exit 1
fi

# Function to run launcher
run_launcher() {
    local phase="$1"
    local dataset="$2"
    
    echo "=========================================================================="
    echo "Launching Phase $phase experiments on $dataset"
    echo "GPUs: $GPUS"
    echo "Parallel: ${PARALLEL:-no}"
    echo "Dry run: ${DRY_RUN:-no}"
    echo "=========================================================================="
    
    python3 scripts/ablation_launcher.py \
        --phase "$phase" \
        --dataset "$dataset" \
        --gpus "$GPUS" \
        --workers "$WORKERS" \
        $PARALLEL \
        $DRY_RUN
}

# Execute based on preset
case $PRESET in
    phase1-lipbengal)
        run_launcher 1 LipBengal
        ;;
    
    phase1-lrwar)
        run_launcher 1 LRW-AR
        ;;
    
    phase2-lipbengal)
        run_launcher 2 LipBengal
        ;;
    
    phase2-lrwar)
        run_launcher 2 LRW-AR
        ;;
    
    phase3-lipbengal)
        run_launcher 3 LipBengal
        ;;
    
    phase3-lrwar)
        run_launcher 3 LRW-AR
        ;;
    
    all-lipbengal)
        run_launcher 1 LipBengal
        run_launcher 2 LipBengal
        run_launcher 3 LipBengal
        ;;
    
    all-lrwar)
        run_launcher 1 LRW-AR
        run_launcher 2 LRW-AR
        run_launcher 3 LRW-AR
        ;;
    
    all)
        echo "=========================================================================="
        echo "RUNNING ALL ABLATION EXPERIMENTS"
        echo "This will take a VERY long time!"
        echo "=========================================================================="
        
        # LipBengal
        run_launcher 1 LipBengal
        run_launcher 2 LipBengal
        run_launcher 3 LipBengal
        
        # LRW-AR
        run_launcher 1 LRW-AR
        run_launcher 2 LRW-AR
        run_launcher 3 LRW-AR
        ;;
    
    *)
        echo "Error: Unknown preset '$PRESET'"
        echo "Run with --help for available presets"
        exit 1
        ;;
esac

echo ""
echo "=========================================================================="
echo "Ablation experiments completed!"
echo "=========================================================================="
echo ""
echo "Next steps:"
echo "  1. Check results: python3 scripts/ablation_tracker.py --report"
echo "  2. Generate plots: python3 scripts/ablation_plotter.py --output plots/"
echo ""

