#!/bin/bash
#
# Phase 1 Runner with Virtual Environment Activation
#

set -e

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

# Activate virtual environment
if [ -d ".venv" ]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
else
    echo "ERROR: Virtual environment not found at .venv"
    echo "Please create it first: python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

# Verify PyTorch is available
python3 -c "import torch; print('✓ PyTorch', torch.__version__, 'found')" || {
    echo "ERROR: PyTorch not found. Install with: pip install torch torchvision torchaudio"
    exit 1
}

# Run Phase 1 for both datasets
echo ""
echo "=========================================================================="
echo "PHASE 1: SCRIPT NORMALIZATION EXPERIMENTS"
echo "=========================================================================="
echo "This will run 8 experiments (4 per dataset)"
echo "Estimated time: ~240 hours on single GPU (10 days)"
echo "=========================================================================="
echo ""

# Launch LipBengal experiments
echo "Starting Phase 1 on LipBengal..."
./scripts/batch_ablation.sh phase1-lipbengal --gpus ${1:-0} > logs/phase1_lipbengal.log 2>&1 &
PID1=$!
echo "  → LipBengal experiments running (PID: $PID1)"
echo "  → Log: logs/phase1_lipbengal.log"

# Wait a bit before starting LRW-AR to avoid conflicts
sleep 5

# Launch LRW-AR experiments
echo ""
echo "Starting Phase 1 on LRW-AR..."
./scripts/batch_ablation.sh phase1-lrwar --gpus ${1:-0} > logs/phase1_lrwar.log 2>&1 &
PID2=$!
echo "  → LRW-AR experiments running (PID: $PID2)"
echo "  → Log: logs/phase1_lrwar.log"

echo ""
echo "=========================================================================="
echo "Phase 1 experiments launched!"
echo "=========================================================================="
echo ""
echo "Monitor progress:"
echo "  LipBengal: tail -f logs/phase1_lipbengal.log"
echo "  LRW-AR:    tail -f logs/phase1_lrwar.log"
echo ""
echo "Check experiment logs:"
echo "  ls logs/ablations/LipBengal/*/training.log"
echo "  ls logs/ablations/LRW-AR/*/training.log"
echo ""
echo "Kill experiments if needed:"
echo "  kill $PID1 $PID2"
echo ""


