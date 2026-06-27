#!/bin/bash
#
# Evaluate checkpoint epoch 177 on LipBengal test dataset
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

echo "========================================"
echo "Evaluating Checkpoint Epoch 177"
echo "========================================"
echo "Checkpoint: callbacks/LipBengal/AV/VisualCE/checkpoints_epoch_177_step_242564.ckpt"
echo "Dataset: LipBengal Test Set"
echo ""

# Run evaluation
python3 eval_epoch177.py 2>&1 | tee logs/eval_epoch177_$(date +%Y%m%d_%H%M%S).log

echo ""
echo "========================================"
echo "Evaluation complete!"
echo "========================================"
echo "Results saved to:"
echo "  - logs/visualce_epoch177_test_eval.tsv (detailed predictions)"
echo "  - logs/visualce_epoch177_test_summary.txt (summary)"
echo "  - logs/eval_epoch177_*.log (full log)"








