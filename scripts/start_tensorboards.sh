#!/bin/bash
#
# Start all TensorBoard instances for Phase 1 monitoring
#

cd "$(dirname "$0")/.."
PROJECT_ROOT=$(pwd)

# Activate venv
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Kill existing TensorBoards
echo "Stopping existing TensorBoard instances..."
pkill -f "tensorboard.*--port.*600" 2>/dev/null
sleep 2

# Start TensorBoard for LipBengal ablations
echo "Starting TensorBoard for LipBengal ablations on port 6006..."
nohup tensorboard --logdir callbacks/LipBengal/AV/ablations --port 6006 --bind_all \
    > /tmp/tensorboard_lipbengal_6006.log 2>&1 &

# Start TensorBoard for LRW-AR ablations
echo "Starting TensorBoard for LRW-AR ablations on port 6009..."
nohup tensorboard --logdir callbacks/LRW-AR/AV/ablations --port 6009 --bind_all \
    > /tmp/tensorboard_lrwar_6009.log 2>&1 &

# Wait for them to start
sleep 3

echo ""
echo "==================================================================="
echo "              TensorBoard Instances Started"
echo "==================================================================="
echo ""
echo "📊 LipBengal Ablations:"
echo "   http://localhost:6006"
echo "   Experiments: S1_phonetic, S1_raw, S1_simple, S1_mixed"
echo ""
echo "📊 LRW-AR Ablations:"
echo "   http://localhost:6009"
echo "   Experiments: S1_phonetic, S1_raw, S1_simple, S1_mixed"
echo ""
echo "==================================================================="
echo ""

# Verify they're running
echo "Verifying TensorBoard processes..."
ps aux | grep "tensorboard.*--port" | grep -v grep | \
    awk '{print "  ✓ PID", $2, "- Port", $NF}' | grep -E "(6006|6009)"

echo ""
echo "Ports listening:"
ss -tuln | grep -E ":(6006|6009)" | awk '{print "  ✓", $5}' | cut -d: -f2

echo ""
echo "==================================================================="
echo "TensorBoard setup complete!"
echo "==================================================================="

