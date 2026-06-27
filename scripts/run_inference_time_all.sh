#!/bin/bash
################################################################################
# Measure Inference Time for All Datasets
################################################################################

PYTHON="/home/thesis/Thesis/AVEC/.venv/bin/python3"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
INFERENCE_SCRIPT="$SCRIPT_DIR/measure_inference_time.py"

# Output directory
OUTPUT_DIR="$PROJECT_DIR/inference_time_results"
mkdir -p "$OUTPUT_DIR"

echo "========================================================================"
echo "           INFERENCE TIME MEASUREMENT - ALL DATASETS"
echo "========================================================================"
echo "Output Directory: $OUTPUT_DIR"
echo ""

# Function to measure inference time
measure_inference() {
    local dataset=$1
    local experiment=$2
    
    config_file="configs/$dataset/AV/ablations/$experiment.py"
    
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  ⏱️  Measuring: $dataset / $experiment"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    cd "$PROJECT_DIR" || exit 1
    
    $PYTHON "$INFERENCE_SCRIPT" \
        -c "$config_file" \
        --num_batches 50 \
        --warmup_batches 5 \
        --output_dir "$OUTPUT_DIR"
    
    if [ $? -eq 0 ]; then
        echo "  ✓ Completed successfully"
    else
        echo "  ✗ Failed"
    fi
    echo ""
}

# Measure for each dataset using best model
echo ""
echo "========================================================================"
echo "  Measuring Inference Time (using best models)"
echo "========================================================================"
echo ""

# LRW-AR - Best model (T3_lr_1_0)
measure_inference "LRW-AR" "T3_lr_1_0"

# LipBengal - Best model (T3_lr_1_0)
measure_inference "LipBengal" "T3_lr_1_0"

echo ""
echo "========================================================================"
echo "              INFERENCE TIME MEASUREMENT COMPLETE"
echo "========================================================================"
echo "Results saved in: $OUTPUT_DIR"
echo ""

# Generate summary report
echo "Generating summary report..."
$PYTHON << 'EOF'
import json
import os
from pathlib import Path

output_dir = Path("inference_time_results")

# Collect all results
results = {}

for json_file in sorted(output_dir.glob("inference_time_*.json")):
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    dataset = data['dataset_name']
    if dataset not in results:
        results[dataset] = []
    results[dataset].append(data)

# Create summary report
report_path = output_dir / "INFERENCE_TIME_SUMMARY.txt"

with open(report_path, 'w') as f:
    f.write("="*80 + "\n")
    f.write("              INFERENCE TIME MEASUREMENT SUMMARY\n")
    f.write("="*80 + "\n\n")
    
    for dataset in sorted(results.keys()):
        f.write(f"\n{'='*80}\n")
        f.write(f"   {dataset}\n")
        f.write(f"{'='*80}\n\n")
        
        for data in results[dataset]:
            exp_name = data['experiment_name']
            f.write(f"{exp_name}:\n")
            f.write(f"  Device: {data['device']}\n")
            f.write(f"  Model:  {data['model_name']}\n\n")
            
            f.write(f"  Batch Time (ms):\n")
            f.write(f"    Mean:   {data['batch_time_ms']['mean']:>8.2f} ± {data['batch_time_ms']['std']:.2f}\n")
            f.write(f"    Median: {data['batch_time_ms']['median']:>8.2f}\n")
            f.write(f"    Range:  {data['batch_time_ms']['min']:.2f} - {data['batch_time_ms']['max']:.2f}\n\n")
            
            f.write(f"  Per-Sample Time (ms):\n")
            f.write(f"    Mean:   {data['sample_time_ms']['mean']:>8.2f} ± {data['sample_time_ms']['std']:.2f}\n")
            f.write(f"    Median: {data['sample_time_ms']['median']:>8.2f}\n")
            f.write(f"    Range:  {data['sample_time_ms']['min']:.2f} - {data['sample_time_ms']['max']:.2f}\n\n")
            
            f.write(f"  Throughput:\n")
            f.write(f"    {data['throughput']['samples_per_second']:.2f} samples/second\n")
            f.write(f"    {data['throughput']['batches_per_second']:.2f} batches/second\n\n")
            
            f.write(f"  Total Samples: {data['total_samples']}\n")
            f.write(f"  Batches Measured: {data['num_batches_measured']}\n\n")
    
    f.write("="*80 + "\n")
    f.write("KEY METRICS COMPARISON\n")
    f.write("="*80 + "\n\n")
    
    for dataset in sorted(results.keys()):
        f.write(f"{dataset}:\n")
        for data in results[dataset]:
            f.write(f"  {data['experiment_name']:30s} | ")
            f.write(f"{data['sample_time_ms']['mean']:6.2f} ms/sample | ")
            f.write(f"{data['throughput']['samples_per_second']:6.2f} samples/sec\n")
        f.write("\n")

print(f"\n✓ Summary report saved to: {report_path}")

# Display summary
with open(report_path, 'r') as f:
    print(f.read())

EOF

echo ""
echo "✓ Summary report generated"
echo ""



