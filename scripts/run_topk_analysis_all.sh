#!/bin/bash
################################################################################
# Run Top-K Analysis on All Completed Experiments
################################################################################

PYTHON="/home/thesis/Thesis/AVEC/.venv/bin/python3"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
TOPK_SCRIPT="$SCRIPT_DIR/topk_analysis.py"

# K values to evaluate
K_VALUES="1 3 5 10 20"

# Output directory
OUTPUT_DIR="$PROJECT_DIR/topk_analysis_results"
mkdir -p "$OUTPUT_DIR"

echo "========================================================================"
echo "                    TOP-K ANALYSIS - BATCH RUN"
echo "========================================================================"
echo "K Values: $K_VALUES"
echo "Output Directory: $OUTPUT_DIR"
echo ""

# Function to run top-k analysis for an experiment
run_topk() {
    local dataset=$1
    local experiment=$2
    
    config_file="configs/$dataset/AV/ablations/$experiment.py"
    callback_dir="callbacks/$dataset/AV/ablations/$experiment"
    
    # Check if experiment completed
    if [ ! -f "$callback_dir/checkpoints_epoch_99"*".ckpt" ]; then
        echo "  ⚠️  Skipping $dataset/$experiment - No epoch 99 checkpoint found"
        return
    fi
    
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  📊 Running: $dataset / $experiment"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    cd "$PROJECT_DIR" || exit 1
    
    $PYTHON "$TOPK_SCRIPT" \
        -c "$config_file" \
        --load_last \
        --k_values $K_VALUES \
        --output_dir "$OUTPUT_DIR/${dataset}_${experiment}" \
        2>&1 | tee "$OUTPUT_DIR/${dataset}_${experiment}_log.txt"
    
    if [ $? -eq 0 ]; then
        echo "  ✓ Completed successfully"
    else
        echo "  ✗ Failed"
    fi
    echo ""
}

# LRW-AR Experiments
echo ""
echo "========================================================================"
echo "  LRW-AR (Arabic) Experiments"
echo "========================================================================"
echo ""

# T1 - Training Strategies
run_topk "LRW-AR" "T1_scratch"
run_topk "LRW-AR" "T1_frontend"
run_topk "LRW-AR" "T1_backend"

# T2 - Transfer Learning
run_topk "LRW-AR" "T2_freeze_0ep"
run_topk "LRW-AR" "T2_freeze_3ep"
run_topk "LRW-AR" "T2_freeze_10ep"

# T3 - Learning Rates
run_topk "LRW-AR" "T3_lr_0_1"
run_topk "LRW-AR" "T3_lr_0_5"
run_topk "LRW-AR" "T3_lr_1_0"

# LipBengal Experiments
echo ""
echo "========================================================================"
echo "  LipBengal (Bengali) Experiments"
echo "========================================================================"
echo ""

# S1 - Data Preprocessing
run_topk "LipBengal" "S1_raw"

# T1 - Training Strategies
run_topk "LipBengal" "T1_scratch"
run_topk "LipBengal" "T1_backend"

# T2 - Transfer Learning
run_topk "LipBengal" "T2_freeze_3ep"

# T3 - Learning Rates
run_topk "LipBengal" "T3_lr_0_1"
run_topk "LipBengal" "T3_lr_0_5"
run_topk "LipBengal" "T3_lr_1_0"

echo ""
echo "========================================================================"
echo "                    TOP-K ANALYSIS COMPLETE"
echo "========================================================================"
echo "Results saved in: $OUTPUT_DIR"
echo ""

# Generate summary report
echo "Generating summary report..."
$PYTHON << 'EOF'
import json
import os
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

output_dir = Path("topk_analysis_results")

# Collect all results
all_results = {}

for dataset in ["LRW-AR", "LipBengal"]:
    all_results[dataset] = {}
    
    for json_file in output_dir.glob(f"{dataset}_*_topk_results_*.json"):
        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            # Extract experiment name
            exp_name = json_file.stem.replace(f'{dataset}_', '').replace('_topk_results_', '')
            all_results[dataset][exp_name] = data['accuracies']
        except Exception as e:
            print(f"Warning: Could not load {json_file}: {e}")

# Create summary report
report_path = output_dir / "TOPK_SUMMARY_REPORT.txt"

with open(report_path, 'w') as f:
    f.write("="*80 + "\n")
    f.write("                    TOP-K ANALYSIS SUMMARY REPORT\n")
    f.write("="*80 + "\n\n")
    
    for dataset in ["LRW-AR", "LipBengal"]:
        if dataset not in all_results or not all_results[dataset]:
            continue
        
        f.write(f"\n{'='*80}\n")
        f.write(f"   {dataset}\n")
        f.write(f"{'='*80}\n\n")
        
        for exp, accs in sorted(all_results[dataset].items()):
            f.write(f"{exp}:\n")
            for k, acc in sorted(accs.items(), key=lambda x: int(x[0].replace('top', ''))):
                k_num = k.replace('top', '')
                f.write(f"  {k:>6s}: {acc:6.2f}%")
                if k != 'top1':
                    improvement = acc - accs['top1']
                    f.write(f"  (+{improvement:5.2f}%)")
                f.write("\n")
            f.write("\n")

print(f"\n✓ Summary report saved to: {report_path}")

# Create comparison plot
if all_results:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    for idx, dataset in enumerate(["LRW-AR", "LipBengal"]):
        if dataset not in all_results or not all_results[dataset]:
            continue
        
        ax = axes[idx]
        
        # Get best 5 experiments
        best_exps = sorted(all_results[dataset].items(), 
                          key=lambda x: x[1].get('top1', 0), reverse=True)[:5]
        
        colors = plt.cm.tab10(np.linspace(0, 1, 10))
        
        for color_idx, (exp, accs) in enumerate(best_exps):
            k_vals = sorted([int(k.replace('top', '')) for k in accs.keys()])
            acc_vals = [accs[f'top{k}'] for k in k_vals]
            
            ax.plot(k_vals, acc_vals, 'o-', label=exp, linewidth=2, 
                   markersize=6, color=colors[color_idx])
        
        ax.set_xlabel('K (Top-K)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
        ax.set_title(f'{dataset} - Top Experiments', fontsize=13, fontweight='bold')
        ax.legend(fontsize=9, loc='lower right')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 100])
    
    plt.tight_layout()
    plot_path = output_dir / "topk_comparison_all.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✓ Comparison plot saved to: {plot_path}")

EOF

echo "✓ Summary report generated"
echo ""



