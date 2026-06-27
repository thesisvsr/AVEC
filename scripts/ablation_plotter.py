#!/usr/bin/env python3
"""
Automated Plotting for Ablation Study Results

Generates performance comparison plots, heatmaps, and statistical visualizations
from tracked experiment results.

Usage:
    python3 scripts/ablation_plotter.py --db results/ablation_results.db --output plots/
"""

import argparse
import sqlite3
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np

# Try to import plotting libraries
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    print("Warning: matplotlib/seaborn not available. Install with: pip install matplotlib seaborn")


class AblationPlotter:
    """Generate plots for ablation study results"""
    
    def __init__(self, db_path: str):
        """
        Initialize plotter
        
        Args:
            db_path: Path to SQLite database
        """
        self.db_path = Path(db_path)
        
        if not PLOTTING_AVAILABLE:
            raise RuntimeError("Plotting libraries not available. Install matplotlib and seaborn.")
        
        # Set style
        sns.set_style("whitegrid")
        sns.set_palette("husl")
    
    def load_experiments(self, dataset: Optional[str] = None, phase: Optional[int] = None) -> List[Dict[str, Any]]:
        """Load experiments from database"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        query = 'SELECT * FROM experiments WHERE 1=1'
        params = []
        
        if dataset:
            query += ' AND dataset = ?'
            params.append(dataset)
        if phase is not None:
            query += ' AND phase = ?'
            params.append(phase)
        
        query += ' ORDER BY phase, experiment_id'
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()
        
        return [dict(row) for row in rows]
    
    def plot_phase1_comparison(self, dataset: str, output_path: str):
        """Plot Phase 1 (Script Normalization) comparison"""
        experiments = self.load_experiments(dataset=dataset, phase=1)
        
        if not experiments:
            print(f"No Phase 1 experiments found for {dataset}")
            return
        
        # Extract data
        exp_ids = [exp['experiment_id'] for exp in experiments]
        top1 = [exp.get('top1_accuracy', 0) for exp in experiments]
        wer = [exp.get('wer', 0) for exp in experiments]
        cer = [exp.get('cer', 0) for exp in experiments]
        
        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Top-1 Accuracy
        axes[0].bar(range(len(exp_ids)), top1, color='steelblue')
        axes[0].set_xticks(range(len(exp_ids)))
        axes[0].set_xticklabels([eid.replace('S1_', '') for eid in exp_ids], rotation=45, ha='right')
        axes[0].set_ylabel('Top-1 Accuracy (%)')
        axes[0].set_title(f'Phase 1: Script Normalization\n{dataset} - Top-1 Accuracy')
        axes[0].grid(axis='y', alpha=0.3)
        
        # WER
        axes[1].bar(range(len(exp_ids)), wer, color='coral')
        axes[1].set_xticks(range(len(exp_ids)))
        axes[1].set_xticklabels([eid.replace('S1_', '') for eid in exp_ids], rotation=45, ha='right')
        axes[1].set_ylabel('Word Error Rate (%)')
        axes[1].set_title(f'Phase 1: Script Normalization\n{dataset} - WER')
        axes[1].grid(axis='y', alpha=0.3)
        
        # CER
        axes[2].bar(range(len(exp_ids)), cer, color='mediumseagreen')
        axes[2].set_xticks(range(len(exp_ids)))
        axes[2].set_xticklabels([eid.replace('S1_', '') for eid in exp_ids], rotation=45, ha='right')
        axes[2].set_ylabel('Character Error Rate (%)')
        axes[2].set_title(f'Phase 1: Script Normalization\n{dataset} - CER')
        axes[2].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved Phase 1 plot: {output_path}")
    
    def plot_phase2_comparison(self, dataset: str, output_path: str):
        """Plot Phase 2 (Transfer Learning) comparison"""
        experiments = self.load_experiments(dataset=dataset, phase=2)
        
        if not experiments:
            print(f"No Phase 2 experiments found for {dataset}")
            return
        
        # Group by experiment type (T1, T2, T3)
        t1_exps = [e for e in experiments if e['experiment_id'].startswith('T1_')]
        t2_exps = [e for e in experiments if e['experiment_id'].startswith('T2_')]
        t3_exps = [e for e in experiments if e['experiment_id'].startswith('T3_')]
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # T1: Transfer modes
        if t1_exps:
            exp_ids = [exp['experiment_id'].replace('T1_', '') for exp in t1_exps]
            top1 = [exp.get('top1_accuracy', 0) for exp in t1_exps]
            axes[0].bar(range(len(exp_ids)), top1, color='steelblue')
            axes[0].set_xticks(range(len(exp_ids)))
            axes[0].set_xticklabels(exp_ids, rotation=45, ha='right')
            axes[0].set_ylabel('Top-1 Accuracy (%)')
            axes[0].set_title(f'T1: Transfer Learning Modes\n{dataset}')
            axes[0].grid(axis='y', alpha=0.3)
        
        # T2: Freezing strategies
        if t2_exps:
            exp_ids = [exp['experiment_id'].replace('T2_', '').replace('freeze_', '').replace('ep', '') for exp in t2_exps]
            top1 = [exp.get('top1_accuracy', 0) for exp in t2_exps]
            axes[1].bar(range(len(exp_ids)), top1, color='coral')
            axes[1].set_xticks(range(len(exp_ids)))
            axes[1].set_xticklabels(exp_ids, rotation=45, ha='right')
            axes[1].set_ylabel('Top-1 Accuracy (%)')
            axes[1].set_title(f'T2: Freezing Strategies\n{dataset}')
            axes[1].grid(axis='y', alpha=0.3)
        
        # T3: Differential LR
        if t3_exps:
            exp_ids = [exp['experiment_id'].replace('T3_lr_', '').replace('_', '.') for exp in t3_exps]
            top1 = [exp.get('top1_accuracy', 0) for exp in t3_exps]
            axes[2].bar(range(len(exp_ids)), top1, color='mediumseagreen')
            axes[2].set_xticks(range(len(exp_ids)))
            axes[2].set_xticklabels(exp_ids, rotation=45, ha='right')
            axes[2].set_ylabel('Top-1 Accuracy (%)')
            axes[2].set_title(f'T3: Encoder LR Multiplier\n{dataset}')
            axes[2].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved Phase 2 plot: {output_path}")
    
    def plot_phase3_comparison(self, dataset: str, output_path: str):
        """Plot Phase 3 (Dataset Analysis) comparison"""
        experiments = self.load_experiments(dataset=dataset, phase=3)
        
        if not experiments:
            print(f"No Phase 3 experiments found for {dataset}")
            return
        
        # D1: Data fraction experiments
        d1_exps = [e for e in experiments if e['experiment_id'].startswith('D1_')]
        
        if not d1_exps:
            print(f"No D1 experiments found for {dataset}")
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Extract data fractions and accuracies
        fractions = [exp.get('data_fraction', 1.0) * 100 for exp in d1_exps]
        top1 = [exp.get('top1_accuracy', 0) for exp in d1_exps]
        
        # Sort by fraction
        sorted_data = sorted(zip(fractions, top1))
        fractions, top1 = zip(*sorted_data) if sorted_data else ([], [])
        
        ax.plot(fractions, top1, marker='o', linewidth=2, markersize=8, color='steelblue')
        ax.set_xlabel('Target Dataset Fraction (%)')
        ax.set_ylabel('Top-1 Accuracy (%)')
        ax.set_title(f'Phase 3: Dataset Size Impact\n{dataset}')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved Phase 3 plot: {output_path}")
    
    def plot_all_phases_summary(self, output_path: str):
        """Create comprehensive summary plot across all phases"""
        experiments = self.load_experiments()
        
        if not experiments:
            print("No experiments found")
            return
        
        # Group by dataset and phase
        datasets = sorted(set(exp['dataset'] for exp in experiments))
        phases = sorted(set(exp.get('phase', 0) for exp in experiments))
        
        fig, axes = plt.subplots(len(phases), len(datasets), figsize=(6*len(datasets), 5*len(phases)))
        
        if len(datasets) == 1 and len(phases) == 1:
            axes = np.array([[axes]])
        elif len(datasets) == 1:
            axes = axes.reshape(-1, 1)
        elif len(phases) == 1:
            axes = axes.reshape(1, -1)
        
        for i, phase in enumerate(phases):
            for j, dataset in enumerate(datasets):
                phase_dataset_exps = [e for e in experiments 
                                     if e.get('phase') == phase and e['dataset'] == dataset]
                
                if not phase_dataset_exps:
                    axes[i, j].text(0.5, 0.5, 'No Data', ha='center', va='center')
                    axes[i, j].set_title(f'Phase {phase} - {dataset}')
                    continue
                
                exp_ids = [exp['experiment_id'] for exp in phase_dataset_exps]
                top1 = [exp.get('top1_accuracy', 0) for exp in phase_dataset_exps]
                
                axes[i, j].bar(range(len(exp_ids)), top1, color='steelblue')
                axes[i, j].set_xticks(range(len(exp_ids)))
                axes[i, j].set_xticklabels([eid[:15] for eid in exp_ids], rotation=45, ha='right', fontsize=8)
                axes[i, j].set_ylabel('Top-1 Accuracy (%)')
                axes[i, j].set_title(f'Phase {phase} - {dataset}')
                axes[i, j].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved summary plot: {output_path}")
    
    def plot_comparison_heatmap(self, output_path: str):
        """Create heatmap comparing all experiments"""
        experiments = self.load_experiments()
        
        if not experiments:
            print("No experiments found")
            return
        
        # Create matrix of results
        exp_ids = [exp['experiment_id'] for exp in experiments]
        metrics = ['top1_accuracy', 'top10_accuracy', 'wer', 'cer']
        
        data = []
        for metric in metrics:
            row = [exp.get(metric, np.nan) for exp in experiments]
            data.append(row)
        
        data = np.array(data)
        
        fig, ax = plt.subplots(figsize=(max(12, len(exp_ids) * 0.5), 6))
        
        sns.heatmap(data, annot=True, fmt='.1f', cmap='RdYlGn', 
                   xticklabels=[eid[:20] for eid in exp_ids],
                   yticklabels=metrics, ax=ax, cbar_kws={'label': 'Value'})
        
        ax.set_title('Ablation Study Results Heatmap')
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved heatmap: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate plots for ablation study")
    parser.add_argument("--db", default="results/ablation_results.db", help="Database path")
    parser.add_argument("--output", default="plots/", help="Output directory for plots")
    parser.add_argument("--dataset", choices=["LipBengal", "LRW-AR"], help="Filter by dataset")
    parser.add_argument("--phase", type=int, choices=[1, 2, 3], help="Filter by phase")
    parser.add_argument("--summary", action="store_true", help="Generate summary plot")
    parser.add_argument("--heatmap", action="store_true", help="Generate heatmap")
    
    args = parser.parse_args()
    
    if not PLOTTING_AVAILABLE:
        print("Error: Plotting libraries not available")
        print("Install with: pip install matplotlib seaborn")
        return
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        plotter = AblationPlotter(args.db)
        
        if args.summary:
            plotter.plot_all_phases_summary(str(output_dir / "summary.png"))
        elif args.heatmap:
            plotter.plot_comparison_heatmap(str(output_dir / "heatmap.png"))
        elif args.phase:
            if args.dataset:
                if args.phase == 1:
                    plotter.plot_phase1_comparison(args.dataset, str(output_dir / f"phase1_{args.dataset}.png"))
                elif args.phase == 2:
                    plotter.plot_phase2_comparison(args.dataset, str(output_dir / f"phase2_{args.dataset}.png"))
                elif args.phase == 3:
                    plotter.plot_phase3_comparison(args.dataset, str(output_dir / f"phase3_{args.dataset}.png"))
            else:
                print("Error: --phase requires --dataset")
        else:
            # Generate all plots
            for dataset in ["LipBengal", "LRW-AR"]:
                plotter.plot_phase1_comparison(dataset, str(output_dir / f"phase1_{dataset}.png"))
                plotter.plot_phase2_comparison(dataset, str(output_dir / f"phase2_{dataset}.png"))
                plotter.plot_phase3_comparison(dataset, str(output_dir / f"phase3_{dataset}.png"))
            
            plotter.plot_all_phases_summary(str(output_dir / "summary.png"))
            plotter.plot_comparison_heatmap(str(output_dir / "heatmap.png"))
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()


