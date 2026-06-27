#!/usr/bin/env python3
"""
Experiment Tracking System for Ablation Study

Provides tools to track, aggregate, and analyze results from ablation experiments.
Includes CSV database, TensorBoard log parsing, and automated plotting.

Usage:
    # Track a single experiment result
    python3 scripts/ablation_tracker.py --track --exp-id S1_raw_script --dataset LipBengal \\
        --top1 1.5 --top10 5.2 --wer 98.5 --cer 97.8
    
    # Aggregate results from TensorBoard logs
    python3 scripts/ablation_tracker.py --aggregate --callback-dir callbacks/LipBengal/AV/ablations
    
    # Generate plots
    python3 scripts/ablation_tracker.py --plot --output plots/ablation_results.png
"""

import os
import csv
import json
import argparse
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
import re


class AblationTracker:
    """Track and manage ablation experiment results"""
    
    def __init__(self, db_path: str = "ablation_results.db", csv_path: str = "ablation_results.csv"):
        """
        Initialize the tracking system
        
        Args:
            db_path: Path to SQLite database
            csv_path: Path to CSV results file
        """
        self.db_path = Path(db_path)
        self.csv_path = Path(csv_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database with schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS experiments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id TEXT NOT NULL,
                dataset TEXT NOT NULL,
                phase INTEGER,
                script_norm TEXT,
                transfer_mode TEXT,
                freeze_epochs INTEGER,
                encoder_lr_mult REAL,
                head_lr_mult REAL,
                data_fraction REAL,
                top1_accuracy REAL,
                top10_accuracy REAL,
                wer REAL,
                cer REAL,
                epochs_trained INTEGER,
                training_time_hours REAL,
                gpu_hours REAL,
                checkpoint_path TEXT,
                callback_path TEXT,
                config_path TEXT,
                notes TEXT,
                timestamp TEXT,
                UNIQUE(experiment_id, dataset)
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS training_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id TEXT NOT NULL,
                dataset TEXT NOT NULL,
                epoch INTEGER,
                train_loss REAL,
                val_loss REAL,
                val_top1 REAL,
                val_top10 REAL,
                val_wer REAL,
                val_cer REAL,
                learning_rate REAL,
                timestamp TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def add_experiment(
        self,
        experiment_id: str,
        dataset: str,
        phase: int,
        script_norm: str = "phonetic",
        transfer_mode: str = "full",
        freeze_epochs: int = 5,
        encoder_lr_mult: float = 0.2,
        head_lr_mult: float = 1.0,
        data_fraction: float = 1.0,
        top1_accuracy: Optional[float] = None,
        top10_accuracy: Optional[float] = None,
        wer: Optional[float] = None,
        cer: Optional[float] = None,
        epochs_trained: Optional[int] = None,
        training_time_hours: Optional[float] = None,
        gpu_hours: Optional[float] = None,
        checkpoint_path: Optional[str] = None,
        callback_path: Optional[str] = None,
        config_path: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> int:
        """Add or update an experiment record"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        timestamp = datetime.now().isoformat()
        
        cursor.execute('''
            INSERT OR REPLACE INTO experiments (
                experiment_id, dataset, phase, script_norm, transfer_mode,
                freeze_epochs, encoder_lr_mult, head_lr_mult, data_fraction,
                top1_accuracy, top10_accuracy, wer, cer,
                epochs_trained, training_time_hours, gpu_hours,
                checkpoint_path, callback_path, config_path, notes, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            experiment_id, dataset, phase, script_norm, transfer_mode,
            freeze_epochs, encoder_lr_mult, head_lr_mult, data_fraction,
            top1_accuracy, top10_accuracy, wer, cer,
            epochs_trained, training_time_hours, gpu_hours,
            checkpoint_path, callback_path, config_path, notes, timestamp
        ))
        
        conn.commit()
        exp_id = cursor.lastrowid
        conn.close()
        
        # Also update CSV
        self._update_csv()
        
        return exp_id
    
    def add_training_history(
        self,
        experiment_id: str,
        dataset: str,
        epoch: int,
        train_loss: Optional[float] = None,
        val_loss: Optional[float] = None,
        val_top1: Optional[float] = None,
        val_top10: Optional[float] = None,
        val_wer: Optional[float] = None,
        val_cer: Optional[float] = None,
        learning_rate: Optional[float] = None,
    ):
        """Add training history for an experiment"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        timestamp = datetime.now().isoformat()
        
        cursor.execute('''
            INSERT INTO training_history (
                experiment_id, dataset, epoch,
                train_loss, val_loss, val_top1, val_top10, val_wer, val_cer,
                learning_rate, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            experiment_id, dataset, epoch,
            train_loss, val_loss, val_top1, val_top10, val_wer, val_cer,
            learning_rate, timestamp
        ))
        
        conn.commit()
        conn.close()
    
    def get_experiment(self, experiment_id: str, dataset: str) -> Optional[Dict[str, Any]]:
        """Get experiment record"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM experiments
            WHERE experiment_id = ? AND dataset = ?
        ''', (experiment_id, dataset))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return dict(row)
        return None
    
    def get_all_experiments(self, dataset: Optional[str] = None, phase: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get all experiment records, optionally filtered"""
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
    
    def get_training_history(self, experiment_id: str, dataset: str) -> List[Dict[str, Any]]:
        """Get training history for an experiment"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM training_history
            WHERE experiment_id = ? AND dataset = ?
            ORDER BY epoch
        ''', (experiment_id, dataset))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [dict(row) for row in rows]
    
    def _update_csv(self):
        """Update CSV file from database"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM experiments ORDER BY phase, experiment_id')
        rows = cursor.fetchall()
        conn.close()
        
        if not rows:
            return
        
        with open(self.csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            for row in rows:
                writer.writerow(dict(row))
        
        print(f"Updated CSV: {self.csv_path}")
    
    def parse_tensorboard_logs(self, callback_dir: str) -> List[Dict[str, Any]]:
        """
        Parse TensorBoard event files from callback directory
        
        Args:
            callback_dir: Directory containing TensorBoard logs
            
        Returns:
            List of parsed experiment results
        """
        callback_path = Path(callback_dir)
        if not callback_path.exists():
            print(f"Callback directory not found: {callback_dir}")
            return []
        
        results = []
        
        # Try to import tensorboard for log parsing
        try:
            from tensorboard.backend.event_processing import event_accumulator
            
            # Find all event files
            log_dirs = []
            for root, dirs, files in os.walk(callback_path):
                if 'events.out.tfevents' in ' '.join(files):
                    log_dirs.append(root)
            
            for log_dir in log_dirs:
                try:
                    # Parse experiment ID from path
                    exp_id = self._extract_experiment_id_from_path(log_dir)
                    if not exp_id:
                        continue
                    
                    # Load TensorBoard logs
                    ea = event_accumulator.EventAccumulator(log_dir)
                    ea.Reload()
                    
                    # Extract metrics
                    result = {
                        'experiment_id': exp_id,
                        'log_dir': log_dir,
                    }
                    
                    # Try to get evaluation metrics
                    for tag in ea.Tags()['scalars']:
                        if 'Evaluation' in tag or 'eval' in tag.lower():
                            events = ea.Scalars(tag)
                            if events:
                                last_event = events[-1]
                                metric_name = tag.split('/')[-1]
                                result[metric_name] = last_event.value
                    
                    results.append(result)
                    
                except Exception as e:
                    print(f"Error parsing {log_dir}: {e}")
                    continue
        
        except ImportError:
            print("TensorBoard not installed. Using simple log file parsing instead.")
            results = self._parse_logs_simple(callback_path)
        
        return results
    
    def _parse_logs_simple(self, callback_path: Path) -> List[Dict[str, Any]]:
        """Simple log file parsing when TensorBoard is not available"""
        results = []
        
        # Look for experiment subdirectories
        for exp_dir in callback_path.iterdir():
            if not exp_dir.is_dir():
                continue
            
            exp_id = exp_dir.name
            result = {
                'experiment_id': exp_id,
                'callback_path': str(exp_dir),
            }
            
            # Try to find checkpoints and extract metrics from filenames or logs
            # This is a placeholder - actual implementation would parse log files
            
            results.append(result)
        
        return results
    
    def _extract_experiment_id_from_path(self, path: str) -> Optional[str]:
        """Extract experiment ID from callback path"""
        # Try to match patterns like S1_raw_script, T1_scratch, etc.
        match = re.search(r'([STD]\d+_[a-z0-9_]+)', path)
        if match:
            return match.group(1)
        return None
    
    def generate_summary_report(self, output_path: Optional[str] = None) -> str:
        """Generate a summary report of all experiments"""
        experiments = self.get_all_experiments()
        
        if not experiments:
            return "No experiments tracked yet."
        
        # Group by phase
        by_phase = {}
        for exp in experiments:
            phase = exp.get('phase', 0)
            if phase not in by_phase:
                by_phase[phase] = []
            by_phase[phase].append(exp)
        
        # Generate report
        lines = []
        lines.append("=" * 80)
        lines.append("ABLATION STUDY SUMMARY REPORT")
        lines.append("=" * 80)
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"Total experiments: {len(experiments)}")
        lines.append("")
        
        for phase in sorted(by_phase.keys()):
            phase_name = {1: "Script Normalization", 2: "Transfer Learning", 3: "Dataset Analysis"}.get(phase, f"Phase {phase}")
            lines.append(f"\n{'=' * 80}")
            lines.append(f"Phase {phase}: {phase_name}")
            lines.append(f"{'=' * 80}")
            
            phase_exps = by_phase[phase]
            
            # Group by dataset
            by_dataset = {}
            for exp in phase_exps:
                dataset = exp.get('dataset', 'Unknown')
                if dataset not in by_dataset:
                    by_dataset[dataset] = []
                by_dataset[dataset].append(exp)
            
            for dataset in sorted(by_dataset.keys()):
                lines.append(f"\n{dataset}:")
                lines.append("-" * 80)
                lines.append(f"{'Experiment':<25} {'Top-1':<8} {'Top-10':<8} {'WER':<8} {'CER':<8} {'Epochs':<8}")
                lines.append("-" * 80)
                
                for exp in by_dataset[dataset]:
                    exp_id = exp.get('experiment_id', 'Unknown')
                    top1 = exp.get('top1_accuracy')
                    top10 = exp.get('top10_accuracy')
                    wer = exp.get('wer')
                    cer = exp.get('cer')
                    epochs = exp.get('epochs_trained')
                    
                    top1_str = f"{top1:.2f}%" if top1 is not None else "N/A"
                    top10_str = f"{top10:.2f}%" if top10 is not None else "N/A"
                    wer_str = f"{wer:.2f}%" if wer is not None else "N/A"
                    cer_str = f"{cer:.2f}%" if cer is not None else "N/A"
                    epochs_str = str(epochs) if epochs is not None else "N/A"
                    
                    lines.append(f"{exp_id:<25} {top1_str:<8} {top10_str:<8} {wer_str:<8} {cer_str:<8} {epochs_str:<8}")
        
        lines.append("\n" + "=" * 80)
        
        report = "\n".join(lines)
        
        if output_path:
            Path(output_path).write_text(report)
            print(f"Report saved to: {output_path}")
        
        return report


def main():
    parser = argparse.ArgumentParser(description="Track and analyze ablation study experiments")
    parser.add_argument("--db", default="results/ablation_results.db", help="Database path")
    parser.add_argument("--csv", default="results/ablation_results.csv", help="CSV output path")
    
    # Tracking commands
    parser.add_argument("--track", action="store_true", help="Track a new experiment")
    parser.add_argument("--exp-id", type=str, help="Experiment ID")
    parser.add_argument("--dataset", type=str, choices=["LipBengal", "LRW-AR"], help="Dataset")
    parser.add_argument("--phase", type=int, choices=[1, 2, 3], help="Phase number")
    parser.add_argument("--top1", type=float, help="Top-1 accuracy")
    parser.add_argument("--top10", type=float, help="Top-10 accuracy")
    parser.add_argument("--wer", type=float, help="Word Error Rate")
    parser.add_argument("--cer", type=float, help="Character Error Rate")
    parser.add_argument("--epochs", type=int, help="Number of epochs trained")
    
    # Aggregation commands
    parser.add_argument("--aggregate", action="store_true", help="Aggregate from TensorBoard logs")
    parser.add_argument("--callback-dir", type=str, help="Callback directory to parse")
    
    # Reporting commands
    parser.add_argument("--report", action="store_true", help="Generate summary report")
    parser.add_argument("--output", type=str, help="Output file path")
    parser.add_argument("--list", action="store_true", help="List all experiments")
    
    args = parser.parse_args()
    
    tracker = AblationTracker(db_path=args.db, csv_path=args.csv)
    
    if args.track:
        if not args.exp_id or not args.dataset or args.phase is None:
            print("Error: --track requires --exp-id, --dataset, and --phase")
            return
        
        tracker.add_experiment(
            experiment_id=args.exp_id,
            dataset=args.dataset,
            phase=args.phase,
            top1_accuracy=args.top1,
            top10_accuracy=args.top10,
            wer=args.wer,
            cer=args.cer,
            epochs_trained=args.epochs,
        )
        print(f"Tracked experiment: {args.exp_id} on {args.dataset}")
    
    elif args.aggregate:
        if not args.callback_dir:
            print("Error: --aggregate requires --callback-dir")
            return
        
        results = tracker.parse_tensorboard_logs(args.callback_dir)
        print(f"Found {len(results)} experiments in {args.callback_dir}")
        for result in results:
            print(f"  - {result}")
    
    elif args.report:
        report = tracker.generate_summary_report(args.output)
        if not args.output:
            print(report)
    
    elif args.list:
        experiments = tracker.get_all_experiments()
        print(f"Total experiments: {len(experiments)}")
        for exp in experiments:
            print(f"  {exp['experiment_id']} ({exp['dataset']}) - Phase {exp['phase']}")
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()


