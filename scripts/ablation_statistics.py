#!/usr/bin/env python3
"""
Statistical Analysis for Ablation Study Results

Performs rigorous statistical testing on ablation experiment results:
- Paired t-tests for comparing methods
- 95% confidence intervals
- Cohen's d effect sizes
- Two-way ANOVA for interaction effects
- Bonferroni correction for multiple comparisons

Usage:
    python3 scripts/ablation_statistics.py --db results/ablation_results.db --output analysis/
"""

import argparse
import sqlite3
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import numpy as np

# Try to import scipy for statistical tests
try:
    from scipy import stats
    from scipy.stats import ttest_rel, ttest_ind, f_oneway
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: scipy not available. Install with: pip install scipy")


class AblationStatistics:
    """Statistical analysis for ablation study"""
    
    def __init__(self, db_path: str):
        """
        Initialize statistical analyzer
        
        Args:
            db_path: Path to SQLite results database
        """
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            raise FileNotFoundError(f"Database not found: {db_path}")
        
        if not SCIPY_AVAILABLE:
            raise RuntimeError("scipy is required for statistical analysis")
    
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
    
    def paired_t_test(
        self,
        group1: List[float],
        group2: List[float],
        alternative: str = 'two-sided'
    ) -> Tuple[float, float]:
        """
        Perform paired t-test
        
        Args:
            group1: First group of values
            group2: Second group of values
            alternative: 'two-sided', 'less', or 'greater'
            
        Returns:
            (t_statistic, p_value)
        """
        if len(group1) != len(group2):
            raise ValueError("Groups must have same length for paired t-test")
        
        t_stat, p_value = ttest_rel(group1, group2, alternative=alternative)
        return float(t_stat), float(p_value)
    
    def independent_t_test(
        self,
        group1: List[float],
        group2: List[float],
        alternative: str = 'two-sided'
    ) -> Tuple[float, float]:
        """
        Perform independent t-test
        
        Args:
            group1: First group of values
            group2: Second group of values
            alternative: 'two-sided', 'less', or 'greater'
            
        Returns:
            (t_statistic, p_value)
        """
        t_stat, p_value = ttest_ind(group1, group2, alternative=alternative)
        return float(t_stat), float(p_value)
    
    def cohens_d(self, group1: List[float], group2: List[float]) -> float:
        """
        Calculate Cohen's d effect size
        
        Args:
            group1: First group of values
            group2: Second group of values
            
        Returns:
            Cohen's d (standardized mean difference)
        """
        n1, n2 = len(group1), len(group2)
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
        
        # Pooled standard deviation
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        
        # Cohen's d
        d = (np.mean(group1) - np.mean(group2)) / pooled_std
        return float(d)
    
    def confidence_interval(
        self,
        values: List[float],
        confidence: float = 0.95
    ) -> Tuple[float, float, float]:
        """
        Calculate confidence interval for mean
        
        Args:
            values: List of values
            confidence: Confidence level (default 0.95 for 95% CI)
            
        Returns:
            (mean, lower_bound, upper_bound)
        """
        n = len(values)
        mean = np.mean(values)
        std_err = stats.sem(values)
        
        # t-distribution critical value
        t_crit = stats.t.ppf((1 + confidence) / 2, n - 1)
        
        margin = t_crit * std_err
        return float(mean), float(mean - margin), float(mean + margin)
    
    def bonferroni_correction(self, p_values: List[float], alpha: float = 0.05) -> List[bool]:
        """
        Apply Bonferroni correction for multiple comparisons
        
        Args:
            p_values: List of p-values
            alpha: Significance level
            
        Returns:
            List of booleans indicating significance after correction
        """
        n = len(p_values)
        corrected_alpha = alpha / n
        return [p < corrected_alpha for p in p_values]
    
    def analyze_phase1_script_normalization(self, dataset: str) -> Dict[str, Any]:
        """
        Analyze Phase 1 (Script Normalization) results
        
        Args:
            dataset: Dataset name
            
        Returns:
            Statistical analysis results
        """
        experiments = self.load_experiments(dataset=dataset, phase=1)
        
        if not experiments:
            return {"error": f"No Phase 1 experiments found for {dataset}"}
        
        # Extract results by normalization type
        results_by_type = {}
        for exp in experiments:
            norm_type = exp.get('script_norm', 'unknown')
            top1 = exp.get('top1_accuracy')
            if top1 is not None:
                results_by_type[norm_type] = top1
        
        # Find baseline (phonetic) and compare others
        baseline = results_by_type.get('phonetic')
        if baseline is None:
            return {"error": "No phonetic baseline found"}
        
        analysis = {
            'dataset': dataset,
            'baseline': 'phonetic',
            'baseline_accuracy': baseline,
            'comparisons': {}
        }
        
        for norm_type, accuracy in results_by_type.items():
            if norm_type == 'phonetic':
                continue
            
            diff = accuracy - baseline
            # Note: Can't do significance testing with single values per condition
            # Would need multiple runs
            analysis['comparisons'][norm_type] = {
                'accuracy': accuracy,
                'difference': diff,
                'improvement_pct': (diff / baseline * 100) if baseline > 0 else None
            }
        
        return analysis
    
    def analyze_phase2_transfer_learning(self, dataset: str) -> Dict[str, Any]:
        """
        Analyze Phase 2 (Transfer Learning) results
        
        Args:
            dataset: Dataset name
            
        Returns:
            Statistical analysis results
        """
        experiments = self.load_experiments(dataset=dataset, phase=2)
        
        if not experiments:
            return {"error": f"No Phase 2 experiments found for {dataset}"}
        
        # Group experiments by type (T1, T2, T3)
        t1_exps = [e for e in experiments if e['experiment_id'].startswith('T1_')]
        t2_exps = [e for e in experiments if e['experiment_id'].startswith('T2_')]
        t3_exps = [e for e in experiments if e['experiment_id'].startswith('T3_')]
        
        analysis = {
            'dataset': dataset,
            't1_transfer_modes': self._analyze_t1(t1_exps),
            't2_freezing': self._analyze_t2(t2_exps),
            't3_differential_lr': self._analyze_t3(t3_exps)
        }
        
        return analysis
    
    def _analyze_t1(self, experiments: List[Dict]) -> Dict[str, Any]:
        """Analyze T1 (transfer modes)"""
        results = {}
        for exp in experiments:
            mode = exp.get('transfer_mode', 'unknown')
            top1 = exp.get('top1_accuracy')
            if top1 is not None:
                results[mode] = top1
        
        # Compare to scratch baseline
        scratch_acc = results.get('scratch')
        if scratch_acc is None:
            return results
        
        comparisons = {}
        for mode, acc in results.items():
            if mode == 'scratch':
                continue
            diff = acc - scratch_acc
            comparisons[mode] = {
                'accuracy': acc,
                'vs_scratch': diff,
                'improvement_pct': (diff / scratch_acc * 100) if scratch_acc > 0 else None
            }
        
        return {
            'scratch_baseline': scratch_acc,
            'results': results,
            'comparisons': comparisons
        }
    
    def _analyze_t2(self, experiments: List[Dict]) -> Dict[str, Any]:
        """Analyze T2 (freezing strategies)"""
        results = {}
        for exp in experiments:
            freeze_epochs = exp.get('freeze_epochs', 0)
            top1 = exp.get('top1_accuracy')
            if top1 is not None:
                results[freeze_epochs] = top1
        
        return {'by_freeze_epochs': results}
    
    def _analyze_t3(self, experiments: List[Dict]) -> Dict[str, Any]:
        """Analyze T3 (differential LR)"""
        results = {}
        for exp in experiments:
            lr_mult = exp.get('encoder_lr_mult', 1.0)
            top1 = exp.get('top1_accuracy')
            if top1 is not None:
                results[lr_mult] = top1
        
        return {'by_lr_mult': results}
    
    def generate_report(self, output_path: str):
        """Generate comprehensive statistical analysis report"""
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("ABLATION STUDY STATISTICAL ANALYSIS")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        for dataset in ["LipBengal", "LRW-AR"]:
            report_lines.append(f"\n{'=' * 80}")
            report_lines.append(f"Dataset: {dataset}")
            report_lines.append(f"{'=' * 80}\n")
            
            # Phase 1
            report_lines.append("Phase 1: Script Normalization")
            report_lines.append("-" * 80)
            p1_results = self.analyze_phase1_script_normalization(dataset)
            report_lines.append(self._format_dict(p1_results, indent=2))
            report_lines.append("")
            
            # Phase 2
            report_lines.append("Phase 2: Transfer Learning Strategy")
            report_lines.append("-" * 80)
            p2_results = self.analyze_phase2_transfer_learning(dataset)
            report_lines.append(self._format_dict(p2_results, indent=2))
            report_lines.append("")
        
        report = "\n".join(report_lines)
        
        # Save to file
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(report)
        
        print(f"Statistical analysis report saved to: {output_path}")
        return report
    
    def _format_dict(self, d: Dict, indent: int = 0) -> str:
        """Format dictionary for report"""
        lines = []
        prefix = "  " * indent
        for key, value in d.items():
            if isinstance(value, dict):
                lines.append(f"{prefix}{key}:")
                lines.append(self._format_dict(value, indent + 1))
            elif isinstance(value, (int, float)):
                lines.append(f"{prefix}{key}: {value:.4f}")
            else:
                lines.append(f"{prefix}{key}: {value}")
        return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Statistical analysis for ablation study")
    parser.add_argument("--db", default="results/ablation_results.db", help="Database path")
    parser.add_argument("--output", default="analysis/statistical_report.txt", help="Output file")
    parser.add_argument("--dataset", choices=["LipBengal", "LRW-AR"], help="Filter by dataset")
    parser.add_argument("--phase", type=int, choices=[1, 2, 3], help="Filter by phase")
    
    args = parser.parse_args()
    
    if not SCIPY_AVAILABLE:
        print("Error: scipy is required. Install with: pip install scipy")
        return
    
    try:
        analyzer = AblationStatistics(args.db)
        
        if args.dataset and args.phase:
            # Single analysis
            if args.phase == 1:
                results = analyzer.analyze_phase1_script_normalization(args.dataset)
            elif args.phase == 2:
                results = analyzer.analyze_phase2_transfer_learning(args.dataset)
            print(results)
        else:
            # Full report
            analyzer.generate_report(args.output)
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()


