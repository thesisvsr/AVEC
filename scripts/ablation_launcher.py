#!/usr/bin/env python3
"""
Experiment Launcher for Ablation Study

Orchestrates the execution of ablation experiments, including config generation,
training launch, and result tracking.

Usage:
    # Launch single experiment
    python3 scripts/ablation_launcher.py --exp-id S1_raw_script --dataset LipBengal --gpus 0
    
    # Launch batch of experiments
    python3 scripts/ablation_launcher.py --phase 1 --dataset LipBengal --gpus 0,1
    
    # Dry run (generate configs only)
    python3 scripts/ablation_launcher.py --phase 1 --dry-run
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
from typing import Optional, List, Dict, Any
import time
import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.ablation_env import AblationEnv, get_experiment_matrix
from configs.ablations.template_generator import AblationConfigGenerator


class AblationLauncher:
    """Launch and manage ablation experiments"""
    
    def __init__(
        self,
        project_root: str = ".",
        gpus: Optional[List[int]] = None,
        dry_run: bool = False,
        auto_track: bool = True
    ):
        """
        Initialize launcher
        
        Args:
            project_root: Project root directory
            gpus: List of GPU IDs to use
            dry_run: If True, only generate configs without training
            auto_track: Automatically track results after training
        """
        self.project_root = Path(project_root).absolute()
        self.gpus = gpus if gpus is not None else [0]
        self.dry_run = dry_run
        self.auto_track = auto_track
        
        self.config_generator = AblationConfigGenerator(
            base_config_dir=str(self.project_root / "configs")
        )
    
    def generate_config(
        self,
        experiment_id: str,
        dataset: str,
        exp_config: Dict[str, Any]
    ) -> str:
        """
        Generate config file for experiment
        
        Args:
            experiment_id: Experiment ID
            dataset: Dataset name
            exp_config: Experiment configuration dict
            
        Returns:
            Path to generated config file
        """
        script_norm = exp_config.get('script_norm', 'phonetic')
        transfer_mode = exp_config.get('transfer_mode', 'full')
        freeze_epochs = exp_config.get('freeze_epochs', 5)
        encoder_lr_mult = exp_config.get('encoder_lr_mult', 0.2)
        head_lr_mult = exp_config.get('head_lr_mult', 1.0)
        data_fraction = exp_config.get('data_fraction', 1.0)
        source_ckpt = exp_config.get('source_ckpt', None)
        
        if dataset == "LipBengal":
            config_path = self.config_generator.generate_lipbengal_config(
                experiment_id=experiment_id,
                script_norm=script_norm,
                transfer_mode=transfer_mode,
                freeze_epochs=freeze_epochs,
                encoder_lr_mult=encoder_lr_mult,
                head_lr_mult=head_lr_mult,
                target_data_fraction=data_fraction,
                source_ckpt=source_ckpt
            )
        elif dataset == "LRW-AR":
            config_path = self.config_generator.generate_lrwar_config(
                experiment_id=experiment_id,
                script_norm=script_norm,
                transfer_mode=transfer_mode,
                freeze_epochs=freeze_epochs,
                encoder_lr_mult=encoder_lr_mult,
                head_lr_mult=head_lr_mult,
                target_data_fraction=data_fraction,
                source_ckpt=source_ckpt
            )
        else:
            raise ValueError(f"Unknown dataset: {dataset}")
        
        return config_path
    
    def launch_training(
        self,
        config_path: str,
        experiment_id: str,
        dataset: str,
        gpu_id: int = 0,
        num_workers: int = 4,
        resume_checkpoint: Optional[str] = None
    ) -> subprocess.Popen:
        """
        Launch training process
        
        Args:
            config_path: Path to config file
            experiment_id: Experiment ID
            dataset: Dataset name
            gpu_id: GPU ID to use
            num_workers: Number of data loading workers
            resume_checkpoint: Checkpoint to resume from
            
        Returns:
            Subprocess handle
        """
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        os.environ['EXPERIMENT_ID'] = experiment_id
        
        # Convert absolute path to relative path from project root
        config_path_obj = Path(config_path)
        if config_path_obj.is_absolute():
            try:
                config_path = str(config_path_obj.relative_to(self.project_root))
            except ValueError:
                # If path is not relative to project root, use as-is
                pass
        
        cmd = [
            'python3', 'main.py',
            '--config_file', config_path,
            '--mode', 'training',
            '-j', str(num_workers),
        ]
        
        if resume_checkpoint:
            cmd.extend(['--checkpoint', resume_checkpoint])
        
        log_dir = self.project_root / 'logs' / 'ablations' / dataset / experiment_id
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / 'training.log'
        
        print(f"\n{'='*80}")
        print(f"Launching: {experiment_id} on {dataset}")
        print(f"Config: {config_path}")
        print(f"GPU: {gpu_id}")
        print(f"Log: {log_file}")
        print(f"{'='*80}\n")
        
        if self.dry_run:
            print("[DRY RUN] Would execute:", ' '.join(cmd))
            return None
        
        with open(log_file, 'w') as f:
            process = subprocess.Popen(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                cwd=str(self.project_root)
            )
        
        return process
    
    def launch_experiment(
        self,
        experiment_id: str,
        dataset: str,
        exp_config: Optional[Dict[str, Any]] = None,
        gpu_id: int = 0
    ) -> Optional[subprocess.Popen]:
        """
        Launch a single experiment
        
        Args:
            experiment_id: Experiment ID
            dataset: Dataset name
            exp_config: Experiment configuration (if None, load from matrix)
            gpu_id: GPU ID to use
            
        Returns:
            Subprocess handle
        """
        # Get experiment config from matrix if not provided
        if exp_config is None:
            matrix = get_experiment_matrix()
            exp_config = matrix.get(experiment_id)
            if exp_config is None:
                raise ValueError(f"Experiment {experiment_id} not found in matrix")
        
        # Generate config file
        config_path = self.generate_config(experiment_id, dataset, exp_config)
        
        # Launch training
        process = self.launch_training(
            config_path=config_path,
            experiment_id=experiment_id,
            dataset=dataset,
            gpu_id=gpu_id
        )
        
        return process
    
    def launch_phase(
        self,
        phase: int,
        dataset: str,
        parallel: bool = False
    ) -> List[subprocess.Popen]:
        """
        Launch all experiments in a phase
        
        Args:
            phase: Phase number (1, 2, or 3)
            dataset: Dataset name
            parallel: If True, launch experiments in parallel across GPUs
            
        Returns:
            List of subprocess handles
        """
        # Get all experiments for this phase
        matrix = get_experiment_matrix()
        phase_exps = {k: v for k, v in matrix.items() if v.get('phase') == phase}
        
        if not phase_exps:
            print(f"No experiments found for Phase {phase}")
            return []
        
        print(f"\n{'='*80}")
        print(f"Launching Phase {phase} experiments on {dataset}")
        print(f"Total experiments: {len(phase_exps)}")
        print(f"GPUs: {self.gpus}")
        print(f"Parallel: {parallel}")
        print(f"{'='*80}\n")
        
        processes = []
        
        if parallel and len(self.gpus) > 1:
            # Launch experiments in parallel across GPUs
            gpu_idx = 0
            for exp_id, exp_config in phase_exps.items():
                gpu_id = self.gpus[gpu_idx % len(self.gpus)]
                process = self.launch_experiment(exp_id, dataset, exp_config, gpu_id)
                if process:
                    processes.append((exp_id, process))
                    gpu_idx += 1
                    time.sleep(5)  # Stagger launches
        else:
            # Launch experiments sequentially
            for exp_id, exp_config in phase_exps.items():
                gpu_id = self.gpus[0]
                process = self.launch_experiment(exp_id, dataset, exp_config, gpu_id)
                if process and not self.dry_run:
                    # Wait for completion before next experiment
                    print(f"Waiting for {exp_id} to complete...")
                    process.wait()
                    processes.append((exp_id, process))
        
        return processes
    
    def launch_batch(
        self,
        experiment_ids: List[str],
        dataset: str,
        parallel: bool = False
    ) -> List[subprocess.Popen]:
        """
        Launch a batch of experiments
        
        Args:
            experiment_ids: List of experiment IDs
            dataset: Dataset name
            parallel: If True, launch in parallel across GPUs
            
        Returns:
            List of subprocess handles
        """
        matrix = get_experiment_matrix()
        
        processes = []
        gpu_idx = 0
        
        for exp_id in experiment_ids:
            if exp_id not in matrix:
                print(f"Warning: Experiment {exp_id} not found in matrix, skipping")
                continue
            
            exp_config = matrix[exp_id]
            gpu_id = self.gpus[gpu_idx % len(self.gpus)]
            
            process = self.launch_experiment(exp_id, dataset, exp_config, gpu_id)
            if process:
                processes.append((exp_id, process))
                
                if parallel and len(self.gpus) > 1:
                    gpu_idx += 1
                    time.sleep(5)
                else:
                    # Sequential: wait for completion
                    if not self.dry_run:
                        print(f"Waiting for {exp_id} to complete...")
                        process.wait()
        
        return processes
    
    def wait_for_completion(self, processes: List[tuple]) -> Dict[str, int]:
        """
        Wait for all processes to complete
        
        Args:
            processes: List of (experiment_id, process) tuples
            
        Returns:
            Dict mapping experiment_id to exit code
        """
        results = {}
        
        for exp_id, process in processes:
            if process is None:
                continue
            
            print(f"Waiting for {exp_id}...")
            exit_code = process.wait()
            results[exp_id] = exit_code
            
            if exit_code == 0:
                print(f"✓ {exp_id} completed successfully")
            else:
                print(f"✗ {exp_id} failed with exit code {exit_code}")
        
        return results


def main():
    parser = argparse.ArgumentParser(description="Launch ablation study experiments")
    parser.add_argument("--exp-id", type=str, help="Single experiment ID to launch")
    parser.add_argument("--dataset", type=str, choices=["LipBengal", "LRW-AR"], 
                       required=True, help="Dataset")
    parser.add_argument("--phase", type=int, choices=[1, 2, 3], 
                       help="Launch all experiments in phase")
    parser.add_argument("--batch", type=str, nargs='+', 
                       help="Launch batch of experiment IDs")
    parser.add_argument("--gpus", type=str, default="0", 
                       help="Comma-separated list of GPU IDs (e.g., 0,1,2)")
    parser.add_argument("--parallel", action="store_true", 
                       help="Launch experiments in parallel across GPUs")
    parser.add_argument("--dry-run", action="store_true", 
                       help="Generate configs only, don't launch training")
    parser.add_argument("--workers", type=int, default=4, 
                       help="Number of data loading workers")
    
    args = parser.parse_args()
    
    # Parse GPU list
    gpu_list = [int(g.strip()) for g in args.gpus.split(',')]
    
    # Initialize launcher
    launcher = AblationLauncher(
        gpus=gpu_list,
        dry_run=args.dry_run
    )
    
    # Launch experiments
    if args.exp_id:
        # Single experiment
        process = launcher.launch_experiment(args.exp_id, args.dataset, gpu_id=gpu_list[0])
        if process and not args.dry_run:
            exit_code = process.wait()
            sys.exit(exit_code)
    
    elif args.phase:
        # Phase batch
        processes = launcher.launch_phase(args.phase, args.dataset, parallel=args.parallel)
        if processes and not args.dry_run:
            results = launcher.wait_for_completion(processes)
            # Exit with error if any experiment failed
            if any(code != 0 for code in results.values()):
                sys.exit(1)
    
    elif args.batch:
        # Custom batch
        processes = launcher.launch_batch(args.batch, args.dataset, parallel=args.parallel)
        if processes and not args.dry_run:
            results = launcher.wait_for_completion(processes)
            if any(code != 0 for code in results.values()):
                sys.exit(1)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

