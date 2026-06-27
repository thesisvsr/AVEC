#!/usr/bin/env python3
"""
Centralized Environment Variable Control System for Ablation Study

Provides utility functions to parse, validate, and manage environment variables
used across all ablation experiments.

Usage:
    from scripts.ablation_env import AblationEnv
    
    env = AblationEnv()
    script_norm = env.get_script_normalization()
    transfer_mode = env.get_transfer_mode()
    freeze_epochs = env.get_freeze_epochs()
"""

import os
from typing import Optional, Dict, Any
from enum import Enum


class ScriptNormalization(Enum):
    """Script normalization types for Phase 1 experiments"""
    RAW = "raw"                # Raw Bengali/Arabic script (expected to fail)
    PHONETIC = "phonetic"      # IPA-based phonetic romanization (your approach)
    SIMPLE = "simple"          # Simple character-level transliteration
    MIXED = "mixed"            # Preserve script + romanization auxiliary


class TransferMode(Enum):
    """Transfer learning modes for Phase 2 experiments"""
    SCRATCH = "scratch"        # Train from random initialization
    FULL = "full"              # Transfer full encoder (frontend + backend)
    FRONTEND = "frontend"      # Transfer frontend only (ResNet-18)
    BACKEND = "backend"        # Transfer backend only (Conformer blocks)


class FreezeMode(Enum):
    """Encoder freezing modes"""
    NONE = "none"              # No freezing
    FULL = "full"              # Freeze encoder, unfreeze after N epochs
    PROGRESSIVE = "progressive" # Progressive layer-by-layer unfreezing
    PERMANENT = "permanent"    # Permanently frozen encoder


class AblationEnv:
    """Centralized environment variable management for ablation experiments"""
    
    # Environment variable names
    SCRIPT_NORMALIZATION_TYPE = "SCRIPT_NORMALIZATION_TYPE"
    TRANSFER_MODE = "TRANSFER_MODE"
    FREEZE_ENCODER_EPOCHS = "FREEZE_ENCODER_EPOCHS"
    FREEZE_MODE = "FREEZE_MODE"
    ENCODER_LR_MULT = "ENCODER_LR_MULT"
    HEAD_LR_MULT = "HEAD_LR_MULT"
    TARGET_DATA_FRACTION = "TARGET_DATA_FRACTION"
    SOURCE_CKPT = "SOURCE_CKPT"
    EXPERIMENT_ID = "EXPERIMENT_ID"
    
    # Dataset-specific prefixes
    LIPBENGAL_PREFIX = "LIPBENGAL_"
    LRWAR_PREFIX = "LRWAR_"
    
    def __init__(self, dataset: Optional[str] = None):
        """
        Initialize environment variable manager
        
        Args:
            dataset: Dataset name ("LipBengal" or "LRW-AR") for dataset-specific vars
        """
        self.dataset = dataset
        self._prefix = ""
        if dataset == "LipBengal":
            self._prefix = self.LIPBENGAL_PREFIX
        elif dataset == "LRW-AR":
            self._prefix = self.LRWAR_PREFIX
    
    def _get_env(self, key: str, default: Any = None, prefix: bool = True) -> str:
        """
        Get environment variable with optional dataset prefix
        
        Args:
            key: Environment variable name
            default: Default value if not set
            prefix: Whether to try prefixed version first
            
        Returns:
            Environment variable value or default
        """
        if prefix and self._prefix:
            # Try prefixed version first (e.g., LIPBENGAL_FREEZE_EPOCHS)
            prefixed_key = f"{self._prefix}{key}"
            if prefixed_key in os.environ:
                return os.environ[prefixed_key]
        
        # Fall back to unprefixed version
        return os.environ.get(key, default)
    
    def get_script_normalization(self, default: str = "phonetic") -> ScriptNormalization:
        """Get script normalization type"""
        value = self._get_env(self.SCRIPT_NORMALIZATION_TYPE, default)
        try:
            return ScriptNormalization(value)
        except ValueError:
            print(f"Warning: Invalid script normalization '{value}', using default '{default}'")
            return ScriptNormalization(default)
    
    def get_transfer_mode(self, default: str = "full") -> TransferMode:
        """Get transfer learning mode"""
        value = self._get_env(self.TRANSFER_MODE, default)
        try:
            return TransferMode(value)
        except ValueError:
            print(f"Warning: Invalid transfer mode '{value}', using default '{default}'")
            return TransferMode(default)
    
    def get_freeze_mode(self, default: str = "full") -> FreezeMode:
        """Get freeze mode"""
        value = self._get_env(self.FREEZE_MODE, default)
        try:
            return FreezeMode(value)
        except ValueError:
            print(f"Warning: Invalid freeze mode '{value}', using default '{default}'")
            return FreezeMode(default)
    
    def get_freeze_epochs(self, default: int = 5) -> int:
        """Get number of epochs to freeze encoder"""
        value = self._get_env(self.FREEZE_ENCODER_EPOCHS, str(default))
        try:
            return int(value)
        except ValueError:
            print(f"Warning: Invalid freeze epochs '{value}', using default {default}")
            return default
    
    def get_encoder_lr_mult(self, default: float = 0.2) -> float:
        """Get encoder learning rate multiplier"""
        value = self._get_env(self.ENCODER_LR_MULT, str(default))
        try:
            return float(value)
        except ValueError:
            print(f"Warning: Invalid encoder LR mult '{value}', using default {default}")
            return default
    
    def get_head_lr_mult(self, default: float = 1.0) -> float:
        """Get head learning rate multiplier"""
        value = self._get_env(self.HEAD_LR_MULT, str(default))
        try:
            return float(value)
        except ValueError:
            print(f"Warning: Invalid head LR mult '{value}', using default {default}")
            return default
    
    def get_target_data_fraction(self, default: float = 1.0) -> float:
        """Get target dataset fraction (for data size ablation)"""
        value = self._get_env(self.TARGET_DATA_FRACTION, str(default))
        try:
            fraction = float(value)
            if not 0.0 < fraction <= 1.0:
                print(f"Warning: Data fraction {fraction} out of range (0,1], using default {default}")
                return default
            return fraction
        except ValueError:
            print(f"Warning: Invalid data fraction '{value}', using default {default}")
            return default
    
    def get_source_checkpoint(self, default: Optional[str] = None) -> Optional[str]:
        """Get source checkpoint path for transfer learning"""
        if default is None:
            default = "callbacks/LRS23/VO/EffConfInterCTC/checkpoints_swa-equal-90-100.ckpt"
        return self._get_env(self.SOURCE_CKPT, default)
    
    def get_experiment_id(self, default: Optional[str] = None) -> Optional[str]:
        """Get experiment ID"""
        return self._get_env(self.EXPERIMENT_ID, default, prefix=False)
    
    def get_all_config(self) -> Dict[str, Any]:
        """Get all configuration values as a dictionary"""
        return {
            "script_normalization": self.get_script_normalization().value,
            "transfer_mode": self.get_transfer_mode().value,
            "freeze_mode": self.get_freeze_mode().value,
            "freeze_epochs": self.get_freeze_epochs(),
            "encoder_lr_mult": self.get_encoder_lr_mult(),
            "head_lr_mult": self.get_head_lr_mult(),
            "target_data_fraction": self.get_target_data_fraction(),
            "source_checkpoint": self.get_source_checkpoint(),
            "experiment_id": self.get_experiment_id(),
            "dataset": self.dataset,
        }
    
    def print_config(self):
        """Print current configuration"""
        config = self.get_all_config()
        print("=" * 60)
        print("Ablation Experiment Configuration")
        print("=" * 60)
        for key, value in config.items():
            print(f"  {key:.<40} {value}")
        print("=" * 60)
    
    @staticmethod
    def set_experiment_env(
        experiment_id: str,
        script_norm: Optional[str] = None,
        transfer_mode: Optional[str] = None,
        freeze_epochs: Optional[int] = None,
        freeze_mode: Optional[str] = None,
        encoder_lr_mult: Optional[float] = None,
        head_lr_mult: Optional[float] = None,
        data_fraction: Optional[float] = None,
        source_ckpt: Optional[str] = None,
    ):
        """
        Set environment variables for an experiment
        
        Useful for programmatically configuring experiments before launching
        """
        os.environ["EXPERIMENT_ID"] = experiment_id
        
        if script_norm is not None:
            os.environ["SCRIPT_NORMALIZATION_TYPE"] = script_norm
        if transfer_mode is not None:
            os.environ["TRANSFER_MODE"] = transfer_mode
        if freeze_epochs is not None:
            os.environ["FREEZE_ENCODER_EPOCHS"] = str(freeze_epochs)
        if freeze_mode is not None:
            os.environ["FREEZE_MODE"] = freeze_mode
        if encoder_lr_mult is not None:
            os.environ["ENCODER_LR_MULT"] = str(encoder_lr_mult)
        if head_lr_mult is not None:
            os.environ["HEAD_LR_MULT"] = str(head_lr_mult)
        if data_fraction is not None:
            os.environ["TARGET_DATA_FRACTION"] = str(data_fraction)
        if source_ckpt is not None:
            os.environ["SOURCE_CKPT"] = source_ckpt
    
    @staticmethod
    def clear_experiment_env():
        """Clear all ablation-related environment variables"""
        env_vars = [
            "EXPERIMENT_ID",
            "SCRIPT_NORMALIZATION_TYPE",
            "TRANSFER_MODE",
            "FREEZE_ENCODER_EPOCHS",
            "FREEZE_MODE",
            "ENCODER_LR_MULT",
            "HEAD_LR_MULT",
            "TARGET_DATA_FRACTION",
            "SOURCE_CKPT",
        ]
        for var in env_vars:
            os.environ.pop(var, None)


def get_experiment_matrix() -> Dict[str, Dict[str, Any]]:
    """
    Define the full experiment matrix for ablation study
    
    Returns:
        Dictionary mapping experiment IDs to their configurations
    """
    experiments = {}
    
    # Phase 1: Script Normalization (S1.X)
    for script_norm in ["raw", "phonetic", "simple", "mixed"]:
        exp_id = f"S1_{script_norm}"
        experiments[exp_id] = {
            "phase": 1,
            "script_norm": script_norm,
            "transfer_mode": "full",
            "freeze_epochs": 5,
            "encoder_lr_mult": 0.2,
        }
    
    # Phase 2: Transfer Learning (T1.X - T3.X)
    
    # T1: Transfer modes
    for transfer_mode in ["scratch", "full", "frontend", "backend"]:
        exp_id = f"T1_{transfer_mode}"
        experiments[exp_id] = {
            "phase": 2,
            "script_norm": "phonetic",
            "transfer_mode": transfer_mode,
            "freeze_epochs": 5 if transfer_mode != "scratch" else 0,
            "encoder_lr_mult": 0.2,
        }
    
    # T2: Freezing strategies
    for freeze_epochs in [0, 3, 5, 10]:
        exp_id = f"T2_freeze_{freeze_epochs}ep"
        experiments[exp_id] = {
            "phase": 2,
            "script_norm": "phonetic",
            "transfer_mode": "full",
            "freeze_epochs": freeze_epochs,
            "encoder_lr_mult": 0.2,
        }
    
    # T3: Differential LR
    for encoder_lr in [0.1, 0.2, 0.5, 1.0]:
        exp_id = f"T3_lr_{str(encoder_lr).replace('.', '_')}"
        experiments[exp_id] = {
            "phase": 2,
            "script_norm": "phonetic",
            "transfer_mode": "full",
            "freeze_epochs": 5,
            "encoder_lr_mult": encoder_lr,
        }
    
    # Phase 3: Dataset Analysis (D1.X - D2.X)
    
    # D1: Data fraction
    for fraction in [0.1, 0.25, 0.5, 0.75, 1.0]:
        exp_id = f"D1_data_{int(fraction*100)}pct"
        experiments[exp_id] = {
            "phase": 3,
            "script_norm": "phonetic",
            "transfer_mode": "full",
            "freeze_epochs": 5,
            "encoder_lr_mult": 0.2,
            "data_fraction": fraction,
        }
    
    return experiments


def main():
    """Test the environment variable system"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test ablation environment variables")
    parser.add_argument("--dataset", choices=["LipBengal", "LRW-AR"], help="Dataset name")
    parser.add_argument("--show-matrix", action="store_true", help="Show experiment matrix")
    
    args = parser.parse_args()
    
    if args.show_matrix:
        matrix = get_experiment_matrix()
        print(f"\nExperiment Matrix ({len(matrix)} experiments):")
        print("=" * 80)
        for exp_id, config in sorted(matrix.items()):
            print(f"{exp_id:.<30} Phase {config['phase']}, {config}")
        return
    
    env = AblationEnv(dataset=args.dataset)
    env.print_config()


if __name__ == "__main__":
    main()


