# Ablation Study Config Generator

This directory contains tools for generating ablation study experiment configurations.

## Quick Start

Generate all Phase 1 configs (Script Normalization):
```bash
python3 configs/ablations/template_generator.py --phase 1
```

Generate all Phase 2 configs (Transfer Learning Strategy):
```bash
python3 configs/ablations/template_generator.py --phase 2
```

Generate all Phase 3 configs (Dataset Analysis):
```bash
python3 configs/ablations/template_generator.py --phase 3
```

Generate ALL configs:
```bash
python3 configs/ablations/template_generator.py --generate-all
```

## Generated Configs

The generator creates config files in:
- `configs/LipBengal/AV/ablations/`
- `configs/LRW-AR/AV/ablations/`

## Experiment Naming Convention

- **S1_X**: Phase 1 - Script Normalization experiments
- **T1_X**: Phase 2 - Transfer Learning Effectiveness
- **T2_X**: Phase 2 - Freezing Strategy
- **T3_X**: Phase 2 - Differential Learning Rates
- **D1_X**: Phase 3 - Dataset Size Impact
- **D2_X**: Phase 3 - Vocabulary Size Impact
- **C1_X**: Phase 3 - Cross-Script Generalization

## Environment Variable Overrides

All generated configs support environment variable overrides:
- `SCRIPT_NORMALIZATION_TYPE`: raw, phonetic, simple, mixed
- `TRANSFER_MODE`: scratch, full, frontend, backend
- `FREEZE_ENCODER_EPOCHS`: integer (0 = no freeze)
- `ENCODER_LR_MULT`: float (e.g., 0.1, 0.2, 0.5, 1.0)
- `HEAD_LR_MULT`: float (typically 1.0)
- `TARGET_DATA_FRACTION`: float 0.0-1.0
- `SOURCE_CKPT`: path to source checkpoint

Example:
```bash
FREEZE_ENCODER_EPOCHS=10 python main.py -c configs/LipBengal/AV/ablations/T2_freeze_3ep.py -m training
```


