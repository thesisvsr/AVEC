#!/usr/bin/env python3
"""Prepare a transfer-learning initialization checkpoint for LipBengal from an LRS2 SWA model.

This script:
 1. Loads LipBengal config to instantiate the target model (with its vocab size).
 2. Loads the LRS2 SWA checkpoint.
 3. Filters out incompatible tensors (classifier head weight/bias) and loads the rest.
 4. Saves a new lightweight checkpoint (model_state_dict only) named
       transfer_from_LRS2_swa.ckpt
    inside LipBengal callback directory for convenient usage:
       python main.py -c configs/LipBengal/AV/VisualCE.py -m training \
           --checkpoint transfer_from_LRS2_swa.ckpt -j 20

Optionally you can pass --freeze_epochs to record in the JSON sidecar how many
initial epochs you intend to keep the encoder frozen (manual enforcement is in
config patch).
"""
from __future__ import annotations
import argparse, os, sys, json
from pathlib import Path
import torch, importlib

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--swa_checkpoint', default='callbacks/LRS2/V/checkpoints_swa-equal-90-100.ckpt', help='Source SWA checkpoint path')
    ap.add_argument('--target_config', default='configs.LipBengal.AV.VisualCE', help='Target config module path')
    ap.add_argument('--out_name', default='transfer_from_LRS2_swa.ckpt', help='Output checkpoint name')
    ap.add_argument('--freeze_epochs', type=int, default=5, help='Planned freeze epochs (metadata only)')
    return ap.parse_args()


def main():
    args = parse_args()
    swa_path = Path(args.swa_checkpoint)
    assert swa_path.is_file(), f"SWA checkpoint not found: {swa_path}"

    cfg = importlib.import_module(args.target_config.replace('.py','').replace('/', '.'))
    model = cfg.model  # Already constructed
    target_state = model.state_dict()

    ckpt = torch.load(str(swa_path), map_location='cpu')
    source_state = ckpt.get('model_state_dict', ckpt)

    # Filter: keep only matching shapes (avoid head mismatch)
    filtered = {}
    skipped = []
    for k, v in source_state.items():
        if k in target_state and target_state[k].shape == v.shape:
            filtered[k] = v
        else:
            if k.startswith('encoder.head.'):
                skipped.append(k)
    missing = [k for k in target_state.keys() if k not in filtered]

    model.load_state_dict(filtered, strict=False)
    print(f"Loaded {len(filtered)}/{len(target_state)} tensors from SWA; skipped {len(skipped)} head params; missing {len(missing)} (expected for new head).")

    # Save lightweight checkpoint (no optimizer)
    out_dir = Path(cfg.callback_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_ckpt = out_dir / args.out_name
    torch.save({'model_state_dict': model.state_dict(), 'transfer_source': str(swa_path), 'freeze_epochs': args.freeze_epochs}, out_ckpt)
    meta = {
        'transfer_source': str(swa_path),
        'output_checkpoint': str(out_ckpt),
        'loaded_tensors': len(filtered),
        'total_target_tensors': len(target_state),
        'skipped': skipped,
        'freeze_epochs': args.freeze_epochs,
    }
    (out_dir / (out_ckpt.stem + '.json')).write_text(json.dumps(meta, indent=2), encoding='utf-8')
    print('Saved transfer checkpoint:', out_ckpt)
    print('Metadata JSON:', out_dir / (out_ckpt.stem + '.json'))

if __name__ == '__main__':
    main()
