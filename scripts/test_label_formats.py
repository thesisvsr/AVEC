#!/usr/bin/env python3
"""
Test that label format mapping is working correctly for all Phase 1 experiments
"""

import sys
import torch
import torch.nn as nn
sys.path.insert(0, ".")

import nnet
from nnet import transforms as vtf

print("=" * 70)
print("Testing Label Format Implementation")
print("=" * 70)
print()

# Setup basic transforms
crop_size = (88, 88)
val_video_transform = vtf.CenterCropVideo(crop_size)
collate_fn = nnet.CollateFn(inputs_params=[{"axis": 0}], targets_params=[{"axis": 2}])

formats = ["phonetic", "raw", "simple", "mixed"]
expected_samples = {
    "phonetic": ["odhjojo়n", "onubhob", "ondhokar"],
    "raw": ["অধ্যয়ন", "অনুভব", "অন্ধকার"],
    "simple": ["adhjj়n", "anubhb", "andhkaar"],
    "mixed": ["odjojo়n", "onubob", "ondokar"],
}

for format_name in formats:
    print(f"Testing label_format='{format_name}'...")
    
    try:
        dataset = nnet.datasets.LipBengal(
            batch_size=32,
            collate_fn=collate_fn,
            mode="val",
            video_transform=val_video_transform,
            fixed_frames=29,
            indices_path="datasets/LipBengal/indices/val.pt",
            subset_fraction=0.01,  # Just a tiny subset for testing
            subset_seed=42,
            prepared_only=True,
            label_format=format_name,
        )
        
        print(f"  ✓ Dataset loaded successfully")
        print(f"  - Num classes: {dataset.num_classes}")
        print(f"  - Sample classes (first 5): {dataset.classes[:5]}")
        
        # Check if expected samples are in classes
        found = 0
        for expected in expected_samples[format_name]:
            if expected in dataset.classes:
                found += 1
        
        if found > 0:
            print(f"  ✓ Found {found}/{len(expected_samples[format_name])} expected samples")
        else:
            print(f"  ⚠ Expected samples not found in classes!")
            print(f"    Expected: {expected_samples[format_name]}")
            print(f"    Got: {dataset.classes[:10]}")
        
        print()
        
    except Exception as e:
        print(f"  ✗ ERROR: {e}")
        print()
        import traceback
        traceback.print_exc()
        continue

print("=" * 70)
print("✓ Label format testing complete!")
print("=" * 70)


