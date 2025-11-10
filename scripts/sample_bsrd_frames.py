#!/usr/bin/env python
import os, sys, torch, random
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
import torchvision
from nnet import datasets as ds

"""Sample a few preprocessed BSRD clips and export representative frames.
Outputs go to export/bsrd_samples_<mode>/sample_<i>_frame_<t>.png
Usage:
  source .venv/bin/activate
  python scripts/sample_bsrd_frames.py [mode] [num_samples] [frames_per_sample]
Defaults: mode=train num_samples=3 frames_per_sample=4
Respects prepared_only logic (will only use prepared frames present in indices file).
"""

def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else 'train'
    num_samples = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    frames_per_sample = int(sys.argv[3]) if len(sys.argv) > 3 else 4

    # BSRDCTC inherits a base Dataset expecting batch_size & collate_fn; we only need raw items here.
    # Provide dummy batch_size=1 and identity collate.
    dataset = ds.BSRDCTC(batch_size=1, collate_fn=lambda x: x, mode=mode, root='datasets', prepared_only=True)
    if len(dataset) == 0:
        print(f"No samples available in mode={mode}")
        return
    export_dir = os.path.join('export', f'bsrd_samples_{mode}')
    os.makedirs(export_dir, exist_ok=True)

    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
    for i, idx in enumerate(indices):
        video, _, label, vlen, _, _ = dataset[idx]
        # video shape (C,T,H,W) after preprocessing; grayscale so C=1
        C, T, H, W = video.shape
        take = min(frames_per_sample, T)
        sel = torch.linspace(0, T-1, steps=take).round().long()
        frames = video[:, sel]  # (C, take, H, W)
        # Undo normalization roughly (assuming mean=0.5 std=0.5 from config)
        frames_vis = frames * 0.5 + 0.5
        frames_vis = torch.clamp(frames_vis, 0, 1)
        for j in range(frames_vis.shape[1]):
            out_path = os.path.join(export_dir, f'sample_{i}_frame_{j}.png')
            torchvision.utils.save_image(frames_vis[:, j], out_path)
        # Save a tiny text file with label indices
        with open(os.path.join(export_dir, f'sample_{i}_meta.txt'), 'w', encoding='utf-8') as f:
            f.write(f'sample_index={idx}\n')
            f.write(f'video_len={int(vlen)}\n')
            f.write(f'label_ids={label.tolist()}\n')
        print(f'Exported sample {i} (dataset idx {idx}) to {export_dir}')

if __name__ == '__main__':
    main()
