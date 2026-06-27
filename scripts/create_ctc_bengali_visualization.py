#!/usr/bin/env python3
"""
Create CTC visualization for LipBengal dataset showing:
1. Full person frames
2. Cropped lip portions (from prepared data)
3. Bengali -> English transliteration with CTC alignment
4. Final word (in both Bengali and English)
"""

import sys
from pathlib import Path
import torch
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import json
import random

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.banglish_lookup import to_banglish


def load_lipbengal_sample(dataset_root, speaker='s100', target_word=None):
    """
    Load a sample from LipBengal dataset.
    
    Returns:
        bengali_word: Original Bengali word
        transliterated: Romanized version
        frames: List of full frame paths
        prepared_pt_path: Path to prepared mouth crop tensor
    """
    root = Path(dataset_root)
    speaker_dir = root / speaker
    
    if not speaker_dir.exists():
        raise ValueError(f"Speaker {speaker} not found")
    
    # Get word directories
    word_dirs = [d for d in speaker_dir.iterdir() if d.is_dir()]
    
    if target_word:
        # Try to find specific word
        matching = [d for d in word_dirs if d.name == target_word]
        if not matching:
            raise ValueError(f"Word {target_word} not found for speaker {speaker}")
        word_dir = matching[0]
    else:
        # Pick a random word
        word_dir = random.choice(word_dirs)
    
    bengali_word = word_dir.name
    transliterated = to_banglish(bengali_word)
    
    # Get frame files
    frame_files = sorted(word_dir.glob("*.jpg"))[:8]  # Take first 8 frames
    
    if len(frame_files) == 0:
        raise ValueError(f"No frames found in {word_dir}")
    
    # Find prepared tensor (mouth crop)
    # Pattern: datasets/LipBengal/prepared/{split}/{speaker}/{word}/{hash}.pt
    prepared_pt = None
    for split in ['train', 'val', 'test']:
        prepared_dir = root / 'prepared' / split / speaker / bengali_word
        if prepared_dir.exists():
            pt_files = list(prepared_dir.glob("*.pt"))
            if pt_files:
                prepared_pt = pt_files[0]
                break
    
    return {
        'bengali_word': bengali_word,
        'transliterated': transliterated,
        'frames': frame_files,
        'prepared_pt': prepared_pt
    }


def load_prepared_mouth_crop(pt_path, num_frames=7):
    """Load and extract frames from prepared .pt file."""
    obj = torch.load(pt_path, map_location='cpu')
    
    if isinstance(obj, dict) and 'frames' in obj:
        frames_tensor = obj['frames']  # (T, H, W) uint8
    elif isinstance(obj, dict) and 'x' in obj:
        frames_tensor = obj['x']  # (T, H, W)
    else:
        raise ValueError(f"Unknown prepared file format: {pt_path}")
    
    # Convert to numpy
    if isinstance(frames_tensor, torch.Tensor):
        frames_np = frames_tensor.numpy()
    else:
        frames_np = frames_tensor
    
    # Ensure uint8
    if frames_np.dtype != np.uint8:
        frames_np = (frames_np * 255).astype(np.uint8)
    
    # Sample frames if we have more than needed
    T = frames_np.shape[0]
    if T > num_frames:
        indices = np.linspace(0, T-1, num_frames).astype(int)
        frames_np = frames_np[indices]
    elif T < num_frames:
        # Pad by repeating last frame
        padding = np.repeat(frames_np[-1:], num_frames - T, axis=0)
        frames_np = np.concatenate([frames_np, padding], axis=0)
    
    return frames_np


def create_ctc_alignment(transliterated_word, num_frames):
    """
    Create realistic CTC alignment with blanks.
    
    Args:
        transliterated_word: Romanized word (e.g., "namoshkar")
        num_frames: Number of video frames
    
    Returns:
        alignment: List of characters (including '-' for blank)
    """
    chars = list(transliterated_word.lower())
    
    if num_frames <= len(chars):
        # Tight alignment - no blanks
        alignment = chars[:num_frames]
        # Pad if needed
        while len(alignment) < num_frames:
            alignment.append('-')
    else:
        # Spread characters across frames with blanks
        alignment = []
        positions = np.linspace(0, num_frames-1, len(chars)).astype(int)
        
        for i in range(num_frames):
            if i in positions:
                char_idx = list(positions).index(i)
                alignment.append(chars[char_idx])
            else:
                alignment.append('-')
    
    return alignment


def create_bengali_ctc_visualization(
    dataset_root='/home/thesis/Thesis/AVEC/datasets/LipBengal',
    speaker='s100',
    target_word=None,
    output_path='export/paper_samples/ctc_bengali.png'
):
    """Create the full visualization."""
    
    # Load sample
    print(f"Loading sample from LipBengal...")
    sample = load_lipbengal_sample(dataset_root, speaker, target_word)
    
    bengali_word = sample['bengali_word']
    transliterated = sample['transliterated']
    frame_files = sample['frames']
    prepared_pt = sample['prepared_pt']
    
    print(f"  Bengali word: {bengali_word}")
    print(f"  Transliterated: {transliterated}")
    print(f"  Frames: {len(frame_files)}")
    print(f"  Prepared mouth crop: {prepared_pt}")
    
    if prepared_pt is None:
        raise ValueError("No prepared mouth crop found for this sample")
    
    # Load frames
    num_frames = min(7, len(frame_files))
    full_frames = []
    for frame_file in frame_files[:num_frames]:
        img = Image.open(frame_file).convert('L')
        full_frames.append(np.array(img))
    
    # Load mouth crops
    mouth_frames = load_prepared_mouth_crop(prepared_pt, num_frames)
    
    # Create CTC alignment
    alignment = create_ctc_alignment(transliterated, num_frames)
    
    # Create visualization
    print("Creating visualization...")
    fig = plt.figure(figsize=(18, 12))
    
    # Calculate layout
    frame_height = 0.12
    spacing = 0.08
    top = 0.95
    
    # Row 1: Full person frames
    row1_y = top
    for i, frame in enumerate(full_frames):
        ax = fig.add_axes([0.1 + i*0.12, row1_y - frame_height, 0.10, frame_height])
        ax.imshow(frame, cmap='gray')
        ax.axis('off')
        ax.set_title(f'Frame {i+1}', fontsize=9, pad=3)
    
    # Add label for full frames
    fig.text(0.02, row1_y - frame_height/2, 'Full\nFrames', 
             fontsize=11, ha='left', va='center', weight='bold')
    
    # Row 2: Lip crops
    row2_y = row1_y - frame_height - spacing
    for i, lip_frame in enumerate(mouth_frames):
        ax = fig.add_axes([0.1 + i*0.12, row2_y - frame_height, 0.10, frame_height])
        ax.imshow(lip_frame, cmap='gray')
        ax.axis('off')
    
    fig.text(0.02, row2_y - frame_height/2, 'Lip\nCrops', 
             fontsize=11, ha='left', va='center', weight='bold')
    
    # Row 3: Transliterated characters with CTC alignment
    row3_y = row2_y - frame_height - spacing
    for i, char in enumerate(alignment):
        x = 0.15 + i*0.12
        y = row3_y - 0.04
        
        # Draw box for character
        if char == '-':
            # Blank token
            bbox = dict(boxstyle='round,pad=0.5', facecolor='lightgray', 
                       edgecolor='gray', linewidth=1.5)
            fig.text(x, y, '∅', fontsize=14, ha='center', va='center', 
                    bbox=bbox, family='monospace')
        else:
            # Character
            bbox = dict(boxstyle='round,pad=0.5', facecolor='lightblue', 
                       edgecolor='blue', linewidth=2)
            fig.text(x, y, char, fontsize=14, ha='center', va='center', 
                    bbox=bbox, weight='bold', family='monospace')
        
        # Draw arrows from lip to character
        arrow = FancyArrowPatch(
            (0.15 + i*0.12, row2_y - frame_height - 0.01),
            (x, y + 0.03),
            arrowstyle='->', 
            lw=1.5, 
            color='darkgreen', 
            alpha=0.6,
            mutation_scale=15
        )
        fig.add_artist(arrow)
    
    fig.text(0.02, row3_y - 0.04, 'CTC\nAlignment', 
             fontsize=11, ha='left', va='center', weight='bold')
    
    # Row 4: Final decoded word
    row4_y = row3_y - 0.15
    
    # Show both Bengali and transliterated
    fig.text(0.5, row4_y, 'Decoded Word', 
             fontsize=14, ha='center', weight='bold')
    
    # Bengali word (larger, on top)
    fig.text(0.5, row4_y - 0.06, bengali_word, 
             fontsize=24, ha='center', weight='bold',
             bbox=dict(boxstyle='round,pad=0.8', facecolor='gold', 
                      edgecolor='orange', linewidth=3))
    
    # Transliterated word (smaller, below)
    fig.text(0.5, row4_y - 0.13, f'({transliterated})', 
             fontsize=16, ha='center', style='italic',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', 
                      edgecolor='orange', linewidth=2))
    
    # Add title
    fig.text(0.5, 0.98, 'CTC Alignment Visualization - LipBengal (Bengali)', 
             fontsize=16, ha='center', weight='bold')
    
    # Add subtitle with transliteration info
    fig.text(0.5, 0.945, 'Original Bengali → Banglish Transliteration → CTC Alignment', 
             fontsize=11, ha='center', style='italic', color='gray')
    
    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"\n✓ Visualization saved to: {output_path}")
    print(f"  Bengali word: {bengali_word}")
    print(f"  Transliterated: {transliterated}")
    print(f"  Frames used: {num_frames}")
    print(f"  CTC alignment: {' '.join(alignment)}")
    
    return output_path


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Create Bengali CTC visualization')
    parser.add_argument('--dataset', type=str, 
                       default='/home/thesis/Thesis/AVEC/datasets/LipBengal',
                       help='Path to LipBengal dataset')
    parser.add_argument('--speaker', type=str, default='s100',
                       help='Speaker ID (e.g., s100)')
    parser.add_argument('--word', type=str, default=None,
                       help='Specific Bengali word to visualize (optional)')
    parser.add_argument('--output', type=str, 
                       default='export/paper_samples/ctc_bengali.png',
                       help='Output image path')
    
    args = parser.parse_args()
    
    try:
        create_bengali_ctc_visualization(
            dataset_root=args.dataset,
            speaker=args.speaker,
            target_word=args.word,
            output_path=args.output
        )
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

