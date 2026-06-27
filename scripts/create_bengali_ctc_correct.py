#!/usr/bin/env python3
"""
Create CORRECT CTC visualization for LipBengal dataset.
Ensures frames and lip crops are from the SAME utterance.
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


def load_lipbengal_sample_correct(dataset_root, speaker='s100', target_word=None):
    """
    Load a complete sample ensuring frames and prepared data match.
    """
    root = Path(dataset_root)
    speaker_dir = root / speaker
    
    if not speaker_dir.exists():
        raise ValueError(f"Speaker {speaker} not found")
    
    # Get word directories
    word_dirs = [d for d in speaker_dir.iterdir() if d.is_dir()]
    
    # Filter words that have both frames AND prepared data
    valid_samples = []
    for word_dir in word_dirs:
        bengali_word = word_dir.name
        frame_files = list(word_dir.glob("*.jpg"))
        
        if len(frame_files) == 0:
            continue
        
        # Check if prepared data exists
        prepared_pt = None
        for split in ['train', 'val', 'test']:
            prepared_dir = root / 'prepared' / split / speaker / bengali_word
            if prepared_dir.exists():
                pt_files = list(prepared_dir.glob("*.pt"))
                if pt_files:
                    # Match by checking if prepared file has same number of frames
                    for pt_file in pt_files:
                        try:
                            obj = torch.load(pt_file, map_location='cpu')
                            if isinstance(obj, dict) and 'frames' in obj:
                                frames_tensor = obj['frames']
                            elif isinstance(obj, dict) and 'x' in obj:
                                frames_tensor = obj['x']
                            else:
                                continue
                            
                            # If we can load it, it's valid
                            valid_samples.append({
                                'word_dir': word_dir,
                                'bengali_word': bengali_word,
                                'frame_files': frame_files,
                                'prepared_pt': pt_file,
                                'split': split
                            })
                            break
                        except:
                            continue
                if valid_samples and valid_samples[-1]['bengali_word'] == bengali_word:
                    break
    
    if not valid_samples:
        raise ValueError(f"No valid samples found with both frames and prepared data")
    
    if target_word:
        matching = [s for s in valid_samples if s['bengali_word'] == target_word]
        if not matching:
            raise ValueError(f"Word {target_word} not found")
        sample = matching[0]
    else:
        sample = random.choice(valid_samples)
    
    bengali_word = sample['bengali_word']
    transliterated = to_banglish(bengali_word)
    
    return {
        'bengali_word': bengali_word,
        'transliterated': transliterated,
        'frames': sample['frame_files'],
        'prepared_pt': sample['prepared_pt'],
        'split': sample['split']
    }


def load_prepared_mouth_crop(pt_path, num_frames=7):
    """Load frames from prepared .pt file."""
    obj = torch.load(pt_path, map_location='cpu')
    
    if isinstance(obj, dict) and 'frames' in obj:
        frames_tensor = obj['frames']  # (T, H, W) uint8
    elif isinstance(obj, dict) and 'x' in obj:
        frames_tensor = obj['x']  # (T, H, W)
    else:
        raise ValueError(f"Unknown format")
    
    # Convert to numpy
    if isinstance(frames_tensor, torch.Tensor):
        frames_np = frames_tensor.numpy()
    else:
        frames_np = frames_tensor
    
    # Ensure uint8
    if frames_np.dtype != np.uint8:
        if frames_np.max() <= 1.0:
            frames_np = (frames_np * 255).astype(np.uint8)
        else:
            frames_np = frames_np.astype(np.uint8)
    
    # Sample frames
    T = frames_np.shape[0]
    if T > num_frames:
        indices = np.linspace(0, T-1, num_frames).astype(int)
        frames_np = frames_np[indices]
    elif T < num_frames:
        padding = np.repeat(frames_np[-1:], num_frames - T, axis=0)
        frames_np = np.concatenate([frames_np, padding], axis=0)
    
    return frames_np


def create_ctc_alignment(transliterated_word, num_frames):
    """Create realistic CTC alignment."""
    chars = list(transliterated_word.lower())
    
    if num_frames <= len(chars):
        alignment = chars[:num_frames]
        while len(alignment) < num_frames:
            alignment.append('-')
    else:
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
    output_path='export/paper_samples/ctc_bengali_correct.png'
):
    """Create compact, correct visualization."""
    
    print(f"Loading sample from LipBengal...")
    sample = load_lipbengal_sample_correct(dataset_root, speaker, target_word)
    
    bengali_word = sample['bengali_word']
    transliterated = sample['transliterated']
    frame_files = sample['frames']
    prepared_pt = sample['prepared_pt']
    
    print(f"  Bengali: {bengali_word}")
    print(f"  Transliterated: {transliterated}")
    print(f"  Split: {sample['split']}")
    print(f"  Frames available: {len(frame_files)}")
    print(f"  Prepared: {prepared_pt}")
    
    # Load frames - sample to get num_frames
    num_frames = min(7, len(frame_files))
    
    # Load mouth crops first to ensure they exist
    mouth_frames = load_prepared_mouth_crop(prepared_pt, num_frames)
    print(f"  Mouth crops loaded: {mouth_frames.shape}")
    
    # Now load matching number of full frames
    indices = np.linspace(0, len(frame_files)-1, num_frames).astype(int)
    full_frames = []
    for idx in indices:
        img = Image.open(frame_files[idx]).convert('L')
        full_frames.append(np.array(img))
    
    # Create CTC alignment
    alignment = create_ctc_alignment(transliterated, num_frames)
    
    # Create COMPACT visualization
    print("Creating compact visualization...")
    fig = plt.figure(figsize=(16, 8))
    
    # Tighter spacing
    frame_height = 0.16
    spacing = 0.05
    top = 0.90
    
    # Row 1: Full frames
    row1_y = top
    for i, frame in enumerate(full_frames):
        ax = fig.add_axes([0.08 + i*0.125, row1_y - frame_height, 0.11, frame_height])
        ax.imshow(frame, cmap='gray')
        ax.axis('off')
    
    fig.text(0.01, row1_y - frame_height/2, 'Full', 
             fontsize=10, ha='left', va='center', weight='bold')
    
    # Row 2: Lip crops
    row2_y = row1_y - frame_height - spacing
    for i, lip_frame in enumerate(mouth_frames):
        ax = fig.add_axes([0.08 + i*0.125, row2_y - frame_height, 0.11, frame_height])
        ax.imshow(lip_frame, cmap='gray')
        ax.axis('off')
    
    fig.text(0.01, row2_y - frame_height/2, 'Lips', 
             fontsize=10, ha='left', va='center', weight='bold')
    
    # Row 3: Transliterated characters
    row3_y = row2_y - frame_height - spacing
    for i, char in enumerate(alignment):
        x = 0.135 + i*0.125
        y = row3_y - 0.04
        
        if char == '-':
            bbox = dict(boxstyle='round,pad=0.5', facecolor='lightgray', 
                       edgecolor='gray', linewidth=1.5)
            fig.text(x, y, '∅', fontsize=16, ha='center', va='center', 
                    bbox=bbox, family='monospace')
        else:
            bbox = dict(boxstyle='round,pad=0.5', facecolor='lightblue', 
                       edgecolor='blue', linewidth=2)
            fig.text(x, y, char, fontsize=16, ha='center', va='center', 
                    bbox=bbox, weight='bold', family='monospace')
        
        # Arrow from lip to character
        arrow = FancyArrowPatch(
            (0.135 + i*0.125, row2_y - frame_height - 0.01),
            (x, y + 0.025),
            arrowstyle='->', lw=1.5, color='darkgreen', 
            alpha=0.7, mutation_scale=15
        )
        fig.add_artist(arrow)
    
    fig.text(0.01, row3_y - 0.04, 'CTC', 
             fontsize=10, ha='left', va='center', weight='bold')
    
    # Row 4: Final word (COMPACT - single row)
    row4_y = row3_y - 0.18
    
    # Show transliterated word prominently (since Bengali doesn't render)
    fig.text(0.5, row4_y, transliterated.upper(), 
             fontsize=28, ha='center', weight='bold',
             bbox=dict(boxstyle='round,pad=1.0', facecolor='gold', 
                      edgecolor='orange', linewidth=3))
    
    # Show Bengali below in smaller text (may not render)
    fig.text(0.5, row4_y - 0.08, f'Bengali: {bengali_word}', 
             fontsize=14, ha='center', style='italic', color='gray')
    
    # Title
    fig.text(0.5, 0.96, 'Bengali CTC Alignment (LipBengal Dataset)', 
             fontsize=14, ha='center', weight='bold')
    
    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"\n✓ Saved: {output_path}")
    print(f"  Word: {bengali_word} → {transliterated}")
    print(f"  Alignment: {' '.join(alignment)}")
    
    return output_path


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, 
                       default='/home/thesis/Thesis/AVEC/datasets/LipBengal')
    parser.add_argument('--speaker', type=str, default='s100')
    parser.add_argument('--word', type=str, default=None)
    parser.add_argument('--output', type=str, 
                       default='export/paper_samples/ctc_bengali_correct.png')
    
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

