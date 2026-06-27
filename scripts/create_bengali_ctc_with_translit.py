#!/usr/bin/env python3
"""
Create Bengali CTC visualization showing the TRANSLITERATION PROCESS.
Flow: Frames → Lips → Bengali Word → Transliteration → CTC Alignment → Final Word
"""

import sys
from pathlib import Path
import torch
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import random

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.banglish_lookup import to_banglish


def load_lipbengal_sample_correct(dataset_root, speaker='s100', target_word=None):
    """Load a complete sample ensuring frames and prepared data match."""
    root = Path(dataset_root)
    speaker_dir = root / speaker
    
    if not speaker_dir.exists():
        raise ValueError(f"Speaker {speaker} not found")
    
    word_dirs = [d for d in speaker_dir.iterdir() if d.is_dir()]
    valid_samples = []
    
    for word_dir in word_dirs:
        bengali_word = word_dir.name
        frame_files = list(word_dir.glob("*.jpg"))
        
        if len(frame_files) == 0:
            continue
        
        # Check if prepared data exists
        for split in ['train', 'val', 'test']:
            prepared_dir = root / 'prepared' / split / speaker / bengali_word
            if prepared_dir.exists():
                pt_files = list(prepared_dir.glob("*.pt"))
                if pt_files:
                    for pt_file in pt_files:
                        try:
                            obj = torch.load(pt_file, map_location='cpu')
                            if isinstance(obj, dict) and ('frames' in obj or 'x' in obj):
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
        raise ValueError(f"No valid samples found")
    
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
        frames_tensor = obj['frames']
    elif isinstance(obj, dict) and 'x' in obj:
        frames_tensor = obj['x']
    else:
        raise ValueError(f"Unknown format")
    
    if isinstance(frames_tensor, torch.Tensor):
        frames_np = frames_tensor.numpy()
    else:
        frames_np = frames_tensor
    
    if frames_np.dtype != np.uint8:
        if frames_np.max() <= 1.0:
            frames_np = (frames_np * 255).astype(np.uint8)
        else:
            frames_np = frames_np.astype(np.uint8)
    
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


def create_bengali_ctc_with_transliteration(
    dataset_root='/home/thesis/Thesis/AVEC/datasets/LipBengal',
    speaker='s100',
    target_word=None,
    output_path='export/paper_samples/ctc_bengali_with_translit.png'
):
    """Create visualization showing transliteration process."""
    
    print(f"Loading sample...")
    sample = load_lipbengal_sample_correct(dataset_root, speaker, target_word)
    
    bengali_word = sample['bengali_word']
    transliterated = sample['transliterated']
    frame_files = sample['frames']
    prepared_pt = sample['prepared_pt']
    
    print(f"  Bengali: {bengali_word}")
    print(f"  Transliterated: {transliterated}")
    print(f"  Frames: {len(frame_files)}")
    
    num_frames = min(7, len(frame_files))
    
    # Load mouth crops
    mouth_frames = load_prepared_mouth_crop(prepared_pt, num_frames)
    
    # Load matching full frames
    indices = np.linspace(0, len(frame_files)-1, num_frames).astype(int)
    full_frames = []
    for idx in indices:
        img = Image.open(frame_files[idx]).convert('L')
        full_frames.append(np.array(img))
    
    # Create CTC alignment
    alignment = create_ctc_alignment(transliterated, num_frames)
    
    # Create COMPACT visualization with transliteration process
    print("Creating visualization...")
    fig = plt.figure(figsize=(15, 9))
    
    # Very compact spacing
    frame_height = 0.13
    spacing = 0.04
    top = 0.92
    
    # Row 1: Full frames (top)
    row1_y = top
    for i, frame in enumerate(full_frames):
        ax = fig.add_axes([0.06 + i*0.12, row1_y - frame_height, 0.10, frame_height])
        ax.imshow(frame, cmap='gray')
        ax.axis('off')
    
    fig.text(0.01, row1_y - frame_height/2, 'Full', 
             fontsize=9, ha='left', va='center', weight='bold', rotation=90)
    
    # Row 2: Lip crops
    row2_y = row1_y - frame_height - spacing
    for i, lip_frame in enumerate(mouth_frames):
        ax = fig.add_axes([0.06 + i*0.12, row2_y - frame_height, 0.10, frame_height])
        ax.imshow(lip_frame, cmap='gray')
        ax.axis('off')
    
    fig.text(0.01, row2_y - frame_height/2, 'Lips', 
             fontsize=9, ha='left', va='center', weight='bold', rotation=90)
    
    # Row 3: TRANSLITERATION PROCESS (NEW!)
    row3_y = row2_y - frame_height - spacing - 0.04
    
    # Bengali word label (text only, no Bengali characters to avoid rendering issues)
    bengali_x = 0.22
    fig.text(bengali_x, row3_y, 'Bengali Word', 
             fontsize=14, ha='center', weight='bold',
             bbox=dict(boxstyle='round,pad=0.6', facecolor='lightcoral', 
                      edgecolor='red', linewidth=2))
    
    # Arrow showing transliteration
    arrow = FancyArrowPatch(
        (bengali_x + 0.12, row3_y),
        (0.45, row3_y),
        arrowstyle='->', lw=3, color='purple', 
        alpha=0.8, mutation_scale=25
    )
    fig.add_artist(arrow)
    
    # "Transliteration" label on arrow
    fig.text(0.335, row3_y + 0.025, 'Transliteration', 
             fontsize=10, ha='center', weight='bold', 
             color='purple', style='italic',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='lavender', 
                      edgecolor='purple', linewidth=1, alpha=0.7))
    
    # Transliterated word in box
    translit_x = 0.58
    fig.text(translit_x, row3_y, transliterated.upper(), 
             fontsize=18, ha='center', weight='bold',
             bbox=dict(boxstyle='round,pad=0.6', facecolor='lightgreen', 
                      edgecolor='green', linewidth=2))
    
    # Labels (show Bengali in ASCII below)
    fig.text(bengali_x, row3_y - 0.04, f'({bengali_word})', 
             fontsize=9, ha='center', color='darkred', style='italic', family='monospace')
    fig.text(translit_x, row3_y - 0.04, 'Romanized', 
             fontsize=8, ha='center', color='gray', style='italic')
    
    # Row 4: CTC Alignment characters
    row4_y = row3_y - 0.14
    for i, char in enumerate(alignment):
        x = 0.11 + i*0.12
        y = row4_y
        
        if char == '-':
            bbox = dict(boxstyle='round,pad=0.4', facecolor='lightgray', 
                       edgecolor='gray', linewidth=1.5)
            fig.text(x, y, '∅', fontsize=14, ha='center', va='center', 
                    bbox=bbox, family='monospace')
        else:
            bbox = dict(boxstyle='round,pad=0.4', facecolor='lightblue', 
                       edgecolor='blue', linewidth=2)
            fig.text(x, y, char, fontsize=14, ha='center', va='center', 
                    bbox=bbox, weight='bold', family='monospace')
        
        # Arrow from lip to character
        arrow = FancyArrowPatch(
            (0.11 + i*0.12, row2_y - frame_height - 0.01),
            (x, y + 0.02),
            arrowstyle='->', lw=1.2, color='darkgreen', 
            alpha=0.6, mutation_scale=12
        )
        fig.add_artist(arrow)
    
    fig.text(0.01, row4_y, 'CTC', 
             fontsize=9, ha='left', va='center', weight='bold', rotation=90)
    
    # Row 5: Final decoded word
    row5_y = row4_y - 0.13
    fig.text(0.5, row5_y, transliterated.upper(), 
             fontsize=24, ha='center', weight='bold',
             bbox=dict(boxstyle='round,pad=0.8', facecolor='gold', 
                      edgecolor='orange', linewidth=3))
    
    # Show Bengali using monospace font to avoid rendering issues
    fig.text(0.5, row5_y - 0.055, f'Original: {bengali_word}', 
             fontsize=10, ha='center', style='italic', color='gray', family='monospace')
    
    # Title
    fig.text(0.5, 0.97, 'Bengali Lip Reading: Transliteration + CTC Alignment', 
             fontsize=13, ha='center', weight='bold')
    
    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"\n✓ Saved: {output_path}")
    print(f"  {bengali_word} → {transliterated}")
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
                       default='export/paper_samples/ctc_bengali_with_translit.png')
    
    args = parser.parse_args()
    
    try:
        create_bengali_ctc_with_transliteration(
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

