#!/usr/bin/env python3
"""
Create Bengali CTC visualization with SIDE-BY-SIDE layout:
LEFT: Frames → Lip crops (vertical)
RIGHT: Bengali → Transliteration (vertical)
MIDDLE: CTC alignment arrows connecting them
BOTTOM: Final output word
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


def create_bengali_ctc_side_by_side(
    dataset_root='/home/thesis/Thesis/AVEC/datasets/LipBengal',
    speaker='s100',
    target_word=None,
    output_path='export/paper_samples/ctc_bengali_sidebyside.png'
):
    """Create side-by-side visualization."""
    
    print(f"Loading sample...")
    sample = load_lipbengal_sample_correct(dataset_root, speaker, target_word)
    
    bengali_word = sample['bengali_word']
    transliterated = sample['transliterated']
    frame_files = sample['frames']
    prepared_pt = sample['prepared_pt']
    
    print(f"  Bengali: {bengali_word}")
    print(f"  Transliterated: {transliterated}")
    
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
    chars = list(transliterated.lower())
    
    print("Creating side-by-side visualization...")
    fig = plt.figure(figsize=(14, 10))
    
    # Layout parameters
    left_x = 0.05  # Left column (frames)
    right_x = 0.75  # Right column (transliteration)
    middle_x = 0.45  # Middle area for alignment
    
    frame_size = 0.12
    spacing = 0.02
    
    # Title
    fig.text(0.5, 0.96, 'Bengali Lip Reading: Visual + Transliteration + Alignment', 
             fontsize=12, ha='center', weight='bold')
    
    # LEFT SIDE: Full Frames
    fig.text(left_x - 0.02, 0.85, 'Visual', fontsize=11, ha='left', weight='bold', rotation=90, va='center')
    
    top_y = 0.88
    for i, frame in enumerate(full_frames):
        y = top_y - i * (frame_size + spacing)
        ax = fig.add_axes([left_x, y - frame_size, frame_size, frame_size])
        ax.imshow(frame, cmap='gray')
        ax.axis('off')
        if i == 0:
            ax.set_title('Full', fontsize=9, pad=2)
    
    # LEFT SIDE: Lip Crops
    lip_x = left_x + frame_size + 0.03
    for i, lip_frame in enumerate(mouth_frames):
        y = top_y - i * (frame_size + spacing)
        ax = fig.add_axes([lip_x, y - frame_size, frame_size, frame_size])
        ax.imshow(lip_frame, cmap='gray')
        ax.axis('off')
        if i == 0:
            ax.set_title('Lips', fontsize=9, pad=2)
    
    # RIGHT SIDE: Transliteration Flow
    fig.text(right_x + 0.10, 0.85, 'Transliteration', fontsize=11, ha='right', weight='bold', rotation=90, va='center')
    
    # Bengali word at top (label only to avoid rendering issues)
    bengali_y = top_y - frame_size/2
    fig.text(right_x, bengali_y, 'Bengali Word', 
             fontsize=13, ha='center', weight='bold',
             bbox=dict(boxstyle='round,pad=0.6', facecolor='lightcoral', 
                      edgecolor='red', linewidth=2))
    fig.text(right_x, bengali_y - 0.05, f'{bengali_word}', 
             fontsize=9, ha='center', color='darkred', style='italic', family='monospace')
    
    # Downward arrow
    arrow_y = bengali_y - 0.08
    arrow = FancyArrowPatch(
        (right_x, bengali_y - 0.06),
        (right_x, arrow_y - 0.01),
        arrowstyle='->', lw=2.5, color='purple', 
        alpha=0.8, mutation_scale=20
    )
    fig.add_artist(arrow)
    fig.text(right_x + 0.04, arrow_y - 0.025, 'Transliterate', 
             fontsize=9, ha='left', weight='bold', color='purple', style='italic')
    
    # English characters (vertical)
    char_start_y = arrow_y - 0.03
    for i, char in enumerate(chars):
        y = char_start_y - i * 0.08
        fig.text(right_x, y, char, 
                fontsize=16, ha='center', weight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', 
                         edgecolor='green', linewidth=2))
    
    fig.text(right_x, char_start_y + 0.04, 'English Characters', 
             fontsize=9, ha='center', style='italic', color='gray')
    
    # MIDDLE: CTC Alignment arrows
    fig.text(middle_x, 0.92, 'CTC Alignment', fontsize=10, ha='center', 
             weight='bold', color='darkblue')
    
    for i in range(num_frames):
        lip_y = top_y - i * (frame_size + spacing) - frame_size/2
        char = alignment[i]
        
        # Find corresponding character position in transliteration
        if char != '-' and char in chars:
            char_idx = chars.index(char)
            char_y = char_start_y - char_idx * 0.08
            
            # Arrow from lip to character
            arrow = FancyArrowPatch(
                (lip_x + frame_size + 0.01, lip_y),
                (right_x - 0.04, char_y),
                arrowstyle='->', lw=1.5, color='darkblue', 
                alpha=0.6, mutation_scale=12,
                linestyle='--' if char == '-' else '-'
            )
            fig.add_artist(arrow)
            
            # Show aligned character in middle
            fig.text(middle_x, lip_y, char if char != '-' else '∅', 
                    fontsize=12, ha='center', weight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', 
                             facecolor='lightblue' if char != '-' else 'lightgray',
                             edgecolor='blue' if char != '-' else 'gray', 
                             linewidth=1.5))
    
    # BOTTOM: Final output word
    final_y = 0.08
    fig.text(0.5, final_y, transliterated.upper(), 
             fontsize=26, ha='center', weight='bold',
             bbox=dict(boxstyle='round,pad=1.0', facecolor='gold', 
                      edgecolor='orange', linewidth=3))
    
    fig.text(0.5, final_y - 0.06, 'Final Decoded Word', 
             fontsize=10, ha='center', style='italic', color='gray')
    
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
                       default='export/paper_samples/ctc_bengali_sidebyside.png')
    
    args = parser.parse_args()
    
    try:
        create_bengali_ctc_side_by_side(
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

