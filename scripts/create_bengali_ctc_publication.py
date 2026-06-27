#!/usr/bin/env python3
"""
Create publication-quality Bengali CTC visualization with correct flow:
1. Video frames → Lip frames (cropping)
2. Bengali word → Transliteration → English word → Individual characters
3. Alignment: Lip frames ↔ English characters
4. Final output
"""

import sys
from pathlib import Path
import torch
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle
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


def create_publication_bengali_ctc(
    dataset_root='/home/thesis/Thesis/AVEC/datasets/LipBengal',
    speaker='s100',
    target_word=None,
    output_path='export/paper_samples/ctc_bengali_final.png'
):
    """Create publication-quality visualization with correct flow."""
    
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
    
    print("Creating publication-quality visualization...")
    fig = plt.figure(figsize=(16, 11))
    
    # Layout parameters
    frame_w = 0.11
    frame_h = 0.10
    spacing_x = 0.015
    
    # STEP 1: Video Frames (Top)
    step1_y = 0.90
    step1_x = 0.08
    
    fig.text(0.5, 0.96, 'Bengali Lip Reading with Transliteration and CTC Alignment', 
             fontsize=14, ha='center', weight='bold')
    
    fig.text(step1_x - 0.03, step1_y - frame_h/2, 'Video\nFrames', 
             fontsize=9, ha='center', va='center', weight='bold')
    
    for i, frame in enumerate(full_frames):
        x = step1_x + i * (frame_w + spacing_x)
        ax = fig.add_axes([x, step1_y - frame_h, frame_w, frame_h])
        ax.imshow(frame, cmap='gray')
        ax.axis('off')
    
    # Arrow down: Cropping
    arrow_y1 = step1_y - frame_h - 0.02
    for i in range(num_frames):
        x = step1_x + i * (frame_w + spacing_x) + frame_w/2
        arrow = FancyArrowPatch(
            (x, step1_y - frame_h - 0.01),
            (x, arrow_y1 - 0.01),
            arrowstyle='->', lw=1.5, color='darkgreen', 
            alpha=0.7, mutation_scale=12
        )
        fig.add_artist(arrow)
    
    fig.text(0.5, arrow_y1 - 0.015, 'Crop Lip Region', 
             fontsize=9, ha='center', style='italic', color='darkgreen', weight='bold')
    
    # STEP 2: Lip Frames (After cropping)
    step2_y = arrow_y1 - 0.06
    
    fig.text(step1_x - 0.03, step2_y - frame_h/2, 'Lip\nFrames', 
             fontsize=9, ha='center', va='center', weight='bold')
    
    for i, lip_frame in enumerate(mouth_frames):
        x = step1_x + i * (frame_w + spacing_x)
        ax = fig.add_axes([x, step2_y - frame_h, frame_w, frame_h])
        ax.imshow(lip_frame, cmap='gray')
        ax.axis('off')
    
    # STEP 3: Bengali Word → Transliteration → English Word → Individual Characters
    step3_y = step2_y - frame_h - 0.12
    
    # Bengali word
    bengali_x = 0.15
    fig.text(bengali_x, step3_y, 'Bengali Script', 
             fontsize=11, ha='center', weight='bold',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightcoral', 
                      edgecolor='red', linewidth=2))
    fig.text(bengali_x, step3_y - 0.04, f'{bengali_word}', 
             fontsize=8, ha='center', color='darkred', style='italic', family='monospace')
    
    # Arrow: Transliteration
    arrow1 = FancyArrowPatch(
        (bengali_x + 0.08, step3_y),
        (0.32, step3_y),
        arrowstyle='->', lw=2.5, color='purple', 
        alpha=0.8, mutation_scale=20
    )
    fig.add_artist(arrow1)
    fig.text(0.235, step3_y + 0.025, 'Transliterate', 
             fontsize=9, ha='center', weight='bold', color='purple', style='italic')
    
    # English word
    english_x = 0.42
    fig.text(english_x, step3_y, transliterated.upper(), 
             fontsize=13, ha='center', weight='bold',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', 
                      edgecolor='green', linewidth=2))
    fig.text(english_x, step3_y - 0.04, 'Romanized Word', 
             fontsize=8, ha='center', color='gray', style='italic')
    
    # Arrow: Break into characters
    arrow2 = FancyArrowPatch(
        (english_x + 0.08, step3_y),
        (0.60, step3_y),
        arrowstyle='->', lw=2.5, color='blue', 
        alpha=0.8, mutation_scale=20
    )
    fig.add_artist(arrow2)
    fig.text(0.51, step3_y + 0.025, 'Break into Characters', 
             fontsize=9, ha='center', weight='bold', color='blue', style='italic')
    
    # Individual characters
    char_start_x = 0.68
    for i, char in enumerate(chars):
        x = char_start_x + i * 0.055
        fig.text(x, step3_y, char, 
                fontsize=12, ha='center', weight='bold',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', 
                         edgecolor='orange', linewidth=1.5))
    
    # STEP 4: CTC Alignment (Lip frames ↔ Characters)
    step4_y = step3_y - 0.16
    
    fig.text(0.5, step4_y + 0.07, 'CTC Alignment', 
             fontsize=11, ha='center', weight='bold', color='darkblue',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='lightcyan', 
                      edgecolor='darkblue', linewidth=2, alpha=0.8))
    
    for i in range(num_frames):
        lip_x = step1_x + i * (frame_w + spacing_x) + frame_w/2
        lip_y_pos = step2_y - frame_h
        
        char = alignment[i]
        
        # Position for aligned character
        aligned_y = step4_y
        
        # Draw aligned character box
        if char == '-':
            fig.text(lip_x, aligned_y, '∅', 
                    fontsize=13, ha='center', weight='bold',
                    bbox=dict(boxstyle='round,pad=0.4', facecolor='lightgray', 
                             edgecolor='gray', linewidth=1.5))
        else:
            fig.text(lip_x, aligned_y, char, 
                    fontsize=13, ha='center', weight='bold',
                    bbox=dict(boxstyle='round,pad=0.4', facecolor='lightblue', 
                             edgecolor='blue', linewidth=2))
        
        # Arrow from lip frame to aligned character
        arrow = FancyArrowPatch(
            (lip_x, lip_y_pos - 0.01),
            (lip_x, aligned_y + 0.025),
            arrowstyle='->', lw=1.5, color='darkblue', 
            alpha=0.6, mutation_scale=12
        )
        fig.add_artist(arrow)
        
        # Arrow from character breakdown to aligned character (if char exists)
        if char != '-' and char in chars:
            char_idx = chars.index(char)
            char_x = char_start_x + char_idx * 0.055
            
            arrow = FancyArrowPatch(
                (char_x, step3_y - 0.025),
                (lip_x, aligned_y + 0.025),
                arrowstyle='->', lw=1.2, color='orange', 
                alpha=0.5, mutation_scale=10, linestyle='--'
            )
            fig.add_artist(arrow)
    
    # STEP 5: Final Output
    final_y = step4_y - 0.15
    
    fig.text(0.5, final_y, transliterated.upper(), 
             fontsize=22, ha='center', weight='bold',
             bbox=dict(boxstyle='round,pad=0.9', facecolor='gold', 
                      edgecolor='orange', linewidth=3))
    
    fig.text(0.5, final_y - 0.055, 'Final Decoded Word', 
             fontsize=10, ha='center', style='italic', color='gray', weight='bold')
    
    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"\n✓ Publication-quality visualization saved: {output_path}")
    print(f"  Bengali: {bengali_word}")
    print(f"  Transliterated: {transliterated}")
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
                       default='export/paper_samples/ctc_bengali_final.png')
    
    args = parser.parse_args()
    
    try:
        create_publication_bengali_ctc(
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

