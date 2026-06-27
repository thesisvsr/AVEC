#!/usr/bin/env python3
"""
Create lip alignment visualization showing:
1. Bangla word
2. English transliteration
3. All preprocessed lip frames
4. Character-to-frame alignment
"""

import sys
from pathlib import Path
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import random
import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.banglish_lookup import to_banglish


def load_lipbengal_sample(dataset_root, speaker='s100', target_word=None):
    """Load a complete sample from LipBengal dataset."""
    root = Path(dataset_root)
    speaker_dir = root / speaker
    
    if not speaker_dir.exists():
        raise ValueError(f"Speaker {speaker} not found")
    
    word_dirs = [d for d in speaker_dir.iterdir() if d.is_dir()]
    valid_samples = []
    
    for word_dir in word_dirs:
        bengali_word = word_dir.name
        frame_files = sorted(list(word_dir.glob("*.jpg")))
        
        if len(frame_files) == 0:
            continue
        
        # Find prepared file
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
        raise ValueError(f"No valid samples found for speaker {speaker}")
    
    if target_word:
        matching = [s for s in valid_samples if s['bengali_word'] == target_word]
        if not matching:
            available = [s['bengali_word'] for s in valid_samples[:5]]
            raise ValueError(f"Word '{target_word}' not found. Available samples: {available}")
        sample = matching[0]
    else:
        sample = random.choice(valid_samples)
    
    return sample


def load_prepared_frames(pt_path):
    """Load all frames from prepared .pt file."""
    obj = torch.load(pt_path, map_location='cpu')
    
    if isinstance(obj, dict) and 'frames' in obj:
        frames_tensor = obj['frames']
    elif isinstance(obj, dict) and 'x' in obj:
        frames_tensor = obj['x']
    else:
        raise ValueError(f"Unknown format in {pt_path}")
    
    if isinstance(frames_tensor, torch.Tensor):
        frames_np = frames_tensor.numpy()
    else:
        frames_np = frames_tensor
    
    # Convert to uint8 if needed
    if frames_np.dtype != np.uint8:
        if frames_np.max() <= 1.0:
            frames_np = (frames_np * 255).astype(np.uint8)
        else:
            frames_np = frames_np.astype(np.uint8)
    
    return frames_np


def create_character_alignment(word, num_frames):
    """Create alignment between characters and frames.
    
    Distributes characters evenly across frames.
    """
    chars = list(word.lower())
    
    if num_frames <= len(chars):
        # More characters than frames - show one char per frame
        alignment = chars[:num_frames]
    else:
        # More frames than characters - spread characters across frames
        alignment = [' '] * num_frames
        
        # Distribute characters evenly
        positions = np.linspace(0, num_frames-1, len(chars)).astype(int)
        
        for i, char in enumerate(chars):
            pos = positions[i]
            alignment[pos] = char
    
    return alignment


def create_visualization(
    bengali_word,
    transliterated_word,
    frames,
    alignment,
    output_path,
    max_frames_per_row=10
):
    """Create the complete visualization."""
    
    num_frames = len(frames)
    num_rows = (num_frames + max_frames_per_row - 1) // max_frames_per_row
    
    # Calculate figure size
    frame_size = 1.2
    fig_width = min(num_frames, max_frames_per_row) * frame_size + 1
    fig_height = num_rows * (frame_size + 0.5) + 2
    
    fig = plt.figure(figsize=(fig_width, fig_height))
    gs = gridspec.GridSpec(
        num_rows + 1, 1,
        height_ratios=[0.4] + [1.0] * num_rows,
        hspace=0.3
    )
    
    # Top section: Word information
    ax_top = fig.add_subplot(gs[0, 0])
    ax_top.axis('off')
    
    # Display text information
    text_y = 0.5
    ax_top.text(0.5, text_y, bengali_word, 
                fontsize=32, ha='center', va='center',
                fontweight='bold', fontfamily='DejaVu Sans')
    ax_top.text(0.5, text_y - 0.4, '↓', 
                fontsize=20, ha='center', va='center')
    ax_top.text(0.5, text_y - 0.8, f'"{transliterated_word}"', 
                fontsize=28, ha='center', va='center',
                fontweight='bold', color='#2E86AB', style='italic')
    
    ax_top.set_xlim(0, 1)
    ax_top.set_ylim(-1, 1)
    
    # Frame sections
    for row in range(num_rows):
        ax_frames = fig.add_subplot(gs[row + 1, 0])
        ax_frames.axis('off')
        
        start_idx = row * max_frames_per_row
        end_idx = min(start_idx + max_frames_per_row, num_frames)
        frames_in_row = end_idx - start_idx
        
        for col in range(frames_in_row):
            idx = start_idx + col
            
            # Calculate position
            x_offset = col / frames_in_row
            width = 0.95 / frames_in_row
            
            # Create axes for character label
            ax_char = fig.add_axes([
                0.05 + x_offset + width * 0.05,
                0.73 - row * 0.45,
                width * 0.9,
                0.08
            ])
            ax_char.axis('off')
            
            # Display character
            char = alignment[idx]
            if char == ' ':
                char_display = ''
            else:
                char_display = char
            
            ax_char.text(0.5, 0.5, char_display,
                        fontsize=24, ha='center', va='center',
                        fontweight='bold', color='#E63946',
                        fontfamily='monospace')
            
            # Create axes for frame
            ax_frame = fig.add_axes([
                0.05 + x_offset + width * 0.05,
                0.30 - row * 0.45,
                width * 0.9,
                0.38
            ])
            ax_frame.axis('off')
            
            # Display frame
            frame_img = frames[idx]
            ax_frame.imshow(frame_img, cmap='gray', aspect='auto')
            
            # Add frame number
            ax_frame.text(0.5, -0.1, f'{idx+1}',
                         fontsize=10, ha='center', va='top',
                         transform=ax_frame.transAxes,
                         color='#555')
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"\n✅ Visualization saved to: {output_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Create lip alignment visualization from LipBengal dataset'
    )
    parser.add_argument('--dataset', type=str,
                       default='/home/thesis/Thesis/AVEC/datasets/LipBengal',
                       help='Path to LipBengal dataset')
    parser.add_argument('--speaker', type=str, default='s100',
                       help='Speaker ID (e.g., s100)')
    parser.add_argument('--word', type=str, default=None,
                       help='Specific Bengali word to visualize (optional)')
    parser.add_argument('--output', type=str,
                       default='export/paper_samples/lip_alignment_full.png',
                       help='Output image path')
    parser.add_argument('--max-frames-per-row', type=int, default=10,
                       help='Maximum number of frames per row')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("LIP ALIGNMENT VISUALIZATION")
    print("="*70)
    
    # Load sample
    print(f"\n📂 Loading sample from speaker: {args.speaker}")
    if args.word:
        print(f"   Target word: {args.word}")
    
    sample = load_lipbengal_sample(args.dataset, args.speaker, args.word)
    
    bengali_word = sample['bengali_word']
    transliterated = to_banglish(bengali_word)
    
    print(f"\n📝 Word Information:")
    print(f"   Bengali: {bengali_word}")
    print(f"   English: {transliterated}")
    print(f"   Split: {sample['split']}")
    
    # Load frames
    print(f"\n🎬 Loading preprocessed frames...")
    frames = load_prepared_frames(sample['prepared_pt'])
    num_frames = len(frames)
    print(f"   Total frames: {num_frames}")
    print(f"   Frame shape: {frames[0].shape}")
    
    # Create alignment
    print(f"\n🔗 Creating character-to-frame alignment...")
    alignment = create_character_alignment(transliterated, num_frames)
    
    # Display alignment
    print(f"\n   Character alignment:")
    chars = list(transliterated.lower())
    print(f"   Word characters: {' '.join(chars)}")
    print(f"   Frame alignment: {' '.join([c if c != ' ' else '·' for c in alignment])}")
    
    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create visualization
    print(f"\n🎨 Creating visualization...")
    create_visualization(
        bengali_word=bengali_word,
        transliterated_word=transliterated,
        frames=frames,
        alignment=alignment,
        output_path=output_path,
        max_frames_per_row=args.max_frames_per_row
    )
    
    # Save metadata
    metadata = {
        'bengali_word': bengali_word,
        'transliterated_word': transliterated,
        'speaker': args.speaker,
        'split': sample['split'],
        'num_frames': num_frames,
        'characters': list(transliterated.lower()),
        'alignment': alignment,
        'frame_to_character': [
            {'frame': i+1, 'character': alignment[i] if alignment[i] != ' ' else 'blank'}
            for i in range(num_frames)
        ]
    }
    
    metadata_path = output_path.with_suffix('.json')
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"📄 Metadata saved to: {metadata_path}")
    
    print("\n" + "="*70)
    print("✅ COMPLETE!")
    print("="*70)
    print(f"\nVisualization: {output_path}")
    print(f"Metadata: {metadata_path}")
    print()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
