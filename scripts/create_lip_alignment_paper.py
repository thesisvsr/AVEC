#!/usr/bin/env python3
"""
Create publication-quality lip alignment visualization.
Optimized for academic papers with larger, clearer frames.
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
from matplotlib.patches import FancyBboxPatch
import random
import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.banglish_lookup import to_banglish


def find_bengali_font():
    """Try to find a Bengali font on the system."""
    font_paths = [
        '/usr/share/fonts/truetype/noto/NotoSansBengali-Regular.ttf',
        '/usr/share/fonts/truetype/fonts-beng-extra/Lohit-Bengali.ttf',
        '/usr/share/fonts/truetype/fonts-beng/Lohit-Bengali.ttf',
        '/usr/share/fonts/truetype/noto/NotoSerifBengali-Regular.ttf',
        '/System/Library/Fonts/Supplemental/Kohinoor.ttc',
    ]
    
    for font_path in font_paths:
        if Path(font_path).exists():
            return font_path
    return None


def create_bengali_text_image(text, font_size=80):
    """Create an image with Bengali text using PIL."""
    font_path = find_bengali_font()
    
    if font_path:
        try:
            font = ImageFont.truetype(font_path, font_size)
        except:
            font = ImageFont.load_default()
    else:
        font = ImageFont.load_default()
    
    # Create a temporary image to measure text size
    dummy_img = Image.new('RGB', (1, 1))
    draw = ImageDraw.Draw(dummy_img)
    
    # Get text bounding box
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    
    # Create the actual image with some padding
    padding = 30
    img = Image.new('RGBA', (text_width + padding*2, text_height + padding*2), (255, 255, 255, 0))
    draw = ImageDraw.Draw(img)
    
    # Draw the text
    draw.text((padding, padding), text, font=font, fill=(0, 0, 0, 255))
    
    return img


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


def sample_frames(frames, num_samples=10):
    """Sample frames evenly from the sequence."""
    total_frames = len(frames)
    if total_frames <= num_samples:
        return frames, list(range(total_frames))
    
    # Sample evenly across the sequence
    indices = np.linspace(0, total_frames - 1, num_samples).astype(int)
    sampled_frames = [frames[i] for i in indices]
    
    return sampled_frames, indices.tolist()


def create_character_alignment(word, frame_indices, total_frames):
    """Create alignment between characters and sampled frames.
    
    Maps characters to their positions in the original sequence,
    then finds which sampled frames they correspond to.
    """
    chars = list(word.lower())
    alignment = []
    
    if total_frames <= len(chars):
        # More characters than frames - distribute what we can
        char_positions = list(range(len(chars)))[:total_frames]
    else:
        # More frames than characters - spread characters evenly
        char_positions = np.linspace(0, total_frames - 1, len(chars)).astype(int).tolist()
    
    # For each sampled frame, find if there's a character nearby
    for frame_idx in frame_indices:
        char_found = None
        # Check if this frame position matches any character position
        for i, char_pos in enumerate(char_positions):
            if abs(frame_idx - char_pos) <= 2:  # Within 2 frames
                char_found = chars[i]
                break
        
        alignment.append(char_found if char_found else ' ')
    
    return alignment


def create_visualization(
    bengali_word,
    transliterated_word,
    frames,
    frame_indices,
    alignment,
    output_path,
    frames_per_row=5
):
    """Create publication-quality visualization with larger frames."""
    
    num_frames = len(frames)
    num_rows = (num_frames + frames_per_row - 1) // frames_per_row
    
    # Calculate figure size for larger, clearer frames
    frame_width = 2.2  # Larger frames
    fig_width = frames_per_row * frame_width + 1.5
    fig_height = num_rows * 3.0 + 2.8
    
    fig = plt.figure(figsize=(fig_width, fig_height))
    
    # More space for header
    gs = gridspec.GridSpec(
        num_rows + 1, 1,
        height_ratios=[0.7] + [1.0] * num_rows,
        hspace=0.35
    )
    
    # Top section: Word information
    ax_top = fig.add_subplot(gs[0, 0])
    ax_top.axis('off')
    
    # Try to create Bengali text as image
    try:
        bengali_img = create_bengali_text_image(bengali_word, font_size=70)
        bengali_array = np.array(bengali_img)
        
        # Display Bengali text as image
        ax_bengali = fig.add_axes([0.15, 0.83, 0.7, 0.13])
        ax_bengali.imshow(bengali_array)
        ax_bengali.axis('off')
        
        # Arrow
        ax_arrow = fig.add_axes([0.45, 0.78, 0.1, 0.05])
        ax_arrow.axis('off')
        ax_arrow.text(0.5, 0.5, '↓', fontsize=32, ha='center', va='center', 
                     fontweight='bold')
        
        # Transliteration
        ax_trans = fig.add_axes([0.15, 0.70, 0.7, 0.09])
        ax_trans.axis('off')
        ax_trans.text(0.5, 0.5, f'"{transliterated_word}"',
                     fontsize=36, ha='center', va='center',
                     fontweight='bold', color='#2E86AB', style='italic')
    except Exception as e:
        print(f"Warning: Could not render Bengali text: {e}")
        ax_top.text(0.5, 0.7, bengali_word, 
                   fontsize=40, ha='center', va='center',
                   fontweight='bold')
        ax_top.text(0.5, 0.4, '↓', 
                   fontsize=28, ha='center', va='center')
        ax_top.text(0.5, 0.1, f'"{transliterated_word}"', 
                   fontsize=36, ha='center', va='center',
                   fontweight='bold', color='#2E86AB', style='italic')
        ax_top.set_xlim(0, 1)
        ax_top.set_ylim(0, 1)
    
    # Frame sections - larger spacing
    base_y = 0.60
    row_height = 0.50 / num_rows if num_rows > 0 else 0.50
    
    for row in range(num_rows):
        start_idx = row * frames_per_row
        end_idx = min(start_idx + frames_per_row, num_frames)
        frames_in_row = end_idx - start_idx
        
        for col in range(frames_in_row):
            idx = start_idx + col
            
            # Calculate position with more space
            x_start = 0.05
            x_span = 0.90
            x_offset = col / frames_per_row
            width = x_span / frames_per_row * 0.92
            
            x_pos = x_start + x_offset * x_span
            
            # Character label - larger
            ax_char = fig.add_axes([
                x_pos,
                base_y - row * row_height,
                width,
                0.08
            ])
            ax_char.axis('off')
            
            # Display character
            char = alignment[idx]
            if char and char != ' ':
                char_display = char
                char_color = '#E63946'
                
                # Background box
                bbox = FancyBboxPatch(
                    (0.15, 0.1), 0.7, 0.8,
                    boxstyle="round,pad=0.08",
                    transform=ax_char.transAxes,
                    facecolor='#FFE5E5',
                    edgecolor=char_color,
                    linewidth=2.5,
                    zorder=1
                )
                ax_char.add_patch(bbox)
                
                ax_char.text(0.5, 0.5, char_display,
                            fontsize=32, ha='center', va='center',
                            fontweight='bold', color=char_color,
                            fontfamily='monospace',
                            zorder=2)
            
            # Frame - much larger
            ax_frame = fig.add_axes([
                x_pos,
                base_y - row * row_height - 0.42,
                width,
                0.32
            ])
            ax_frame.axis('off')
            
            # Display frame with border
            frame_img = frames[idx]
            ax_frame.imshow(frame_img, cmap='gray', aspect='equal', 
                          interpolation='bilinear')
            
            # Thicker border
            for spine in ax_frame.spines.values():
                spine.set_visible(True)
                spine.set_color('#999')
                spine.set_linewidth(2)
            
            # Frame number (original index)
            ax_frame.text(0.5, -0.15, f'Frame {frame_indices[idx]+1}',
                         fontsize=12, ha='center', va='top',
                         transform=ax_frame.transAxes,
                         color='#555',
                         fontweight='600')
    
    plt.savefig(output_path, dpi=200, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"\n✅ Visualization saved to: {output_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Create publication-quality lip alignment visualization'
    )
    parser.add_argument('--dataset', type=str,
                       default='/home/thesis/Thesis/AVEC/datasets/LipBengal',
                       help='Path to LipBengal dataset')
    parser.add_argument('--speaker', type=str, default='s100',
                       help='Speaker ID (e.g., s100)')
    parser.add_argument('--word', type=str, default=None,
                       help='Specific Bengali word to visualize (optional)')
    parser.add_argument('--output', type=str,
                       default='export/paper_samples/lip_alignment_paper.png',
                       help='Output image path')
    parser.add_argument('--num-frames', type=int, default=10,
                       help='Number of frames to display (default: 10)')
    parser.add_argument('--frames-per-row', type=int, default=5,
                       help='Frames per row (default: 5)')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("PUBLICATION-QUALITY LIP ALIGNMENT VISUALIZATION")
    print("="*70)
    
    # Check for Bengali font
    font_path = find_bengali_font()
    if font_path:
        print(f"\n✓ Found Bengali font: {font_path}")
    else:
        print(f"\n⚠ No Bengali font found, will use fallback rendering")
    
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
    all_frames = load_prepared_frames(sample['prepared_pt'])
    total_frames = len(all_frames)
    print(f"   Total frames available: {total_frames}")
    
    # Sample frames for display
    print(f"\n✂️  Sampling {args.num_frames} frames for visualization...")
    sampled_frames, frame_indices = sample_frames(all_frames, args.num_frames)
    print(f"   Selected frame indices: {frame_indices}")
    print(f"   Frame shape: {sampled_frames[0].shape}")
    
    # Create alignment
    print(f"\n🔗 Creating character-to-frame alignment...")
    alignment = create_character_alignment(transliterated, frame_indices, total_frames)
    
    # Display alignment
    print(f"\n   Character alignment:")
    chars = list(transliterated.lower())
    print(f"   Word characters: {' '.join(chars)}")
    print(f"   Frame alignment: {' '.join([c if c and c != ' ' else '·' for c in alignment])}")
    
    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create visualization
    print(f"\n🎨 Creating visualization...")
    create_visualization(
        bengali_word=bengali_word,
        transliterated_word=transliterated,
        frames=sampled_frames,
        frame_indices=frame_indices,
        alignment=alignment,
        output_path=output_path,
        frames_per_row=args.frames_per_row
    )
    
    # Save metadata
    metadata = {
        'bengali_word': bengali_word,
        'transliterated_word': transliterated,
        'speaker': args.speaker,
        'split': sample['split'],
        'total_frames': total_frames,
        'displayed_frames': len(sampled_frames),
        'frame_indices': frame_indices,
        'characters': list(transliterated.lower()),
        'alignment': alignment,
        'frame_to_character': [
            {
                'display_index': i+1,
                'original_frame': frame_indices[i]+1,
                'character': alignment[i] if alignment[i] and alignment[i] != ' ' else 'blank'
            }
            for i in range(len(sampled_frames))
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
    print(f"\nDisplaying {len(sampled_frames)} out of {total_frames} frames")
    print(f"for optimal clarity in academic publication.")
    print()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

