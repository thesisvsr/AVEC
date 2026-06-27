#!/usr/bin/env python3
"""
Create a visualization of the lip movement and transcript alignment process.

Shows:
1. Preprocessed lip frames (7-8 frames)
2. Arrows pointing to character predictions  
3. Final decoded word

Usage:
    python scripts/visualize_ctc_alignment.py --sample export/paper_samples/LRS2/sample_1_00001 --output export/ctc_alignment_visualization.png
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Tuple
import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    print("Error: Pillow is required. Install with: pip install Pillow")
    sys.exit(1)

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.patches import FancyArrowPatch
except ImportError:
    print("Error: matplotlib is required. Install with: pip install matplotlib")
    sys.exit(1)


def load_frames(sample_dir: Path, num_frames: int = 8) -> List[np.ndarray]:
    """Load preprocessed lip frames from a sample directory."""
    frames = []
    for i in range(1, num_frames + 1):
        frame_path = sample_dir / f"frame_{i:02d}.jpg"
        if not frame_path.exists():
            print(f"Warning: {frame_path} not found, skipping")
            continue
        img = Image.open(frame_path).convert('L')  # Convert to grayscale
        frames.append(np.array(img))
    return frames


def read_metadata(sample_dir: Path) -> dict:
    """Read metadata from the sample directory."""
    metadata_path = sample_dir / "metadata.txt"
    metadata = {}
    
    if metadata_path.exists():
        with open(metadata_path, 'r', encoding='utf-8') as f:
            for line in f:
                if ':' in line:
                    key, value = line.strip().split(':', 1)
                    metadata[key.strip()] = value.strip()
    
    return metadata


def simulate_ctc_alignment(word: str, num_frames: int) -> List[str]:
    """
    Simulate CTC alignment by distributing characters across frames.
    In real CTC, some frames may output blanks or repeated characters.
    """
    # For visualization purposes, we'll show a plausible alignment
    # Real CTC would have blanks between characters
    
    if not word:
        return [''] * num_frames
    
    # Create a realistic CTC alignment pattern
    # Some frames get blanks (-), some get characters, some get repeats
    alignment = []
    chars = list(word)
    
    # Calculate spacing
    chars_per_frame = len(chars) / num_frames
    
    char_idx = 0
    for frame_idx in range(num_frames):
        expected_char_idx = int((frame_idx + 0.5) * chars_per_frame)
        
        if char_idx < len(chars):
            if expected_char_idx == char_idx:
                # Emit character
                alignment.append(chars[char_idx])
                char_idx += 1
            elif expected_char_idx > char_idx:
                # Emit character and advance
                alignment.append(chars[char_idx])
                char_idx += 1
            else:
                # Emit blank or repeat
                if frame_idx > 0 and alignment[-1] != '-':
                    alignment.append(alignment[-1])  # Repeat
                else:
                    alignment.append('-')  # Blank
        else:
            alignment.append('-')  # Blank after all characters emitted
    
    return alignment


def create_alignment_visualization(
    frames: List[np.ndarray],
    alignment: List[str],
    word: str,
    output_path: Path,
    title: str = "Lip Movement → Character Alignment (CTC)"
):
    """
    Create a visualization showing frames, arrows, character predictions, and final word.
    """
    num_frames = len(frames)
    
    # Set up the figure
    fig_width = max(14, num_frames * 1.5)
    fig_height = 8
    fig = plt.figure(figsize=(fig_width, fig_height))
    
    # Create custom layout
    # Top: frames
    # Middle: arrows and characters
    # Bottom: final word
    
    # Calculate positions
    frame_height = 0.35
    frame_y_top = 0.55
    arrow_y = 0.35
    char_y = 0.25
    word_y = 0.05
    
    frame_spacing = 0.9 / num_frames
    frame_width = frame_spacing * 0.85
    
    # Add title
    fig.suptitle(title, fontsize=18, fontweight='bold', y=0.98)
    
    # Draw frames
    for i, frame in enumerate(frames):
        x_pos = 0.05 + i * frame_spacing + frame_spacing * 0.075
        ax = fig.add_axes([x_pos, frame_y_top, frame_width, frame_height])
        ax.imshow(frame, cmap='gray', aspect='auto')
        ax.set_title(f'Frame {i+1}', fontsize=10, pad=5)
        ax.axis('off')
    
    # Create axis for arrows and text
    ax_main = fig.add_axes([0, 0, 1, 1])
    ax_main.set_xlim(0, 1)
    ax_main.set_ylim(0, 1)
    ax_main.axis('off')
    
    # Draw arrows and character predictions
    for i in range(num_frames):
        x_center = 0.05 + i * frame_spacing + frame_spacing / 2
        
        # Draw arrow
        arrow = FancyArrowPatch(
            (x_center, frame_y_top - 0.02),
            (x_center, char_y + 0.08),
            arrowstyle='->,head_width=0.4,head_length=0.4',
            color='#2E86AB',
            linewidth=2.5,
            mutation_scale=20
        )
        ax_main.add_patch(arrow)
        
        # Draw character prediction box
        char = alignment[i]
        if char == '-':
            # Blank symbol (epsilon)
            display_char = 'ε'
            box_color = '#F4F4F4'
            text_color = '#888888'
            edge_color = '#CCCCCC'
        else:
            display_char = char
            box_color = '#A23B72'
            text_color = 'white'
            edge_color = '#7A2D56'
        
        # Character box
        box_width = frame_width * 0.8
        box_height = 0.08
        box_x = x_center - box_width / 2
        box_y = char_y
        
        rect = mpatches.FancyBboxPatch(
            (box_x, box_y),
            box_width,
            box_height,
            boxstyle="round,pad=0.01",
            facecolor=box_color,
            edgecolor=edge_color,
            linewidth=2
        )
        ax_main.add_patch(rect)
        
        # Character text
        ax_main.text(
            x_center,
            char_y + box_height / 2,
            display_char,
            ha='center',
            va='center',
            fontsize=16,
            fontweight='bold',
            color=text_color,
            family='monospace'
        )
    
    # Draw CTC collapse operation
    # Show unique characters after removing blanks and consecutive duplicates
    collapsed_chars = []
    prev = None
    for c in alignment:
        if c == '-':
            continue
        if c != prev:
            collapsed_chars.append(c)
            prev = c
    
    # Draw collapse indicator
    ax_main.text(
        0.5,
        char_y - 0.08,
        'CTC Collapse: Remove blanks (ε) & consecutive duplicates →',
        ha='center',
        va='center',
        fontsize=12,
        style='italic',
        color='#555555'
    )
    
    # Draw final word box
    word_box_width = 0.4
    word_box_height = 0.12
    word_box_x = 0.5 - word_box_width / 2
    word_box_y = word_y
    
    word_rect = mpatches.FancyBboxPatch(
        (word_box_x, word_box_y),
        word_box_width,
        word_box_height,
        boxstyle="round,pad=0.02",
        facecolor='#F18F01',
        edgecolor='#C87200',
        linewidth=3
    )
    ax_main.add_patch(word_rect)
    
    # Final word text
    ax_main.text(
        0.5,
        word_y + word_box_height / 2,
        f'Final Word: "{word}"',
        ha='center',
        va='center',
        fontsize=18,
        fontweight='bold',
        color='white'
    )
    
    # Add legend explaining the process
    legend_x = 0.02
    legend_y = 0.92
    legend_text = (
        "CTC Alignment Process:\n"
        "1. Each frame produces a character prediction\n"
        "2. Model can output blanks (ε) or repeated characters\n"
        "3. CTC decoding removes blanks and consecutive duplicates\n"
        "4. Remaining characters form the final word"
    )
    
    ax_main.text(
        legend_x,
        legend_y,
        legend_text,
        ha='left',
        va='top',
        fontsize=9,
        bbox=dict(
            boxstyle='round,pad=0.8',
            facecolor='#FFF9E6',
            edgecolor='#FFD700',
            linewidth=2,
            alpha=0.95
        ),
        family='sans-serif'
    )
    
    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved visualization to: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize CTC alignment for lip reading"
    )
    parser.add_argument(
        '--sample',
        type=str,
        required=True,
        help='Path to sample directory containing frames and metadata'
    )
    parser.add_argument(
        '--word',
        type=str,
        default=None,
        help='Word to visualize (if not in metadata). Example: "HELLO"'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='export/ctc_alignment_visualization.png',
        help='Output path for visualization'
    )
    parser.add_argument(
        '--num-frames',
        type=int,
        default=8,
        help='Number of frames to use (default: 8)'
    )
    parser.add_argument(
        '--title',
        type=str,
        default='Lip Movement → Character Alignment (CTC)',
        help='Title for the visualization'
    )
    
    args = parser.parse_args()
    
    sample_dir = Path(args.sample)
    if not sample_dir.exists():
        print(f"Error: Sample directory not found: {sample_dir}")
        sys.exit(1)
    
    # Load frames
    print(f"Loading frames from: {sample_dir}")
    frames = load_frames(sample_dir, num_frames=args.num_frames)
    
    if len(frames) == 0:
        print("Error: No frames found in sample directory")
        sys.exit(1)
    
    print(f"Loaded {len(frames)} frames")
    
    # Get word from metadata or argument
    metadata = read_metadata(sample_dir)
    word = args.word
    
    if word is None:
        # Try to extract word from metadata or directory name
        if 'Word' in metadata:
            word = metadata['Word']
        else:
            # Try to extract from directory name
            dir_name = sample_dir.name
            # Format: sample_1_s1_word or sample_1_00001
            parts = dir_name.split('_')
            if len(parts) >= 3:
                word = '_'.join(parts[2:])
            else:
                word = "EXAMPLE"
    
    print(f"Word: {word}")
    
    # Generate CTC alignment
    alignment = simulate_ctc_alignment(word, len(frames))
    print(f"Simulated CTC alignment: {alignment}")
    
    # Create visualization
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    create_alignment_visualization(
        frames=frames,
        alignment=alignment,
        word=word,
        output_path=output_path,
        title=args.title
    )
    
    print(f"\n✓ Visualization complete!")
    print(f"  Frames: {len(frames)}")
    print(f"  Word: {word}")
    print(f"  Alignment: {' '.join(alignment)}")
    print(f"  Output: {output_path.absolute()}")


if __name__ == '__main__':
    main()

