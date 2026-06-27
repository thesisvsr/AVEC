#!/usr/bin/env python3
"""
Enhanced CTC Alignment Visualization for Paper

Shows realistic CTC alignment patterns with blanks interspersed.
Demonstrates how the same character can span multiple frames.

Usage:
    python scripts/visualize_ctc_alignment_v2.py --sample export/paper_samples/LRS2/sample_1_00001 --alignment "- H - E E L L O" --word "HELLO"
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np

try:
    from PIL import Image
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


def parse_alignment(alignment_str: str) -> List[str]:
    """Parse alignment string. Use '-' for blank/epsilon."""
    return alignment_str.strip().split()


def ctc_collapse(alignment: List[str]) -> str:
    """Apply CTC collapse: remove blanks and consecutive duplicates.
    
    CTC decoding works in two steps:
    1. Remove consecutive duplicates (including from the original sequence)
    2. Remove blanks
    """
    # Step 1: Remove consecutive duplicates from original sequence
    collapsed = []
    prev = None
    for char in alignment:
        if char != prev:
            collapsed.append(char)
        prev = char
    
    # Step 2: Remove blanks
    result = [c for c in collapsed if c != '-']
    
    return ''.join(result)


def create_paper_ready_visualization(
    frames: List[np.ndarray],
    alignment: List[str],
    word: str,
    output_path: Path,
    title: str = "CTC Alignment Process for Lip Reading",
    show_frame_numbers: bool = True,
    color_scheme: str = "purple"
):
    """
    Create a publication-ready CTC alignment visualization.
    """
    num_frames = len(frames)
    
    # Color schemes
    colors = {
        'purple': {
            'char_box': '#8B5A8D',
            'char_edge': '#6B4A6D',
            'blank_box': '#F5F5F5',
            'blank_edge': '#BBBBBB',
            'arrow': '#2E7D9B',
            'word_box': '#E88D2C',
            'word_edge': '#B86D0C'
        },
        'blue': {
            'char_box': '#3A86B8',
            'char_edge': '#2A6698',
            'blank_box': '#F5F5F5',
            'blank_edge': '#BBBBBB',
            'arrow': '#5AA854',
            'word_box': '#E88D2C',
            'word_edge': '#B86D0C'
        }
    }
    
    scheme = colors.get(color_scheme, colors['purple'])
    
    # Set up figure with higher quality
    fig_width = max(16, num_frames * 1.8)
    fig_height = 7.5
    fig = plt.figure(figsize=(fig_width, fig_height))
    
    # Layout positions
    frame_height = 0.32
    frame_y_top = 0.58
    arrow_y_start = frame_y_top - 0.02
    arrow_y_end = 0.32
    char_y = 0.24
    collapse_y = 0.15
    word_y = 0.04
    
    frame_spacing = 0.88 / num_frames
    frame_width = frame_spacing * 0.82
    
    # Title
    fig.suptitle(title, fontsize=20, fontweight='bold', y=0.97)
    
    # Draw frames
    for i, frame in enumerate(frames):
        x_pos = 0.06 + i * frame_spacing + frame_spacing * 0.09
        ax = fig.add_axes([x_pos, frame_y_top, frame_width, frame_height])
        ax.imshow(frame, cmap='gray', aspect='auto')
        if show_frame_numbers:
            ax.set_title(f't={i+1}', fontsize=11, pad=5, fontweight='bold')
        ax.axis('off')
    
    # Main axis for annotations
    ax_main = fig.add_axes([0, 0, 1, 1])
    ax_main.set_xlim(0, 1)
    ax_main.set_ylim(0, 1)
    ax_main.axis('off')
    
    # Draw frame-to-character alignments
    for i in range(num_frames):
        x_center = 0.06 + i * frame_spacing + frame_spacing / 2
        
        # Arrow
        arrow = FancyArrowPatch(
            (x_center, arrow_y_start),
            (x_center, arrow_y_end),
            arrowstyle='->,head_width=0.35,head_length=0.35',
            color=scheme['arrow'],
            linewidth=2.8,
            mutation_scale=25,
            zorder=10
        )
        ax_main.add_patch(arrow)
        
        # Character prediction
        char = alignment[i] if i < len(alignment) else '-'
        is_blank = (char == '-')
        
        display_char = 'ε' if is_blank else char
        box_color = scheme['blank_box'] if is_blank else scheme['char_box']
        text_color = '#999999' if is_blank else 'white'
        edge_color = scheme['blank_edge'] if is_blank else scheme['char_edge']
        
        # Character box
        box_width = frame_width * 0.75
        box_height = 0.075
        box_x = x_center - box_width / 2
        box_y = char_y
        
        rect = mpatches.FancyBboxPatch(
            (box_x, box_y),
            box_width,
            box_height,
            boxstyle="round,pad=0.008",
            facecolor=box_color,
            edgecolor=edge_color,
            linewidth=2.5,
            zorder=5
        )
        ax_main.add_patch(rect)
        
        # Character text
        ax_main.text(
            x_center,
            char_y + box_height / 2,
            display_char,
            ha='center',
            va='center',
            fontsize=17,
            fontweight='bold',
            color=text_color,
            family='monospace',
            zorder=15
        )
    
    # CTC Collapse explanation with visual separator
    collapsed = ctc_collapse(alignment)
    
    # Draw separator line
    ax_main.plot([0.06, 0.94], [collapse_y + 0.03, collapse_y + 0.03], 
                 'k--', linewidth=1.5, alpha=0.3)
    
    # Collapse operation text
    ax_main.text(
        0.5,
        collapse_y,
        'CTC Decoding: Remove blanks (ε) and consecutive duplicates',
        ha='center',
        va='center',
        fontsize=13,
        style='italic',
        color='#333333',
        bbox=dict(
            boxstyle='round,pad=0.5',
            facecolor='white',
            edgecolor='#CCCCCC',
            linewidth=1.5,
            alpha=0.9
        )
    )
    
    # Show the collapse step-by-step
    step_text = f"[{' '.join(alignment)}] → [{'  '.join([c for c in alignment if c != '-'])}] → \"{collapsed}\""
    if len(step_text) < 80:
        ax_main.text(
            0.5,
            collapse_y - 0.045,
            step_text,
            ha='center',
            va='center',
            fontsize=10,
            family='monospace',
            color='#555555'
        )
    
    # Final word box
    word_box_width = min(0.45, 0.08 * len(word) + 0.15)
    word_box_height = 0.09
    word_box_x = 0.5 - word_box_width / 2
    word_box_y = word_y
    
    word_rect = mpatches.FancyBboxPatch(
        (word_box_x, word_box_y),
        word_box_width,
        word_box_height,
        boxstyle="round,pad=0.015",
        facecolor=scheme['word_box'],
        edgecolor=scheme['word_edge'],
        linewidth=3.5,
        zorder=5
    )
    ax_main.add_patch(word_rect)
    
    # Final word text
    ax_main.text(
        0.5,
        word_y + word_box_height / 2,
        f'Output: "{word}"',
        ha='center',
        va='center',
        fontsize=19,
        fontweight='bold',
        color='white',
        zorder=15
    )
    
    # Add process explanation box
    explanation_text = (
        "CTC Alignment:\n"
        "• Each frame produces a prediction\n"
        "• Blanks (ε) separate characters\n"
        "• Same character can span frames\n"
        "• Duplicates are collapsed"
    )
    
    ax_main.text(
        0.02,
        0.94,
        explanation_text,
        ha='left',
        va='top',
        fontsize=10,
        bbox=dict(
            boxstyle='round,pad=0.6',
            facecolor='#FFFAF0',
            edgecolor='#DAA520',
            linewidth=2.5,
            alpha=0.98
        ),
        family='sans-serif',
        linespacing=1.6
    )
    
    # Save
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved visualization to: {output_path}")
    
    # Also save as PDF for paper
    pdf_path = output_path.with_suffix('.pdf')
    plt.savefig(pdf_path, bbox_inches='tight', facecolor='white')
    print(f"✓ Saved PDF version to: {pdf_path}")
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Create enhanced CTC alignment visualization for papers"
    )
    parser.add_argument(
        '--sample',
        type=str,
        required=True,
        help='Path to sample directory with frames'
    )
    parser.add_argument(
        '--alignment',
        type=str,
        required=True,
        help='Frame-level alignment (space-separated). Use "-" for blank. Example: "- H H - E L L O"'
    )
    parser.add_argument(
        '--word',
        type=str,
        required=True,
        help='Final decoded word'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='export/ctc_alignment_paper.png',
        help='Output path'
    )
    parser.add_argument(
        '--title',
        type=str,
        default='CTC Alignment Process for Lip Reading',
        help='Visualization title'
    )
    parser.add_argument(
        '--color-scheme',
        type=str,
        default='purple',
        choices=['purple', 'blue'],
        help='Color scheme'
    )
    parser.add_argument(
        '--num-frames',
        type=int,
        default=8,
        help='Number of frames to load'
    )
    
    args = parser.parse_args()
    
    # Load frames
    sample_dir = Path(args.sample)
    if not sample_dir.exists():
        print(f"Error: Sample directory not found: {sample_dir}")
        sys.exit(1)
    
    print(f"Loading {args.num_frames} frames from: {sample_dir}")
    frames = load_frames(sample_dir, num_frames=args.num_frames)
    
    if len(frames) == 0:
        print("Error: No frames found")
        sys.exit(1)
    
    print(f"Loaded {len(frames)} frames")
    
    # Parse alignment
    alignment = parse_alignment(args.alignment)
    
    if len(alignment) != len(frames):
        print(f"Warning: Alignment length ({len(alignment)}) != frames ({len(frames)})")
        print(f"Adjusting alignment...")
        if len(alignment) < len(frames):
            alignment.extend(['-'] * (len(frames) - len(alignment)))
        else:
            alignment = alignment[:len(frames)]
    
    # Verify alignment produces correct word
    collapsed = ctc_collapse(alignment)
    if collapsed != args.word:
        print(f"Warning: Collapsed alignment '{collapsed}' != expected word '{args.word}'")
        print("Continuing anyway...")
    
    print(f"Alignment: {' '.join(alignment)}")
    print(f"Collapsed: {collapsed}")
    print(f"Expected:  {args.word}")
    
    # Create visualization
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    create_paper_ready_visualization(
        frames=frames,
        alignment=alignment,
        word=args.word,
        output_path=output_path,
        title=args.title,
        color_scheme=args.color_scheme
    )
    
    print(f"\n✓ Visualization complete!")
    print(f"  Output: {output_path.absolute()}")


if __name__ == '__main__':
    main()

