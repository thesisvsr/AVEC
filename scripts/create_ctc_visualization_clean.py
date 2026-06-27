#!/usr/bin/env python3
"""
Create clean CTC alignment visualizations with proper spacing (no overlaps).

Usage:
    python scripts/create_ctc_visualization_clean.py
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
from pathlib import Path


def create_clean_ctc_visualization(
    frames_data,
    alignment,
    word,
    full_transcript,
    output_path,
    title="CTC Alignment for Lip Reading"
):
    """
    Create a clean CTC visualization with NO overlapping elements.
    
    Args:
        frames_data: List of numpy arrays (grayscale images)
        alignment: List of characters (use '-' for blank)
        word: Final decoded word
        full_transcript: Full sentence transcript
        output_path: Where to save the PNG/PDF
    """
    num_frames = len(frames_data)
    
    # Figure dimensions - wider for better spacing
    fig_width = max(18, num_frames * 2)
    fig_height = 6.5
    
    fig = plt.figure(figsize=(fig_width, fig_height), facecolor='white')
    
    # Layout coordinates (carefully chosen to avoid overlap)
    frame_y_bottom = 0.40  # Frames at 40% from bottom
    frame_height = 0.48     # Frames take 48% of height
    frame_y_top = frame_y_bottom + frame_height
    
    arrow_start_y = frame_y_bottom - 0.03
    arrow_end_y = 0.24
    
    char_box_y = 0.16
    char_box_height = 0.06
    
    separator_y = 0.09
    
    word_box_y = 0.01
    word_box_height = 0.06
    
    # Frame positioning
    frame_spacing = 0.86 / num_frames
    frame_width = frame_spacing * 0.75
    frame_left_margin = 0.07
    
    # Title
    fig.suptitle(title, fontsize=18, fontweight='bold', y=0.96)
    
    # Subtitle with full transcript
    fig.text(0.5, 0.915, f'Full transcript: "{full_transcript}"', 
             ha='center', fontsize=11, style='italic', color='#555555')
    
    # Add context note about aligned segment
    fig.text(0.5, 0.88, f'Aligned word segment: "{word}"',
             ha='center', fontsize=10, weight='bold', color='#2E7D9B')
    
    # Draw frames
    for i, frame in enumerate(frames_data):
        x_pos = frame_left_margin + i * frame_spacing + frame_spacing * 0.125
        ax = fig.add_axes([x_pos, frame_y_bottom, frame_width, frame_height])
        ax.imshow(frame, cmap='gray', aspect='auto')
        ax.set_title(f't={i+1}', fontsize=10, pad=4, fontweight='bold')
        ax.axis('off')
    
    # Main axis for annotations
    ax_main = fig.add_axes([0, 0, 1, 1])
    ax_main.set_xlim(0, 1)
    ax_main.set_ylim(0, 1)
    ax_main.axis('off')
    
    # Draw arrows and character boxes (BELOW frames, no overlap)
    for i in range(num_frames):
        x_center = frame_left_margin + i * frame_spacing + frame_spacing / 2
        
        # Arrow from frame to character box
        arrow = FancyArrowPatch(
            (x_center, arrow_start_y),
            (x_center, arrow_end_y),
            arrowstyle='->,head_width=0.3,head_length=0.3',
            color='#2E7D9B',
            linewidth=2.5,
            mutation_scale=20,
            zorder=5
        )
        ax_main.add_patch(arrow)
        
        # Character prediction box
        char = alignment[i] if i < len(alignment) else '-'
        is_blank = (char == '-')
        
        display_char = 'ε' if is_blank else char
        box_color = '#E8E8E8' if is_blank else '#7B68A6'
        text_color = '#999999' if is_blank else 'white'
        edge_color = '#CCCCCC' if is_blank else '#5A4A7A'
        
        box_width = frame_width * 0.7
        box_x = x_center - box_width / 2
        
        rect = mpatches.FancyBboxPatch(
            (box_x, char_box_y),
            box_width,
            char_box_height,
            boxstyle="round,pad=0.005",
            facecolor=box_color,
            edgecolor=edge_color,
            linewidth=2,
            zorder=10
        )
        ax_main.add_patch(rect)
        
        # Character text
        ax_main.text(
            x_center,
            char_box_y + char_box_height / 2,
            display_char,
            ha='center',
            va='center',
            fontsize=15,
            fontweight='bold',
            color=text_color,
            family='monospace',
            zorder=15
        )
    
    # Separator line
    ax_main.plot([frame_left_margin, 0.93], [separator_y, separator_y],
                 'k--', linewidth=1, alpha=0.3, zorder=1)
    
    # Final decoded word box
    word_box_width = min(0.35, len(word) * 0.04 + 0.1)
    word_box_x = 0.5 - word_box_width / 2
    
    word_rect = mpatches.FancyBboxPatch(
        (word_box_x, word_box_y),
        word_box_width,
        word_box_height,
        boxstyle="round,pad=0.01",
        facecolor='#E88D2C',
        edgecolor='#C87200',
        linewidth=2.5,
        zorder=10
    )
    ax_main.add_patch(word_rect)
    
    ax_main.text(
        0.5,
        word_box_y + word_box_height / 2,
        f'Decoded Output: "{word}"',
        ha='center',
        va='center',
        fontsize=16,
        fontweight='bold',
        color='white',
        zorder=15
    )
    
    # Add legend/explanation box (top left)
    legend_text = (
        "CTC Alignment:\n"
        "• Each video frame generates a prediction\n"
        "• Model outputs characters or blanks (ε)\n"
        "• Blanks allow character boundaries\n"
        "• Consecutive duplicates collapse to one\n"
        "• Monotonic alignment (no backtracking)"
    )
    
    ax_main.text(
        0.015,
        0.82,
        legend_text,
        ha='left',
        va='top',
        fontsize=8.5,
        bbox=dict(
            boxstyle='round,pad=0.6',
            facecolor='#FFFBEE',
            edgecolor='#DAA520',
            linewidth=2,
            alpha=0.95
        ),
        family='sans-serif',
        linespacing=1.5,
        zorder=20
    )
    
    # Save
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0.1)
    print(f"✓ Saved: {output_path}")
    
    # Also save PDF
    pdf_path = output_path.with_suffix('.pdf')
    plt.savefig(pdf_path, bbox_inches='tight', facecolor='white', pad_inches=0.1)
    print(f"✓ Saved: {pdf_path}")
    
    plt.close()


def create_sample_visualization_chips():
    """Create the CHIPS example with mock frames."""
    # Create sample grayscale frames (mock data for demonstration)
    # In real use, you'd load actual preprocessed frames
    frames = []
    for i in range(7):
        # Create a simple gradient pattern as placeholder
        frame = np.ones((88, 88), dtype=np.uint8) * (100 + i * 20)
        frames.append(frame)
    
    alignment = ['ε', 'C', 'H', 'I', 'P', 'S', 'ε']
    word = "CHIPS"
    full_transcript = "WHEN YOU'RE COOKING CHIPS AT HOME"
    
    output_path = Path("export/paper_samples/ctc_clean_chips.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    create_clean_ctc_visualization(
        frames_data=frames,
        alignment=alignment,
        word=word,
        full_transcript=full_transcript,
        output_path=output_path,
        title="CTC Alignment for Lip Reading"
    )


def create_sample_visualization_cooking():
    """Create the COOKING example."""
    frames = []
    for i in range(7):
        frame = np.ones((88, 88), dtype=np.uint8) * (80 + i * 25)
        frames.append(frame)
    
    alignment = ['C', 'O', 'ε', 'O', 'K', 'I', 'N']
    word = "COOKING"
    full_transcript = "WHEN YOU'RE COOKING CHIPS AT HOME"
    
    output_path = Path("export/paper_samples/ctc_clean_cooking.png")
    
    create_clean_ctc_visualization(
        frames_data=frames,
        alignment=alignment,
        word=word,
        full_transcript=full_transcript,
        output_path=output_path,
        title="CTC Alignment for Lip Reading"
    )


def create_sample_visualization_home():
    """Create the HOME example."""
    frames = []
    for i in range(8):
        frame = np.ones((88, 88), dtype=np.uint8) * (90 + i * 20)
        frames.append(frame)
    
    alignment = ['ε', 'H', 'O', 'O', 'M', 'M', 'E', 'ε']
    word = "HOME"
    full_transcript = "WHEN YOU'RE COOKING CHIPS AT HOME"
    
    output_path = Path("export/paper_samples/ctc_clean_home.png")
    
    create_clean_ctc_visualization(
        frames_data=frames,
        alignment=alignment,
        word=word,
        full_transcript=full_transcript,
        output_path=output_path,
        title="CTC Alignment for Lip Reading"
    )


def main():
    print("Creating clean CTC alignment visualizations...")
    print("=" * 70)
    
    create_sample_visualization_chips()
    create_sample_visualization_cooking()
    create_sample_visualization_home()
    
    print("=" * 70)
    print("✓ All visualizations created successfully!")
    print("\nNote: These use placeholder frames. To use real frames:")
    print("  1. Load actual preprocessed frames from your dataset")
    print("  2. Pass them to create_clean_ctc_visualization()")
    print("\nOutput location: export/paper_samples/ctc_clean_*.png")


if __name__ == '__main__':
    main()

