#!/usr/bin/env python3
"""
Create COMPACT 3-layer CTC visualization using REAL preprocessed lip crops.

Usage:
    python scripts/create_compact_ctc_viz.py
"""

import sys
from pathlib import Path
import torch
import torchvision
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch


def load_video_frames(video_path, num_frames=7, start_frame=5):
    """Load frames from video."""
    vid, _, _ = torchvision.io.read_video(str(video_path), pts_unit='sec')
    
    if vid.numel() == 0:
        raise ValueError(f"Empty video: {video_path}")
    
    vid_np = vid.numpy()  # (T, H, W, C)
    T = vid_np.shape[0]
    
    end_frame = min(start_frame + num_frames, T)
    selected_frames = vid_np[start_frame:end_frame]
    
    # Convert to grayscale
    grayscale_frames = []
    for frame in selected_frames:
        gray = (frame[:, :, 0] * 0.299 + 
                frame[:, :, 1] * 0.587 + 
                frame[:, :, 2] * 0.114).astype(np.uint8)
        grayscale_frames.append(gray)
    
    return grayscale_frames


def create_compact_visualization(
    full_frames,
    lip_frames,
    alignment,
    word,
    output_path
):
    """
    Create COMPACT 3-layer visualization with preprocessed lip crops.
    """
    num_frames = len(full_frames)
    
    # COMPACT dimensions
    fig_width = max(14, num_frames * 1.8)
    fig_height = 5.5  # Much more compact
    
    fig = plt.figure(figsize=(fig_width, fig_height), facecolor='white')
    
    # COMPACT layout - tighter spacing
    # Layer 1: Full frames (top)
    full_frame_y_bottom = 0.58
    full_frame_height = 0.38
    
    # Layer 2: Lip crops (middle) - SMALLER
    lip_frame_y_bottom = 0.35
    lip_frame_height = 0.20
    
    # Arrow
    arrow_start_y = lip_frame_y_bottom - 0.02
    arrow_end_y = 0.24
    
    # Layer 3: Character boxes
    char_box_y = 0.16
    char_box_height = 0.065
    
    # Separator
    separator_y = 0.08
    
    # Layer 4: Final word
    word_box_y = 0.01
    word_box_height = 0.065
    
    # Horizontal spacing - more compact
    frame_spacing = 0.88 / num_frames
    frame_width = frame_spacing * 0.80
    frame_left_margin = 0.06
    
    # Draw FULL frames at top
    for i, frame in enumerate(full_frames):
        x_pos = frame_left_margin + i * frame_spacing + frame_spacing * 0.10
        ax = fig.add_axes([x_pos, full_frame_y_bottom, frame_width, full_frame_height])
        ax.imshow(frame, cmap='gray', aspect='auto', interpolation='bilinear')
        ax.set_title(f't={i+1}', fontsize=9, pad=2, fontweight='bold')
        ax.axis('off')
    
    # Draw PREPROCESSED LIP crops in middle
    for i, lip_frame in enumerate(lip_frames):
        x_pos = frame_left_margin + i * frame_spacing + frame_spacing * 0.10
        ax = fig.add_axes([x_pos, lip_frame_y_bottom, frame_width, lip_frame_height])
        ax.imshow(lip_frame, cmap='gray', aspect='auto', interpolation='bilinear')
        ax.axis('off')
        # Blue border
        for spine in ax.spines.values():
            spine.set_edgecolor('#2E7D9B')
            spine.set_linewidth(2)
    
    # Main axis for annotations
    ax_main = fig.add_axes([0, 0, 1, 1])
    ax_main.set_xlim(0, 1)
    ax_main.set_ylim(0, 1)
    ax_main.axis('off')
    
    # Draw arrows from lips to characters
    for i in range(num_frames):
        x_center = frame_left_margin + i * frame_spacing + frame_spacing / 2
        
        # Arrow
        arrow = FancyArrowPatch(
            (x_center, arrow_start_y),
            (x_center, arrow_end_y),
            arrowstyle='->,head_width=0.3,head_length=0.3',
            color='#2E7D9B',
            linewidth=2.2,
            mutation_scale=18,
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
        
        box_width = frame_width * 0.75
        box_x = x_center - box_width / 2
        
        rect = mpatches.FancyBboxPatch(
            (box_x, char_box_y),
            box_width,
            char_box_height,
            boxstyle="round,pad=0.005",
            facecolor=box_color,
            edgecolor=edge_color,
            linewidth=1.8,
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
            fontsize=14,
            fontweight='bold',
            color=text_color,
            family='monospace',
            zorder=15
        )
    
    # Separator line
    ax_main.plot([frame_left_margin, 0.94], [separator_y, separator_y],
                 'k--', linewidth=0.8, alpha=0.3, zorder=1)
    
    # Final decoded word box
    word_box_width = min(0.35, len(word) * 0.04 + 0.12)
    word_box_x = 0.5 - word_box_width / 2
    
    word_rect = mpatches.FancyBboxPatch(
        (word_box_x, word_box_y),
        word_box_width,
        word_box_height,
        boxstyle="round,pad=0.01",
        facecolor='#E88D2C',
        edgecolor='#C87200',
        linewidth=2.2,
        zorder=10
    )
    ax_main.add_patch(word_rect)
    
    ax_main.text(
        0.5,
        word_box_y + word_box_height / 2,
        f'"{word}"',
        ha='center',
        va='center',
        fontsize=15,
        fontweight='bold',
        color='white',
        zorder=15
    )
    
    # Save
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0.02)
    print(f"✓ Saved: {output_path}")
    
    # PDF version
    pdf_path = output_path.with_suffix('.pdf')
    plt.savefig(pdf_path, bbox_inches='tight', facecolor='white', pad_inches=0.02)
    print(f"✓ Saved: {pdf_path}")
    
    plt.close()


def main():
    print("Creating COMPACT visualization with REAL preprocessed lip crops...")
    print("=" * 70)
    
    # Find LRS2 videos
    lrs2_root = Path("datasets/LRS2/mvlrs_v1")
    
    # Find a video WITH corresponding mouth crop
    video_dirs = list((lrs2_root / "pretrain").glob("*/"))
    
    video_path = None
    mouth_path = None
    
    for vid_dir in video_dirs:
        videos = list(vid_dir.glob("*.mp4"))
        for v in videos:
            # Check if there's a corresponding mouth video
            mouth_candidate = v.parent / (v.stem + "_mouth.mp4")
            if mouth_candidate.exists():
                video_path = v
                mouth_path = mouth_candidate
                break
        if video_path:
            break
    
    if not video_path or not mouth_path:
        # Try with the one we found earlier
        mouth_path = Path("datasets/LRS2/mvlrs_v1/pretrain/6077571138829809070/00036_mouth.mp4")
        video_path = Path("datasets/LRS2/mvlrs_v1/pretrain/6077571138829809070/00036.mp4")
        
        if not mouth_path.exists():
            print("ERROR: Could not find preprocessed mouth crops!")
            return
    
    print(f"Using video: {video_path}")
    print(f"Using mouth: {mouth_path}")
    
    try:
        # Load full frames
        full_frames = load_video_frames(video_path, num_frames=7, start_frame=5)
        print(f"Loaded {len(full_frames)} full frames")
        
        # Load PREPROCESSED lip frames
        lip_frames = load_video_frames(mouth_path, num_frames=7, start_frame=5)
        print(f"Loaded {len(lip_frames)} preprocessed lip frames")
        
        # Create CHIPS visualization
        output_path = Path("export/paper_samples/ctc_compact_chips.png")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        create_compact_visualization(
            full_frames=full_frames,
            lip_frames=lip_frames,
            alignment=['ε', 'C', 'H', 'I', 'P', 'S', 'ε'],
            word="CHIPS",
            output_path=output_path
        )
        
        # Create COOKING visualization
        if len(full_frames) >= 7:
            output_path2 = Path("export/paper_samples/ctc_compact_cooking.png")
            create_compact_visualization(
                full_frames=full_frames[:7],
                lip_frames=lip_frames[:7],
                alignment=['C', 'O', 'ε', 'O', 'K', 'I', 'N'],
                word="COOKING",
                output_path=output_path2
            )
        
        # Create HOME visualization (8 frames)
        full_frames_8 = load_video_frames(video_path, num_frames=8, start_frame=5)
        lip_frames_8 = load_video_frames(mouth_path, num_frames=8, start_frame=5)
        
        if len(full_frames_8) >= 8 and len(lip_frames_8) >= 8:
            output_path3 = Path("export/paper_samples/ctc_compact_home.png")
            create_compact_visualization(
                full_frames=full_frames_8,
                lip_frames=lip_frames_8,
                alignment=['ε', 'H', 'O', 'O', 'M', 'M', 'E', 'ε'],
                word="HOME",
                output_path=output_path3
            )
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("=" * 70)
    print("✅ COMPACT visualizations created!")
    print("\n📊 Features:")
    print("  • Uses REAL preprocessed lip crops (not manually extracted)")
    print("  • Much more compact layout (less vertical space)")
    print("  • Clean 3-layer flow: person → lips → characters → word")
    print("  • NO text at top")
    print("\nFiles created:")
    print("  • ctc_compact_chips.png/pdf    ⭐ PERFECT FOR PAPER")
    print("  • ctc_compact_cooking.png/pdf")
    print("  • ctc_compact_home.png/pdf")
    print("\nLocation: export/paper_samples/")


if __name__ == '__main__':
    main()

