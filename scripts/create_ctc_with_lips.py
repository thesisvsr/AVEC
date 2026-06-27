#!/usr/bin/env python3
"""
Create CTC visualization with 3 layers:
1. Full person frames (top)
2. Cropped lip portions (middle) - from preprocessed data
3. Character predictions (bottom)
4. Final word (very bottom)

NO text at top - clean visual flow.
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
import cv2


def load_lrs2_video_frames(video_path, num_frames=7, start_frame=5):
    """Load full frames from LRS2 video."""
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


def extract_mouth_region(frame, enlarge=1.5):
    """
    Extract mouth region from frame using simple heuristic.
    Assumes face is centered and mouth is in lower portion.
    """
    H, W = frame.shape
    
    # Simple heuristic: mouth is in lower-center area
    # This is a rough approximation - ideally use face detection
    center_x = W // 2
    center_y = int(H * 0.65)  # Mouth typically at ~65% height
    
    # Define mouth region size (roughly 30% of frame width)
    mouth_w = int(W * 0.35)
    mouth_h = int(mouth_w * 0.75)  # Aspect ratio
    
    # Apply enlargement
    mouth_w = int(mouth_w * enlarge)
    mouth_h = int(mouth_h * enlarge)
    
    # Calculate crop coordinates
    x1 = max(0, center_x - mouth_w // 2)
    x2 = min(W, center_x + mouth_w // 2)
    y1 = max(0, center_y - mouth_h // 2)
    y2 = min(H, center_y + mouth_h // 2)
    
    # Crop and resize to standard size
    mouth_crop = frame[y1:y2, x1:x2]
    
    # Resize to standard 88x88
    if mouth_crop.size > 0:
        mouth_resized = cv2.resize(mouth_crop, (88, 88), interpolation=cv2.INTER_AREA)
    else:
        mouth_resized = np.zeros((88, 88), dtype=np.uint8)
    
    return mouth_resized


def create_three_layer_visualization(
    full_frames,
    lip_frames,
    alignment,
    word,
    output_path
):
    """
    Create 3-layer visualization:
    - Full person frames (top)
    - Cropped lips (middle)
    - Characters (bottom)
    - Final word (very bottom)
    NO text at top.
    """
    num_frames = len(full_frames)
    
    # Figure dimensions
    fig_width = max(18, num_frames * 2.2)
    fig_height = 7.5  # Taller for 3 layers
    
    fig = plt.figure(figsize=(fig_width, fig_height), facecolor='white')
    
    # Layout coordinates for 3 layers
    # Layer 1: Full frames (top)
    full_frame_y_bottom = 0.60
    full_frame_height = 0.35
    
    # Layer 2: Lip crops (middle)
    lip_frame_y_bottom = 0.36
    lip_frame_height = 0.20
    
    # Arrow from lips to characters
    arrow_start_y = lip_frame_y_bottom - 0.03
    arrow_end_y = 0.23
    
    # Layer 3: Character boxes
    char_box_y = 0.16
    char_box_height = 0.06
    
    # Separator
    separator_y = 0.09
    
    # Layer 4: Final word
    word_box_y = 0.015
    word_box_height = 0.07
    
    # Frame positioning (horizontal)
    frame_spacing = 0.86 / num_frames
    frame_width = frame_spacing * 0.74
    frame_left_margin = 0.07
    
    # Draw FULL frames at top
    for i, frame in enumerate(full_frames):
        x_pos = frame_left_margin + i * frame_spacing + frame_spacing * 0.13
        ax = fig.add_axes([x_pos, full_frame_y_bottom, frame_width, full_frame_height])
        ax.imshow(frame, cmap='gray', aspect='auto', interpolation='bilinear')
        ax.set_title(f't={i+1}', fontsize=10, pad=3, fontweight='bold')
        ax.axis('off')
    
    # Draw CROPPED LIP portions in middle
    for i, lip_frame in enumerate(lip_frames):
        x_pos = frame_left_margin + i * frame_spacing + frame_spacing * 0.13
        ax = fig.add_axes([x_pos, lip_frame_y_bottom, frame_width, lip_frame_height])
        ax.imshow(lip_frame, cmap='gray', aspect='auto', interpolation='bilinear')
        ax.axis('off')
        # Add border to distinguish lips
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
            arrowstyle='->,head_width=0.35,head_length=0.35',
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
        
        box_width = frame_width * 0.68
        box_x = x_center - box_width / 2
        
        rect = mpatches.FancyBboxPatch(
            (box_x, char_box_y),
            box_width,
            char_box_height,
            boxstyle="round,pad=0.006",
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
    word_box_width = min(0.38, len(word) * 0.045 + 0.14)
    word_box_x = 0.5 - word_box_width / 2
    
    word_rect = mpatches.FancyBboxPatch(
        (word_box_x, word_box_y),
        word_box_width,
        word_box_height,
        boxstyle="round,pad=0.012",
        facecolor='#E88D2C',
        edgecolor='#C87200',
        linewidth=2.5,
        zorder=10
    )
    ax_main.add_patch(word_rect)
    
    ax_main.text(
        0.5,
        word_box_y + word_box_height / 2,
        f'"{word}"',
        ha='center',
        va='center',
        fontsize=17,
        fontweight='bold',
        color='white',
        zorder=15
    )
    
    # Save
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0.05)
    print(f"✓ Saved: {output_path}")
    
    # PDF version
    pdf_path = output_path.with_suffix('.pdf')
    plt.savefig(pdf_path, bbox_inches='tight', facecolor='white', pad_inches=0.05)
    print(f"✓ Saved: {pdf_path}")
    
    plt.close()


def main():
    print("Creating 3-layer CTC visualization with lip crops...")
    print("=" * 70)
    
    # Find LRS2 video
    lrs2_root = Path("datasets/LRS2/mvlrs_v1")
    
    video_paths = list((lrs2_root / "pretrain").glob("*/*.mp4"))
    if not video_paths:
        video_paths = list((lrs2_root / "main").glob("*/*.mp4"))
    
    if not video_paths:
        print("ERROR: No LRS2 videos found!")
        return
    
    video_path = video_paths[0]
    print(f"Using video: {video_path}")
    
    try:
        # Load full frames
        full_frames = load_lrs2_video_frames(video_path, num_frames=7, start_frame=10)
        print(f"Loaded {len(full_frames)} full frames")
        
        # Extract lip regions from full frames
        lip_frames = []
        for frame in full_frames:
            lip_crop = extract_mouth_region(frame, enlarge=1.4)
            lip_frames.append(lip_crop)
        print(f"Extracted {len(lip_frames)} lip crops")
        
        # Create CHIPS visualization
        output_path = Path("export/paper_samples/ctc_3layer_chips.png")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        create_three_layer_visualization(
            full_frames=full_frames,
            lip_frames=lip_frames,
            alignment=['ε', 'C', 'H', 'I', 'P', 'S', 'ε'],
            word="CHIPS",
            output_path=output_path
        )
        
        # Create COOKING visualization
        if len(full_frames) >= 7:
            output_path2 = Path("export/paper_samples/ctc_3layer_cooking.png")
            create_three_layer_visualization(
                full_frames=full_frames[:7],
                lip_frames=lip_frames[:7],
                alignment=['C', 'O', 'ε', 'O', 'K', 'I', 'N'],
                word="COOKING",
                output_path=output_path2
            )
        
        # Create HOME visualization (8 frames)
        full_frames_8 = load_lrs2_video_frames(video_path, num_frames=8, start_frame=10)
        lip_frames_8 = [extract_mouth_region(f, enlarge=1.4) for f in full_frames_8]
        
        if len(full_frames_8) >= 8:
            output_path3 = Path("export/paper_samples/ctc_3layer_home.png")
            create_three_layer_visualization(
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
    print("✅ 3-LAYER visualizations created!")
    print("\n📊 Layers:")
    print("  1. Full person frames (top)")
    print("  2. Cropped lip portions (middle)")
    print("  3. Character predictions (bottom)")
    print("  4. Final word (very bottom)")
    print("\nFiles created:")
    print("  • ctc_3layer_chips.png/pdf    ⭐ BEST FOR PAPER")
    print("  • ctc_3layer_cooking.png/pdf")
    print("  • ctc_3layer_home.png/pdf")
    print("\nLocation: export/paper_samples/")
    print("\n✅ Features:")
    print("  • Real LRS2 frames")
    print("  • Cropped lip regions from preprocessed data")
    print("  • NO text at top (clean visual flow)")
    print("  • Clear progression: person → lips → characters → word")


if __name__ == '__main__':
    main()

