#!/usr/bin/env python3
"""
Create CTC visualization using REAL transcript and matching video frames.

Uses actual word timing from LRS2 transcript files.
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


def read_lrs2_transcript(txt_path):
    """Read LRS2 transcript file and extract word timings."""
    with open(txt_path, 'r') as f:
        lines = f.readlines()
    
    # First line contains the full text
    full_text = lines[0].replace("Text:", "").strip()
    
    # Parse word timings
    words = []
    for line in lines[4:]:  # Skip first 4 lines (including WORD START END header)
        parts = line.strip().split()
        if len(parts) >= 4:
            try:
                word = parts[0]
                start = float(parts[1])
                end = float(parts[2])
                words.append({'word': word, 'start': start, 'end': end})
            except ValueError:
                continue  # Skip lines that can't be parsed
    
    return full_text, words


def load_video_frames_for_word(video_path, start_time, end_time, target_frames=7):
    """Load frames for a specific word based on timing."""
    vid, _, info = torchvision.io.read_video(str(video_path), pts_unit='sec')
    
    if vid.numel() == 0:
        raise ValueError(f"Empty video: {video_path}")
    
    vid_np = vid.numpy()  # (T, H, W, C)
    T, H, W, C = vid_np.shape
    
    # Get FPS (usually 25 for LRS2)
    fps = 25
    
    # Calculate frame indices for the word
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)
    
    # Extract frames for this word
    word_frames = vid_np[start_frame:end_frame+1]
    
    # Resample to target number of frames if needed
    if len(word_frames) > target_frames:
        # Uniformly sample target_frames
        indices = np.linspace(0, len(word_frames)-1, target_frames).astype(int)
        word_frames = word_frames[indices]
    elif len(word_frames) < target_frames:
        # Pad by repeating last frame
        padding = [word_frames[-1]] * (target_frames - len(word_frames))
        word_frames = np.concatenate([word_frames, padding], axis=0)
    
    # Convert to grayscale
    grayscale_frames = []
    for frame in word_frames:
        gray = (frame[:, :, 0] * 0.299 + 
                frame[:, :, 1] * 0.587 + 
                frame[:, :, 2] * 0.114).astype(np.uint8)
        grayscale_frames.append(gray)
    
    return grayscale_frames


def create_realistic_ctc_alignment(word, num_frames):
    """Create a realistic CTC alignment for a word."""
    word = word.upper()
    chars = list(word)
    
    # Create alignment with blanks
    # Pattern: often starts with blank, has blanks between characters
    alignment = []
    
    if num_frames <= len(chars):
        # Fewer frames than characters - tight alignment
        for i in range(num_frames):
            if i < len(chars):
                alignment.append(chars[i])
            else:
                alignment.append('-')
    else:
        # More frames than characters - add blanks
        # Start with blank
        alignment.append('-')
        remaining_frames = num_frames - 1
        
        # Distribute characters across remaining frames
        for i, char in enumerate(chars):
            alignment.append(char)
            remaining_frames -= 1
            
            # Add blank between some characters
            if i < len(chars) - 1 and remaining_frames > len(chars) - i - 1:
                alignment.append('-')
                remaining_frames -= 1
        
        # Fill remaining with blanks or character repeats
        while len(alignment) < num_frames:
            alignment.append('-')
    
    return alignment


def create_compact_visualization(
    full_frames,
    lip_frames,
    alignment,
    word,
    output_path
):
    """Create compact visualization."""
    num_frames = len(full_frames)
    
    fig_width = max(14, num_frames * 1.8)
    fig_height = 5.5
    
    fig = plt.figure(figsize=(fig_width, fig_height), facecolor='white')
    
    # Layout
    full_frame_y_bottom = 0.58
    full_frame_height = 0.38
    
    lip_frame_y_bottom = 0.35
    lip_frame_height = 0.20
    
    arrow_start_y = lip_frame_y_bottom - 0.02
    arrow_end_y = 0.24
    
    char_box_y = 0.16
    char_box_height = 0.065
    
    separator_y = 0.08
    
    word_box_y = 0.01
    word_box_height = 0.065
    
    frame_spacing = 0.88 / num_frames
    frame_width = frame_spacing * 0.80
    frame_left_margin = 0.06
    
    # Draw full frames
    for i, frame in enumerate(full_frames):
        x_pos = frame_left_margin + i * frame_spacing + frame_spacing * 0.10
        ax = fig.add_axes([x_pos, full_frame_y_bottom, frame_width, full_frame_height])
        ax.imshow(frame, cmap='gray', aspect='auto', interpolation='bilinear')
        ax.set_title(f't={i+1}', fontsize=9, pad=2, fontweight='bold')
        ax.axis('off')
    
    # Draw lip crops
    for i, lip_frame in enumerate(lip_frames):
        x_pos = frame_left_margin + i * frame_spacing + frame_spacing * 0.10
        ax = fig.add_axes([x_pos, lip_frame_y_bottom, frame_width, lip_frame_height])
        ax.imshow(lip_frame, cmap='gray', aspect='auto', interpolation='bilinear')
        ax.axis('off')
        for spine in ax.spines.values():
            spine.set_edgecolor('#2E7D9B')
            spine.set_linewidth(2)
    
    # Main axis
    ax_main = fig.add_axes([0, 0, 1, 1])
    ax_main.set_xlim(0, 1)
    ax_main.set_ylim(0, 1)
    ax_main.axis('off')
    
    # Arrows and characters
    for i in range(num_frames):
        x_center = frame_left_margin + i * frame_spacing + frame_spacing / 2
        
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
    
    # Separator
    ax_main.plot([frame_left_margin, 0.94], [separator_y, separator_y],
                 'k--', linewidth=0.8, alpha=0.3, zorder=1)
    
    # Final word
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
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0.02)
    print(f"✓ Saved: {output_path}")
    
    pdf_path = output_path.with_suffix('.pdf')
    plt.savefig(pdf_path, bbox_inches='tight', facecolor='white', pad_inches=0.02)
    print(f"✓ Saved: {pdf_path}")
    
    plt.close()


def main():
    print("Creating visualization with REAL matching transcript...")
    print("=" * 70)
    
    # Use the video we know
    base_path = Path("datasets/LRS2/mvlrs_v1/pretrain/6077571138829809070/00006")
    video_path = base_path.with_suffix('.mp4')
    mouth_path = Path(str(base_path) + "_mouth.mp4")
    txt_path = base_path.with_suffix('.txt')
    
    print(f"Reading transcript: {txt_path}")
    full_text, words = read_lrs2_transcript(txt_path)
    print(f"Full text: {full_text}")
    print(f"Found {len(words)} words")
    
    # Select a good word to visualize (short word)
    # Let's use "BIT" - it's short and clear
    target_word_obj = None
    for w in words:
        if w['word'] in ['BIT', 'FROM', 'PRISON']:
            target_word_obj = w
            break
    
    if not target_word_obj:
        target_word_obj = words[1]  # Use second word
    
    word = target_word_obj['word']
    start = target_word_obj['start']
    end = target_word_obj['end']
    
    print(f"\nUsing word: '{word}' (from {start:.2f}s to {end:.2f}s)")
    
    try:
        # Load frames for this specific word
        num_frames = 7
        full_frames = load_video_frames_for_word(video_path, start, end, num_frames)
        print(f"Loaded {len(full_frames)} full frames for word '{word}'")
        
        # Load lip frames for same word
        lip_frames = load_video_frames_for_word(mouth_path, start, end, num_frames)
        print(f"Loaded {len(lip_frames)} lip frames")
        
        # Create realistic alignment
        alignment = create_realistic_ctc_alignment(word, num_frames)
        print(f"Created alignment: {' '.join(alignment)}")
        
        # Create visualization
        output_path = Path(f"export/paper_samples/ctc_real_{word.lower()}.png")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        create_compact_visualization(
            full_frames=full_frames,
            lip_frames=lip_frames,
            alignment=alignment,
            word=word,
            output_path=output_path
        )
        
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("=" * 70)
    print("✅ Visualization with REAL matching transcript created!")
    print(f"\n📊 Word: {word}")
    print(f"   Timing: {start:.2f}s - {end:.2f}s")
    print(f"   Frames: {num_frames}")
    print(f"   Alignment: {' '.join(alignment)}")
    print(f"\nFile: export/paper_samples/ctc_real_{word.lower()}.png/pdf")
    print("\n✅ Now the lip movements MATCH the transcript!")


if __name__ == '__main__':
    main()

