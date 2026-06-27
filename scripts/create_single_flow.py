#!/usr/bin/env python3
"""
Create a compact single-flow diagram of the visual-only transfer learning system.
Shows the complete pipeline in one streamlined view.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle, Polygon
from matplotlib.gridspec import GridSpec
import torch
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent))


# Color scheme
COLORS = {
    'pretrain': '#3498DB',      # Blue for pre-training
    'finetune': '#E67E22',      # Orange for fine-tuning
    'frozen': '#95A5A6',        # Gray for frozen
    'active': '#27AE60',        # Green for active/fine-tuned
    'preprocessing': '#9B59B6', # Purple for preprocessing
    'output': '#E74C3C',        # Red for output
    'arrow': '#2C3E50',
    'text': '#2C3E50',
    'light_blue': '#EBF5FB',
    'light_orange': '#FEF5E7',
    'light_green': '#E8F8F5',
    'light_purple': '#F4ECF7',
    'light_red': '#FADBD8'
}


def load_sample_frames(dataset_root, num_frames=3):
    """Load sample video frames from LipBengal dataset."""
    root = Path(dataset_root)
    
    for speaker_dir in root.iterdir():
        if not speaker_dir.is_dir() or speaker_dir.name.startswith('.'):
            continue
        
        for word_dir in speaker_dir.iterdir():
            if not word_dir.is_dir():
                continue
            frame_files = sorted(list(word_dir.glob("*.jpg")))
            if len(frame_files) >= num_frames:
                indices = np.linspace(0, len(frame_files)-1, num_frames).astype(int)
                frames = []
                for idx in indices:
                    img = Image.open(frame_files[idx])
                    frames.append(np.array(img))
                
                # Try to load lip crops
                bengali_word = word_dir.name
                speaker = speaker_dir.name
                for split in ['train', 'val', 'test']:
                    prepared_dir = root / 'prepared' / split / speaker / bengali_word
                    if prepared_dir.exists():
                        pt_files = list(prepared_dir.glob("*.pt"))
                        if pt_files:
                            try:
                                obj = torch.load(pt_files[0], map_location='cpu')
                                if isinstance(obj, dict) and 'frames' in obj:
                                    lip_frames = obj['frames']
                                elif isinstance(obj, dict) and 'x' in obj:
                                    lip_frames = obj['x']
                                else:
                                    continue
                                
                                if isinstance(lip_frames, torch.Tensor):
                                    lip_frames = lip_frames.numpy()
                                
                                if lip_frames.dtype != np.uint8:
                                    if lip_frames.max() <= 1.0:
                                        lip_frames = (lip_frames * 255).astype(np.uint8)
                                    else:
                                        lip_frames = lip_frames.astype(np.uint8)
                                
                                T = lip_frames.shape[0]
                                if T >= num_frames:
                                    indices_lip = np.linspace(0, T-1, num_frames).astype(int)
                                    lip_frames = lip_frames[indices_lip]
                                
                                return frames, lip_frames, bengali_word
                            except:
                                continue
    
    return None, None, None


def draw_box(ax, xy, width, height, text, bg_color, border_color, fontsize=9, fontweight='normal'):
    """Draw a simple rounded box."""
    box = FancyBboxPatch(
        xy, width, height,
        boxstyle="round,pad=0.008",
        facecolor=bg_color,
        edgecolor=border_color,
        linewidth=2.5,
        transform=ax.transAxes,
        zorder=2
    )
    ax.add_patch(box)
    
    # Split text by newline and render
    lines = text.split('\n')
    y_offset = xy[1] + height/2
    
    ax.text(
        xy[0] + width/2, y_offset,
        text,
        ha='center', va='center',
        fontsize=fontsize,
        fontweight=fontweight,
        color=COLORS['text'],
        transform=ax.transAxes,
        zorder=3
    )


def draw_thick_arrow(ax, x1, y1, x2, y2, color, label='', width=0.01):
    """Draw a thick arrow."""
    arrow = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle='->,head_width=0.4,head_length=0.3',
        color=color,
        linewidth=3,
        transform=ax.transAxes,
        zorder=1,
        mutation_scale=30
    )
    ax.add_patch(arrow)
    
    if label:
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2
        ax.text(
            mid_x, mid_y + 0.03,
            label,
            ha='center', va='bottom',
            fontsize=7,
            fontweight='bold',
            color=color,
            transform=ax.transAxes,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                     edgecolor=color, linewidth=1.5),
            zorder=3
        )


def create_single_flow(dataset_root='/home/thesis/Thesis/AVEC/datasets/LipBengal',
                      output_path='export/single_flow.png'):
    """Create compact single-flow diagram."""
    
    print("Creating single-flow diagram...")
    
    # Load sample data
    print("Loading sample frames...")
    video_frames, lip_frames, word = load_sample_frames(dataset_root)
    
    if video_frames is None:
        print("Warning: Could not load samples, using placeholders")
        video_frames = [np.random.randint(0, 255, (160, 160, 3), dtype=np.uint8) for _ in range(3)]
        lip_frames = [np.random.randint(0, 255, (88, 88), dtype=np.uint8) for _ in range(3)]
        word = "example"
    
    # Create figure
    fig = plt.figure(figsize=(18, 10))
    
    # Main axis
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # =========================================================================
    # TITLE
    # =========================================================================
    fig.text(0.5, 0.97, 'Visual-Only Lip Reading: Transfer Learning Pipeline',
             ha='center', va='top', fontsize=18, fontweight='bold', color=COLORS['text'])
    
    # =========================================================================
    # SECTION 1: TRANSFER LEARNING STRATEGY (TOP)
    # =========================================================================
    
    y_transfer = 0.75
    
    # Phase 1: Pre-training
    draw_box(ax, (0.05, y_transfer), 0.22, 0.15,
             'PHASE 1: PRE-TRAINING\n\nLRS2/LRS3 (English)\n150K samples\nTrain all layers',
             COLORS['light_blue'], COLORS['pretrain'], fontsize=8, fontweight='bold')
    
    # Arrow
    draw_thick_arrow(ax, 0.27, y_transfer + 0.075, 0.36, y_transfer + 0.075,
                    COLORS['pretrain'], 'Train')
    
    # Learned model
    draw_box(ax, (0.36, y_transfer - 0.02), 0.28, 0.19,
             'TRAINED MODEL\n\n3D CNN → ResNet-18 → Conformer → CTC Head\n\nLearned Visual Features',
             COLORS['light_green'], COLORS['active'], fontsize=8, fontweight='bold')
    
    # Arrow
    draw_thick_arrow(ax, 0.64, y_transfer + 0.075, 0.73, y_transfer + 0.075,
                    COLORS['finetune'], 'Transfer')
    
    # Phase 2: Fine-tuning
    draw_box(ax, (0.73, y_transfer), 0.22, 0.15,
             'PHASE 2: FINE-TUNING\n\nBengali/Arabic\n1K-5K samples\nFreeze CNN+ResNet',
             COLORS['light_orange'], COLORS['finetune'], fontsize=8, fontweight='bold')
    
    # =========================================================================
    # SECTION 2: SINGLE FLOW PIPELINE (MIDDLE TO BOTTOM)
    # =========================================================================
    
    y_start = 0.60
    
    # Step numbers
    step_y = y_start + 0.05
    
    # -------------------------------------------------------------------------
    # STEP 1: INPUT VIDEO
    # -------------------------------------------------------------------------
    
    ax.text(0.09, step_y, '❶', ha='center', va='center',
           fontsize=20, fontweight='bold', color=COLORS['pretrain'],
           transform=ax.transAxes)
    
    ax.text(0.09, step_y - 0.06, 'INPUT', ha='center', va='top',
           fontsize=10, fontweight='bold', color=COLORS['text'],
           transform=ax.transAxes)
    
    # Video frames
    frame_size = 0.08
    frame_y = 0.42
    for i, frame in enumerate(video_frames):
        x_pos = 0.05 + i * 0.03
        extent = [x_pos, x_pos + frame_size, frame_y, frame_y + frame_size]
        ax.imshow(frame, extent=extent, aspect='auto', transform=ax.transAxes, zorder=2)
        rect = Rectangle((x_pos, frame_y), frame_size, frame_size,
                        linewidth=2, edgecolor=COLORS['pretrain'],
                        facecolor='none', transform=ax.transAxes, zorder=3)
        ax.add_patch(rect)
    
    ax.text(0.09, frame_y - 0.02, 'Raw Video\nFrames', ha='center', va='top',
           fontsize=7, color=COLORS['text'], transform=ax.transAxes)
    
    # Arrow to preprocessing
    draw_thick_arrow(ax, 0.16, step_y - 0.03, 0.20, step_y - 0.03,
                    COLORS['arrow'])
    
    # -------------------------------------------------------------------------
    # STEP 2: PREPROCESSING
    # -------------------------------------------------------------------------
    
    ax.text(0.28, step_y, '❷', ha='center', va='center',
           fontsize=20, fontweight='bold', color=COLORS['preprocessing'],
           transform=ax.transAxes)
    
    ax.text(0.28, step_y - 0.06, 'PREPROCESS', ha='center', va='top',
           fontsize=10, fontweight='bold', color=COLORS['text'],
           transform=ax.transAxes)
    
    # Preprocessing steps
    preproc_y = 0.42
    draw_box(ax, (0.20, preproc_y + 0.05), 0.16, 0.05,
             'Face Detection → Landmarks → Crop Lips',
             COLORS['light_purple'], COLORS['preprocessing'], fontsize=7)
    
    # Lip frames
    lip_size = 0.045
    lip_y = 0.42
    for i in range(len(lip_frames)):
        x_pos = 0.22 + i * 0.05
        extent = [x_pos, x_pos + lip_size, lip_y, lip_y + lip_size]
        if lip_frames is not None and i < len(lip_frames):
            ax.imshow(lip_frames[i], extent=extent, aspect='auto',
                     cmap='gray', transform=ax.transAxes, zorder=2)
        rect = Rectangle((x_pos, lip_y), lip_size, lip_size,
                        linewidth=2, edgecolor=COLORS['preprocessing'],
                        facecolor='none', transform=ax.transAxes, zorder=3)
        ax.add_patch(rect)
    
    ax.text(0.28, lip_y - 0.02, 'Lip Crops\n88×88', ha='center', va='top',
           fontsize=7, color=COLORS['text'], transform=ax.transAxes)
    
    # Arrow to model
    draw_thick_arrow(ax, 0.37, step_y - 0.03, 0.41, step_y - 0.03,
                    COLORS['arrow'])
    
    # -------------------------------------------------------------------------
    # STEP 3: VISUAL ENCODER (with freeze/finetune indicators)
    # -------------------------------------------------------------------------
    
    ax.text(0.58, step_y, '❸', ha='center', va='center',
           fontsize=20, fontweight='bold', color=COLORS['active'],
           transform=ax.transAxes)
    
    ax.text(0.58, step_y - 0.06, 'VISUAL ENCODER', ha='center', va='top',
           fontsize=10, fontweight='bold', color=COLORS['text'],
           transform=ax.transAxes)
    
    # Model components
    comp_y = 0.42
    comp_width = 0.075
    comp_height = 0.08
    spacing = 0.08
    
    components = [
        ('3D\nCNN', COLORS['frozen'], '❄'),
        ('ResNet\n18', COLORS['frozen'], '❄'),
        ('Conformer\nEncoder', COLORS['active'], '🔥'),
        ('CTC\nHead', COLORS['active'], '🔥')
    ]
    
    start_x = 0.43
    for i, (name, color, icon) in enumerate(components):
        x = start_x + i * spacing
        
        # Box
        box_color = COLORS['light_blue'] if color == COLORS['frozen'] else COLORS['light_green']
        draw_box(ax, (x, comp_y), comp_width, comp_height, name, box_color, color, fontsize=7)
        
        # Icon
        ax.text(x + comp_width/2, comp_y + comp_height + 0.01,
               icon, ha='center', va='bottom', fontsize=8,
               transform=ax.transAxes)
        
        # Arrow between components
        if i < len(components) - 1:
            ax.plot([x + comp_width + 0.001, x + comp_width + 0.005],
                   [comp_y + comp_height/2, comp_y + comp_height/2],
                   color=COLORS['arrow'], linewidth=2, transform=ax.transAxes, zorder=1)
    
    # Labels
    ax.text(start_x + 0.04, comp_y - 0.02, 'Frozen\n(Pre-trained)',
           ha='center', va='top', fontsize=6, color=COLORS['frozen'],
           transform=ax.transAxes)
    
    ax.text(start_x + 0.20, comp_y - 0.02, 'Fine-tuned\n(Target Task)',
           ha='center', va='top', fontsize=6, color=COLORS['active'],
           transform=ax.transAxes)
    
    # Arrow to output
    draw_thick_arrow(ax, 0.75, step_y - 0.03, 0.79, step_y - 0.03,
                    COLORS['arrow'])
    
    # -------------------------------------------------------------------------
    # STEP 4: OUTPUT
    # -------------------------------------------------------------------------
    
    ax.text(0.87, step_y, '❹', ha='center', va='center',
           fontsize=20, fontweight='bold', color=COLORS['output'],
           transform=ax.transAxes)
    
    ax.text(0.87, step_y - 0.06, 'OUTPUT', ha='center', va='top',
           fontsize=10, fontweight='bold', color=COLORS['text'],
           transform=ax.transAxes)
    
    # CTC Decoding
    output_y = 0.46
    draw_box(ax, (0.79, output_y), 0.16, 0.045,
             'CTC Decoding → Text Output',
             COLORS['light_red'], COLORS['output'], fontsize=7)
    
    # Final prediction
    prediction = "abhimot"
    draw_box(ax, (0.81, 0.40), 0.12, 0.055,
             f'"{prediction}"',
             'white', COLORS['output'], fontsize=11, fontweight='bold')
    
    if word:
        ax.text(0.87, 0.395, f'({word})',
               ha='center', va='top', fontsize=7, style='italic',
               color=COLORS['text'], transform=ax.transAxes)
    
    # =========================================================================
    # SECTION 3: DATA FLOW VISUALIZATION (BOTTOM)
    # =========================================================================
    
    flow_y = 0.28
    
    ax.text(0.5, flow_y + 0.05, 'Data Flow & CTC Alignment',
           ha='center', va='top', fontsize=11, fontweight='bold',
           color=COLORS['text'], transform=ax.transAxes)
    
    # Timeline representation
    timeline_y = flow_y - 0.02
    num_steps = 6
    step_width = 0.08
    start_x_timeline = 0.5 - (num_steps * step_width) / 2
    
    # Sample CTC alignment
    ctc_outputs = ['o', 'b', 'h', '-', 'i', 'm']
    
    for i in range(num_steps):
        x = start_x_timeline + i * step_width
        
        # Time step box
        ax.text(x + step_width/2, timeline_y, f't={i+1}',
               ha='center', va='center', fontsize=7,
               color=COLORS['text'], transform=ax.transAxes)
        
        # Character output
        char = ctc_outputs[i] if i < len(ctc_outputs) else '-'
        char_color = COLORS['output'] if char != '-' else COLORS['frozen']
        bg_color = COLORS['light_red'] if char != '-' else 'white'
        
        char_display = char if char != '-' else '∅'
        
        rect = Rectangle((x, timeline_y - 0.05), step_width * 0.9, 0.04,
                        facecolor=bg_color, edgecolor=char_color,
                        linewidth=2, transform=ax.transAxes)
        ax.add_patch(rect)
        
        ax.text(x + step_width/2, timeline_y - 0.03,
               char_display, ha='center', va='center',
               fontsize=10, fontweight='bold', color=COLORS['text'],
               transform=ax.transAxes)
    
    # Labels
    ax.text(start_x_timeline - 0.05, timeline_y, 'Time:',
           ha='right', va='center', fontsize=8, fontweight='bold',
           color=COLORS['text'], transform=ax.transAxes)
    
    ax.text(start_x_timeline - 0.05, timeline_y - 0.03, 'CTC:',
           ha='right', va='center', fontsize=8, fontweight='bold',
           color=COLORS['text'], transform=ax.transAxes)
    
    # =========================================================================
    # LEGEND
    # =========================================================================
    
    legend_y = 0.08
    
    legend_items = [
        ('Pre-training (English)', COLORS['light_blue'], COLORS['pretrain']),
        ('Fine-tuning (Target)', COLORS['light_orange'], COLORS['finetune']),
        ('Frozen Layers', COLORS['light_blue'], COLORS['frozen']),
        ('Fine-tuned Layers', COLORS['light_green'], COLORS['active']),
    ]
    
    legend_start = 0.25
    legend_spacing = 0.14
    
    for i, (label, bg, border) in enumerate(legend_items):
        x = legend_start + i * legend_spacing
        
        rect = Rectangle((x, legend_y), 0.02, 0.02,
                        facecolor=bg, edgecolor=border,
                        linewidth=2, transform=ax.transAxes)
        ax.add_patch(rect)
        
        ax.text(x + 0.025, legend_y + 0.01, label,
               ha='left', va='center', fontsize=7,
               color=COLORS['text'], transform=ax.transAxes)
    
    # =========================================================================
    # KEY INSIGHTS BOX
    # =========================================================================
    
    insights_y = 0.02
    ax.text(0.5, insights_y, 
           'Transfer Learning: Train on 150K English samples → Fine-tune on 1K-5K target samples | '
           'Freeze low-level features (CNN) → Adapt high-level features (Conformer)',
           ha='center', va='bottom', fontsize=7, style='italic',
           color=COLORS['text'], transform=ax.transAxes,
           bbox=dict(boxstyle='round,pad=0.5', facecolor='#F8F9F9',
                    edgecolor=COLORS['text'], linewidth=1))
    
    # Save
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n✓ Single-flow diagram saved to: {output_file}")
    print(f"  Resolution: 300 DPI\n")
    
    # PDF version
    pdf_file = output_file.with_suffix('.pdf')
    plt.savefig(pdf_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ PDF version saved to: {pdf_file}\n")
    
    plt.close()
    
    return output_file


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Create single-flow diagram')
    parser.add_argument('--dataset', type=str,
                       default='/home/thesis/Thesis/AVEC/datasets/LipBengal',
                       help='Path to dataset')
    parser.add_argument('--output', type=str,
                       default='export/single_flow.png',
                       help='Output file path')
    
    args = parser.parse_args()
    
    try:
        create_single_flow(args.dataset, args.output)
        print("Single-flow diagram created successfully!")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)





