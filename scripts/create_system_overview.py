#!/usr/bin/env python3
"""
Create a comprehensive system overview diagram showing the complete pipeline
from raw video to final text output.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle
from matplotlib.gridspec import GridSpec
import torch
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, str(Path(__file__).parent.parent))


# Color scheme
COLORS = {
    'preprocessing': '#E8F4F8',  # Light blue
    'visual_path': '#FFE6E6',    # Light red
    'fusion': '#E6F4E6',         # Light green
    'decoder': '#FFF4E6',        # Light orange
    'output': '#F0E6FF',         # Light purple
    'border_preprocessing': '#4A90A4',
    'border_visual': '#D64545',
    'border_fusion': '#45B545',
    'border_decoder': '#E8A845',
    'border_output': '#9B59B6',
    'arrow': '#555555',
    'text': '#2C3E50'
}


def load_sample_frames(dataset_root, speaker='s100', num_frames=5):
    """Load sample video frames from LipBengal dataset."""
    root = Path(dataset_root)
    speaker_dir = root / speaker
    
    if not speaker_dir.exists():
        return None, None, None
    
    # Find any word directory with frames
    for word_dir in speaker_dir.iterdir():
        if not word_dir.is_dir():
            continue
        frame_files = sorted(list(word_dir.glob("*.jpg")))
        if len(frame_files) >= num_frames:
            # Load frames
            indices = np.linspace(0, len(frame_files)-1, num_frames).astype(int)
            frames = []
            for idx in indices:
                img = Image.open(frame_files[idx])
                frames.append(np.array(img))
            
            # Try to load preprocessed lip crops
            bengali_word = word_dir.name
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
                            
                            # Ensure uint8
                            if lip_frames.dtype != np.uint8:
                                if lip_frames.max() <= 1.0:
                                    lip_frames = (lip_frames * 255).astype(np.uint8)
                                else:
                                    lip_frames = lip_frames.astype(np.uint8)
                            
                            # Sample to match num_frames
                            T = lip_frames.shape[0]
                            if T >= num_frames:
                                indices_lip = np.linspace(0, T-1, num_frames).astype(int)
                                lip_frames = lip_frames[indices_lip]
                            
                            return frames, lip_frames, bengali_word
                        except:
                            continue
    
    return None, None, None


def draw_rounded_box(ax, xy, width, height, label, color, border_color, fontsize=10):
    """Draw a rounded rectangle with label."""
    box = FancyBboxPatch(
        xy, width, height,
        boxstyle="round,pad=0.01",
        facecolor=color,
        edgecolor=border_color,
        linewidth=2.5,
        transform=ax.transAxes
    )
    ax.add_patch(box)
    
    # Add text
    ax.text(
        xy[0] + width/2, xy[1] + height/2,
        label,
        ha='center', va='center',
        fontsize=fontsize,
        fontweight='bold',
        color=COLORS['text'],
        transform=ax.transAxes
    )


def draw_arrow(ax, start, end, label='', color='#555555', style='->'):
    """Draw an arrow with optional label."""
    arrow = FancyArrowPatch(
        start, end,
        arrowstyle=style,
        color=color,
        linewidth=2.5,
        transform=ax.transAxes,
        mutation_scale=20,
        zorder=1
    )
    ax.add_patch(arrow)
    
    if label:
        mid_x = (start[0] + end[0]) / 2
        mid_y = (start[1] + end[1]) / 2
        ax.text(
            mid_x, mid_y + 0.02,
            label,
            ha='center', va='bottom',
            fontsize=8,
            color=color,
            transform=ax.transAxes,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='none', alpha=0.8)
        )


def create_system_overview(dataset_root='/home/thesis/Thesis/AVEC/datasets/LipBengal',
                          output_path='export/system_overview.png'):
    """Create the complete system overview diagram."""
    
    print("Creating system overview diagram...")
    
    # Load sample data
    print("Loading sample frames...")
    video_frames, lip_frames, word = load_sample_frames(dataset_root)
    
    if video_frames is None:
        print("Warning: Could not load sample frames, using placeholders")
        video_frames = [np.random.randint(0, 255, (160, 160, 3), dtype=np.uint8) for _ in range(5)]
        lip_frames = [np.random.randint(0, 255, (88, 88), dtype=np.uint8) for _ in range(5)]
        word = "অভিমত"
    
    # Create figure
    fig = plt.figure(figsize=(20, 14))
    gs = GridSpec(5, 5, figure=fig, hspace=0.35, wspace=0.3,
                  left=0.05, right=0.95, top=0.94, bottom=0.03)
    
    # Main title
    fig.suptitle('Visual-Only Lip Reading with Transfer Learning: Complete Pipeline Overview',
                 fontsize=18, fontweight='bold', color=COLORS['text'], y=0.97)
    
    # ===================================================================
    # SECTION 0: TRANSFER LEARNING OVERVIEW
    # ===================================================================
    
    ax_transfer = fig.add_subplot(gs[0, :])
    ax_transfer.set_xlim(0, 1)
    ax_transfer.set_ylim(0, 1)
    ax_transfer.axis('off')
    
    # Title
    ax_transfer.text(0.5, 0.95, 'TRANSFER LEARNING WORKFLOW',
                     ha='center', va='top', fontsize=14, fontweight='bold',
                     color='#C0392B', transform=ax_transfer.transAxes)
    
    # Phase 1: Pre-training
    phase1_color = '#E8F4F8'
    phase1_border = '#2980B9'
    draw_rounded_box(ax_transfer, (0.05, 0.35), 0.25, 0.45,
                     'PHASE 1:\nPRE-TRAINING\n\nSource Dataset:\nLRS2/LRS3\n(English)\n\n~150K samples',
                     phase1_color, phase1_border, fontsize=10)
    
    # Arrow with "Train Model"
    draw_arrow(ax_transfer, (0.30, 0.575), (0.40, 0.575), 
               label='Train Full Model', color='#2980B9')
    
    # Pre-trained model
    draw_rounded_box(ax_transfer, (0.40, 0.25), 0.18, 0.65,
                     'Visual Encoder\n\n3D CNN\n↓\nResNet-18\n↓\nConformer\n\nLearned Weights',
                     '#D5F4E6', '#27AE60', fontsize=9)
    
    # Arrow with "Transfer"
    draw_arrow(ax_transfer, (0.58, 0.575), (0.68, 0.575), 
               label='Transfer Weights', color='#E67E22', style='->')
    
    # Phase 2: Fine-tuning
    phase2_color = '#FFF4E6'
    phase2_border = '#E67E22'
    draw_rounded_box(ax_transfer, (0.68, 0.35), 0.27, 0.45,
                     'PHASE 2: FINE-TUNING\n\nTarget Dataset:\nLipBengal (Bengali)\nLRW-AR (Arabic)\n\n~1K-5K samples\n\nFreeze: 3D CNN + ResNet\nTrain: Conformer + Head',
                     phase2_color, phase2_border, fontsize=9)
    
    # Label sections
    ax_transfer.text(0.175, 0.15, 'Large-scale\nEnglish data',
                     ha='center', va='center', fontsize=8, style='italic',
                     color=phase1_border, transform=ax_transfer.transAxes)
    
    ax_transfer.text(0.815, 0.15, 'Low-resource\nTarget language',
                     ha='center', va='center', fontsize=8, style='italic',
                     color=phase2_border, transform=ax_transfer.transAxes)
    
    # ===================================================================
    # SECTION 1: INPUT & PREPROCESSING
    # ===================================================================
    
    # Video frames strip (top row)
    ax_video = fig.add_subplot(gs[1, :])
    ax_video.set_xlim(0, 1)
    ax_video.set_ylim(0, 1)
    ax_video.axis('off')
    
    # Title
    ax_video.text(0.5, 0.95, '① INPUT: Raw Video Sequence',
                  ha='center', va='top', fontsize=14, fontweight='bold',
                  color=COLORS['border_preprocessing'], transform=ax_video.transAxes)
    
    # Draw video frames
    frame_width = 0.15
    frame_spacing = 0.17
    start_x = 0.5 - (len(video_frames) * frame_spacing - (frame_spacing - frame_width)) / 2
    
    for i, frame in enumerate(video_frames):
        x_pos = start_x + i * frame_spacing
        
        # Add frame
        extent = [x_pos, x_pos + frame_width, 0.15, 0.75]
        ax_video.imshow(frame, extent=extent, aspect='auto', transform=ax_video.transAxes)
        
        # Frame border
        rect = Rectangle((x_pos, 0.15), frame_width, 0.6,
                        linewidth=2, edgecolor=COLORS['border_preprocessing'],
                        facecolor='none', transform=ax_video.transAxes)
        ax_video.add_patch(rect)
        
        # Frame number
        ax_video.text(x_pos + frame_width/2, 0.05,
                     f'Frame {i+1}',
                     ha='center', va='center', fontsize=9,
                     color=COLORS['text'], transform=ax_video.transAxes)
    
    # ===================================================================
    # SECTION 2: FACE DETECTION & LIP CROPPING
    # ===================================================================
    
    ax_preproc = fig.add_subplot(gs[2, :])
    ax_preproc.set_xlim(0, 1)
    ax_preproc.set_ylim(0, 1)
    ax_preproc.axis('off')
    
    # Title
    ax_preproc.text(0.5, 0.95, '② PREPROCESSING: Face Detection → Lip Region Extraction → Normalization',
                    ha='center', va='top', fontsize=14, fontweight='bold',
                    color=COLORS['border_preprocessing'], transform=ax_preproc.transAxes)
    
    # Processing steps
    step_y = 0.55
    
    # Step boxes
    draw_rounded_box(ax_preproc, (0.1, step_y), 0.18, 0.25,
                    'Face\nDetection\n(RetinaFace)', 
                    COLORS['preprocessing'], COLORS['border_preprocessing'], fontsize=9)
    
    draw_arrow(ax_preproc, (0.28, step_y + 0.125), (0.32, step_y + 0.125))
    
    draw_rounded_box(ax_preproc, (0.32, step_y), 0.18, 0.25,
                    'Landmark\nDetection\n(68 points)', 
                    COLORS['preprocessing'], COLORS['border_preprocessing'], fontsize=9)
    
    draw_arrow(ax_preproc, (0.50, step_y + 0.125), (0.54, step_y + 0.125))
    
    draw_rounded_box(ax_preproc, (0.54, step_y), 0.18, 0.25,
                    'Mouth Region\nCrop & Resize\n(88×88)', 
                    COLORS['preprocessing'], COLORS['border_preprocessing'], fontsize=9)
    
    draw_arrow(ax_preproc, (0.72, step_y + 0.125), (0.76, step_y + 0.125))
    
    draw_rounded_box(ax_preproc, (0.76, step_y), 0.14, 0.25,
                    'Grayscale\nNormalize', 
                    COLORS['preprocessing'], COLORS['border_preprocessing'], fontsize=9)
    
    # Show lip crops
    lip_width = 0.12
    lip_spacing = 0.14
    start_x_lip = 0.5 - (len(lip_frames) * lip_spacing - (lip_spacing - lip_width)) / 2
    
    for i in range(len(lip_frames)):
        x_pos = start_x_lip + i * lip_spacing
        
        # Add lip frame
        extent = [x_pos, x_pos + lip_width, 0.05, 0.45]
        if lip_frames is not None and i < len(lip_frames):
            ax_preproc.imshow(lip_frames[i], extent=extent, aspect='auto',
                            cmap='gray', transform=ax_preproc.transAxes)
        
        # Frame border
        rect = Rectangle((x_pos, 0.05), lip_width, 0.4,
                        linewidth=2, edgecolor=COLORS['border_preprocessing'],
                        facecolor='none', transform=ax_preproc.transAxes)
        ax_preproc.add_patch(rect)
    
    # ===================================================================
    # SECTION 3: MODEL ARCHITECTURE
    # ===================================================================
    
    ax_model = fig.add_subplot(gs[3:, :])
    ax_model.set_xlim(0, 1)
    ax_model.set_ylim(0, 1)
    ax_model.axis('off')
    
    # Title
    ax_model.text(0.5, 0.98, '③ VISUAL FEATURE NETWORK: 3D CNN → ResNet → Conformer Encoder → CTC Decoder',
                  ha='center', va='top', fontsize=14, fontweight='bold',
                  color=COLORS['border_visual'], transform=ax_model.transAxes)
    
    # Input representation
    ax_model.text(0.03, 0.7, 'Lip\nSequence\n(T×88×88)',
                 ha='center', va='center', fontsize=9,
                 color=COLORS['text'], transform=ax_model.transAxes,
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray',
                          edgecolor=COLORS['border_preprocessing'], linewidth=2))
    
    # Visual Feature Extraction
    draw_rounded_box(ax_model, (0.08, 0.55), 0.12, 0.35,
                    '3D CNN\n(Conv3D)\n↓\nMaxPool3D',
                    COLORS['visual_path'], COLORS['border_visual'], fontsize=9)
    
    # Add "Frozen" label
    ax_model.text(0.14, 0.50, '❄ Frozen',
                 ha='center', va='top', fontsize=7, fontweight='bold',
                 color='#2980B9', transform=ax_model.transAxes,
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='#E8F4F8', 
                          edgecolor='#2980B9', linewidth=1.5))
    
    draw_arrow(ax_model, (0.20, 0.725), (0.23, 0.725))
    
    draw_rounded_box(ax_model, (0.23, 0.55), 0.12, 0.35,
                    'ResNet-18\n(2D)\n↓\nFeature\nMaps',
                    COLORS['visual_path'], COLORS['border_visual'], fontsize=9)
    
    # Add "Frozen" label
    ax_model.text(0.29, 0.50, '❄ Frozen',
                 ha='center', va='top', fontsize=7, fontweight='bold',
                 color='#2980B9', transform=ax_model.transAxes,
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='#E8F4F8', 
                          edgecolor='#2980B9', linewidth=1.5))
    
    draw_arrow(ax_model, (0.35, 0.725), (0.38, 0.725), label='Visual Features')
    
    # Conformer Encoder
    draw_rounded_box(ax_model, (0.38, 0.50), 0.17, 0.45,
                    'Conformer\nEncoder\n\n• Multi-Head\n  Attention\n• Conv Module\n• Feed Forward\n• Layer Norm',
                    COLORS['fusion'], COLORS['border_fusion'], fontsize=8)
    
    # Add "Fine-tuned" label
    ax_model.text(0.465, 0.45, '🔥 Fine-tuned',
                 ha='center', va='top', fontsize=7, fontweight='bold',
                 color='#E67E22', transform=ax_model.transAxes,
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFF4E6', 
                          edgecolor='#E67E22', linewidth=1.5))
    
    # InterCTC blocks
    ax_model.text(0.465, 0.40, '↓ InterCTC',
                 ha='center', va='top', fontsize=7, style='italic',
                 color=COLORS['border_decoder'], transform=ax_model.transAxes)
    
    draw_arrow(ax_model, (0.55, 0.725), (0.59, 0.725), label='Encoded Features')
    
    # Sequence Decoder
    draw_rounded_box(ax_model, (0.59, 0.55), 0.14, 0.35,
                    'Multi-Stage\nTemporal\nConvolution\n(MS-TCN)',
                    COLORS['decoder'], COLORS['border_decoder'], fontsize=9)
    
    draw_arrow(ax_model, (0.73, 0.725), (0.77, 0.725))
    
    # CTC Head
    draw_rounded_box(ax_model, (0.77, 0.60), 0.10, 0.25,
                    'Linear\nProjection\n↓\nCTC Loss',
                    COLORS['decoder'], COLORS['border_decoder'], fontsize=9)
    
    # Add "Fine-tuned" label
    ax_model.text(0.82, 0.55, '🔥 Fine-tuned',
                 ha='center', va='top', fontsize=7, fontweight='bold',
                 color='#E67E22', transform=ax_model.transAxes,
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFF4E6', 
                          edgecolor='#E67E22', linewidth=1.5))
    
    draw_arrow(ax_model, (0.87, 0.725), (0.91, 0.725))
    
    # Output
    draw_rounded_box(ax_model, (0.91, 0.63), 0.07, 0.19,
                    'Softmax\n↓\nDecode',
                    COLORS['output'], COLORS['border_output'], fontsize=8)
    
    # ===================================================================
    # SECTION 4: CTC ALIGNMENT & OUTPUT
    # ===================================================================
    
    # CTC Alignment visualization
    ax_model.text(0.5, 0.40, '④ CTC ALIGNMENT & OUTPUT',
                  ha='center', va='top', fontsize=12, fontweight='bold',
                  color=COLORS['border_output'], transform=ax_model.transAxes)
    
    # Frame-to-character alignment
    ctc_chars = ['o', 'b', 'h', '-', 'i', 'm', 'o', 't'][:5]  # Example
    frame_y = 0.28
    char_y = 0.15
    
    ax_model.text(0.15, frame_y + 0.05, 'Video Frames:',
                 ha='right', va='center', fontsize=9, fontweight='bold',
                 color=COLORS['text'], transform=ax_model.transAxes)
    
    ax_model.text(0.15, char_y + 0.05, 'CTC Output:',
                 ha='right', va='center', fontsize=9, fontweight='bold',
                 color=COLORS['text'], transform=ax_model.transAxes)
    
    # Draw alignment
    align_start_x = 0.20
    align_width = 0.08
    align_spacing = 0.10
    
    for i in range(5):
        x_pos = align_start_x + i * align_spacing
        
        # Frame box
        rect = Rectangle((x_pos, frame_y), align_width, 0.08,
                        linewidth=2, edgecolor=COLORS['border_preprocessing'],
                        facecolor=COLORS['preprocessing'], transform=ax_model.transAxes)
        ax_model.add_patch(rect)
        ax_model.text(x_pos + align_width/2, frame_y + 0.04,
                     f't={i+1}',
                     ha='center', va='center', fontsize=8,
                     color=COLORS['text'], transform=ax_model.transAxes)
        
        # Character box
        char_box_color = COLORS['output'] if ctc_chars[i] != '-' else 'white'
        char_border = COLORS['border_output'] if ctc_chars[i] != '-' else '#CCCCCC'
        rect = Rectangle((x_pos, char_y), align_width, 0.08,
                        linewidth=2, edgecolor=char_border,
                        facecolor=char_box_color, transform=ax_model.transAxes)
        ax_model.add_patch(rect)
        
        char_display = ctc_chars[i] if ctc_chars[i] != '-' else '∅'
        ax_model.text(x_pos + align_width/2, char_y + 0.04,
                     f'{char_display}',
                     ha='center', va='center', fontsize=10,
                     fontweight='bold', color=COLORS['text'],
                     transform=ax_model.transAxes)
        
        # Connection arrow
        arrow = FancyArrowPatch(
            (x_pos + align_width/2, frame_y),
            (x_pos + align_width/2, char_y + 0.08),
            arrowstyle='->',
            color=char_border,
            linewidth=1.5,
            transform=ax_model.transAxes,
            alpha=0.6
        )
        ax_model.add_patch(arrow)
    
    # Final output
    ax_model.text(0.72, 0.19, 'Final Prediction:',
                 ha='right', va='center', fontsize=10, fontweight='bold',
                 color=COLORS['text'], transform=ax_model.transAxes)
    
    prediction_text = "abhimot"  # Example transliteration
    rect = FancyBboxPatch((0.73, 0.12), 0.20, 0.14,
                         boxstyle="round,pad=0.02",
                         facecolor=COLORS['output'],
                         edgecolor=COLORS['border_output'],
                         linewidth=3,
                         transform=ax_model.transAxes)
    ax_model.add_patch(rect)
    
    ax_model.text(0.83, 0.19, f'"{prediction_text}"',
                 ha='center', va='center', fontsize=14,
                 fontweight='bold', style='italic',
                 color=COLORS['border_output'],
                 transform=ax_model.transAxes)
    
    if word:
        ax_model.text(0.83, 0.12, f'(Bengali: {word})',
                     ha='center', va='bottom', fontsize=8,
                     color=COLORS['text'], style='italic',
                     transform=ax_model.transAxes)
    
    # Legend
    legend_y = 0.03
    legend_elements = [
        ('Preprocessing', COLORS['preprocessing'], COLORS['border_preprocessing']),
        ('Feature Extraction', COLORS['visual_path'], COLORS['border_visual']),
        ('Sequence Modeling', COLORS['fusion'], COLORS['border_fusion']),
        ('Decoding', COLORS['decoder'], COLORS['border_decoder']),
        ('Output', COLORS['output'], COLORS['border_output'])
    ]
    
    legend_start = 0.15
    legend_spacing = 0.15
    
    for i, (label, face_color, edge_color) in enumerate(legend_elements):
        x_pos = legend_start + i * legend_spacing
        
        rect = Rectangle((x_pos, legend_y), 0.025, 0.025,
                        facecolor=face_color, edgecolor=edge_color,
                        linewidth=2, transform=ax_model.transAxes)
        ax_model.add_patch(rect)
        
        ax_model.text(x_pos + 0.03, legend_y + 0.0125,
                     label,
                     ha='left', va='center', fontsize=7,
                     color=COLORS['text'], transform=ax_model.transAxes)
    
    # Save figure
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"\n✓ System overview saved to: {output_file}")
    print(f"  Resolution: 300 DPI")
    print(f"  Format: PNG\n")
    
    # Also save as PDF for publication quality
    pdf_file = output_file.with_suffix('.pdf')
    plt.savefig(pdf_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ PDF version saved to: {pdf_file}\n")
    
    plt.close()
    
    return output_file


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Create system overview diagram')
    parser.add_argument('--dataset', type=str,
                       default='/home/thesis/Thesis/AVEC/datasets/LipBengal',
                       help='Path to dataset')
    parser.add_argument('--output', type=str,
                       default='export/system_overview.png',
                       help='Output file path')
    
    args = parser.parse_args()
    
    try:
        create_system_overview(args.dataset, args.output)
        print("System overview diagram created successfully!")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

