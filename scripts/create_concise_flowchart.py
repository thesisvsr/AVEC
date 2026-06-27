#!/usr/bin/env python3
"""
Create a single concise flowchart with everything.
"""

import sys
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.gridspec import GridSpec

# Colors
COLORS = {
    'pretrain': '#3498DB',
    'finetune': '#E67E22',
    'frozen': '#BDC3C7',
    'active': '#27AE60',
    'data': '#9B59B6',
    'output': '#E74C3C',
    'arrow': '#34495E',
    'text': '#2C3E50'
}


def draw_box(ax, xy, w, h, text, bg, border, fs=9, fw='normal'):
    """Draw a box."""
    box = FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.01",
                         facecolor=bg, edgecolor=border, linewidth=2.5,
                         transform=ax.transAxes, zorder=2)
    ax.add_patch(box)
    ax.text(xy[0]+w/2, xy[1]+h/2, text, ha='center', va='center',
           fontsize=fs, fontweight=fw, color=COLORS['text'],
           transform=ax.transAxes, zorder=3)


def draw_arrow(ax, x1, y1, x2, y2, color, label=''):
    """Draw arrow."""
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                           arrowstyle='->,head_width=0.3,head_length=0.2',
                           color=color, linewidth=3,
                           transform=ax.transAxes, zorder=1, mutation_scale=20)
    ax.add_patch(arrow)
    if label:
        ax.text((x1+x2)/2, (y1+y2)/2+0.01, label, ha='center', va='bottom',
               fontsize=7, fontweight='bold', color=color,
               transform=ax.transAxes, zorder=3,
               bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                        edgecolor=color, linewidth=1.5))


def create_concise_flowchart(output_path='export/concise_flowchart.png'):
    """Create single concise flowchart."""
    
    print("\n" + "="*70)
    print("CREATING CONCISE FLOWCHART")
    print("="*70 + "\n")
    
    fig = plt.figure(figsize=(16, 11))
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # Title
    fig.text(0.5, 0.97, 'Visual-Only Lip Reading with Transfer Learning: Complete Flow',
             ha='center', va='top', fontsize=16, fontweight='bold', color=COLORS['text'])
    
    # Y positions
    y = 0.88
    dy = 0.10
    
    # =====================================================================
    # TRANSFER LEARNING
    # =====================================================================
    
    # Phase 1: Pre-training
    draw_box(ax, (0.05, y), 0.25, 0.07,
             'PHASE 1: PRE-TRAINING\nLRS2/LRS3 (English)\n150K samples',
             '#EBF5FB', COLORS['pretrain'], 8, 'bold')
    
    draw_arrow(ax, 0.30, y+0.035, 0.37, y+0.035, COLORS['pretrain'], 'Train All')
    
    # Trained model
    draw_box(ax, (0.37, y), 0.26, 0.07,
             'Train Complete Model\n3D CNN → ResNet → Conformer → Head',
             '#E8F8F5', COLORS['active'], 8, 'bold')
    
    draw_arrow(ax, 0.63, y+0.035, 0.70, y+0.035, COLORS['finetune'], 'Transfer')
    
    # Phase 2: Fine-tuning
    draw_box(ax, (0.70, y), 0.25, 0.07,
             'PHASE 2: FINE-TUNING\nBengali/Arabic\n1K-5K samples',
             '#FEF5E7', COLORS['finetune'], 8, 'bold')
    
    y -= dy
    
    # =====================================================================
    # INPUT
    # =====================================================================
    
    ax.text(0.5, y+0.03, '▼ DATA FLOW ▼', ha='center', va='center',
           fontsize=11, fontweight='bold', color=COLORS['text'],
           transform=ax.transAxes)
    
    y -= 0.06
    
    draw_box(ax, (0.30, y), 0.40, 0.06,
             'INPUT: Raw Video Frames (T × H × W × 3)',
             '#F4ECF7', COLORS['data'], 9)
    
    draw_arrow(ax, 0.5, y, 0.5, y-0.04, COLORS['arrow'])
    
    y -= 0.09
    
    # =====================================================================
    # PREPROCESSING
    # =====================================================================
    
    draw_box(ax, (0.25, y), 0.50, 0.06,
             'PREPROCESSING: Face Detection → Landmarks → Lip Crop → Normalize',
             '#F4ECF7', COLORS['data'], 8)
    
    draw_arrow(ax, 0.5, y, 0.5, y-0.04, COLORS['arrow'])
    
    y -= 0.09
    
    draw_box(ax, (0.35, y), 0.30, 0.05,
             'Lip Sequence (T × 88 × 88)',
             '#F4ECF7', COLORS['data'], 9)
    
    draw_arrow(ax, 0.5, y, 0.5, y-0.04, COLORS['arrow'])
    
    y -= 0.08
    
    # =====================================================================
    # MODEL ARCHITECTURE
    # =====================================================================
    
    ax.text(0.5, y+0.01, '━━━━━━ VISUAL ENCODER ━━━━━━',
           ha='center', va='center', fontsize=11, fontweight='bold',
           color=COLORS['text'], transform=ax.transAxes)
    
    y -= 0.05
    
    # Architecture components
    comp_w = 0.18
    comp_h = 0.09
    spacing = 0.20
    x_start = 0.5 - (4*spacing - spacing + comp_w)/2
    
    components = [
        ('3D CNN\nConv3D\nMaxPool', True, '#ECF0F1', COLORS['frozen'], '❄'),
        ('ResNet-18\nResidual\nBlocks', True, '#ECF0F1', COLORS['frozen'], '❄'),
        ('Conformer\nAttention\nConv+FF', False, '#E8F8F5', COLORS['active'], '🔥'),
        ('CTC Head\nLinear\nSoftmax', False, '#E8F8F5', COLORS['active'], '🔥')
    ]
    
    for i, (name, frozen, bg, border, icon) in enumerate(components):
        x = x_start + i*spacing
        
        draw_box(ax, (x, y), comp_w, comp_h, name, bg, border, 8, 'bold')
        
        # Status
        status = 'FROZEN' if frozen else 'FINE-TUNED'
        color = COLORS['pretrain'] if frozen else COLORS['finetune']
        ax.text(x+comp_w/2, y+comp_h+0.005, f'{icon} {status}',
               ha='center', va='bottom', fontsize=6, fontweight='bold',
               color=color, transform=ax.transAxes,
               bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                        edgecolor=color, linewidth=1.5))
        
        # Arrow
        if i < len(components)-1:
            draw_arrow(ax, x+comp_w+0.005, y+comp_h/2,
                      x_start+(i+1)*spacing-0.005, y+comp_h/2,
                      COLORS['arrow'])
    
    draw_arrow(ax, 0.5, y, 0.5, y-0.04, COLORS['arrow'])
    
    y -= 0.08
    
    # =====================================================================
    # OUTPUT
    # =====================================================================
    
    draw_box(ax, (0.30, y), 0.40, 0.05,
             'CTC Decoding (Beam Search)',
             '#FADBD8', COLORS['output'], 9)
    
    draw_arrow(ax, 0.5, y, 0.5, y-0.04, COLORS['arrow'])
    
    y -= 0.08
    
    draw_box(ax, (0.33, y), 0.34, 0.06,
             'OUTPUT: Predicted Text (Transliterated)',
             '#FADBD8', COLORS['output'], 9, 'bold')
    
    # =====================================================================
    # KEY INFORMATION (BOTTOM)
    # =====================================================================
    
    info_y = 0.08
    
    # Left: Architecture specs
    ax.text(0.02, info_y+0.04, 'ARCHITECTURE:', ha='left', va='top',
           fontsize=8, fontweight='bold', color=COLORS['text'],
           transform=ax.transAxes)
    
    specs = '''3D CNN: Conv3D(64, 5×7×7) + MaxPool3D
ResNet-18: 4 blocks, 256-dim output
Conformer: 6+6 blocks, 360-dim, 4 heads
CTC Head: Linear(360→vocab) + CTC Loss'''
    
    ax.text(0.02, info_y, specs, ha='left', va='top',
           fontsize=6, color=COLORS['text'], transform=ax.transAxes,
           linespacing=1.5, family='monospace')
    
    # Center: Transfer learning strategy
    ax.text(0.38, info_y+0.04, 'TRANSFER LEARNING:', ha='left', va='top',
           fontsize=8, fontweight='bold', color=COLORS['text'],
           transform=ax.transAxes)
    
    strategy = '''Phase 1: Train all layers on 150K English samples
Phase 2: Freeze CNN+ResNet, fine-tune Conformer+Head
         on 1K-5K Bengali/Arabic samples (10x less data)'''
    
    ax.text(0.38, info_y, strategy, ha='left', va='top',
           fontsize=6, color=COLORS['text'], transform=ax.transAxes,
           linespacing=1.5)
    
    # Right: Legend
    ax.text(0.78, info_y+0.04, 'LEGEND:', ha='left', va='top',
           fontsize=8, fontweight='bold', color=COLORS['text'],
           transform=ax.transAxes)
    
    legend_items = [
        ('Pre-training', COLORS['pretrain']),
        ('Fine-tuning', COLORS['finetune']),
        ('Frozen (❄)', COLORS['frozen']),
        ('Trained (🔥)', COLORS['active'])
    ]
    
    y_leg = info_y - 0.005
    for label, color in legend_items:
        rect = Rectangle((0.78, y_leg), 0.015, 0.015,
                        facecolor=color, edgecolor=color,
                        linewidth=2, transform=ax.transAxes)
        ax.add_patch(rect)
        ax.text(0.798, y_leg+0.0075, label, ha='left', va='center',
               fontsize=6, color=COLORS['text'], transform=ax.transAxes)
        y_leg -= 0.02
    
    # Save
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Concise flowchart saved to: {output_file}")
    
    pdf_file = output_file.with_suffix('.pdf')
    plt.savefig(pdf_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ PDF version saved to: {pdf_file}")
    
    print("\n" + "="*70)
    print("✅ CONCISE FLOWCHART COMPLETE")
    print("="*70 + "\n")
    
    plt.close()
    return output_file


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Create concise flowchart')
    parser.add_argument('--output', type=str,
                       default='export/concise_flowchart.png')
    
    args = parser.parse_args()
    
    try:
        create_concise_flowchart(args.output)
        print("Concise flowchart created successfully!")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)





