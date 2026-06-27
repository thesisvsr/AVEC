#!/usr/bin/env python3
"""
Create a detailed architecture flowchart with transfer learning.
Shows the complete model architecture and transfer learning strategy.
"""

import sys
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Circle, Polygon
from matplotlib.gridspec import GridSpec

# Color scheme
COLORS = {
    'phase1': '#3498DB',        # Blue - Pre-training
    'phase2': '#E67E22',        # Orange - Fine-tuning
    'frozen': '#95A5A6',        # Gray - Frozen layers
    'trainable': '#27AE60',     # Green - Trainable layers
    'data': '#9B59B6',          # Purple - Data
    'output': '#E74C3C',        # Red - Output
    'arrow': '#34495E',
    'text': '#2C3E50',
    'light_blue': '#EBF5FB',
    'light_orange': '#FEF5E7',
    'light_gray': '#ECF0F1',
    'light_green': '#E8F8F5',
    'light_purple': '#F4ECF7',
    'light_red': '#FADBD8'
}


def draw_box(ax, xy, width, height, text, bg_color, border_color, fontsize=9, 
             fontweight='normal', text_color=None):
    """Draw a box with text."""
    if text_color is None:
        text_color = COLORS['text']
    
    box = FancyBboxPatch(
        xy, width, height,
        boxstyle="round,pad=0.01",
        facecolor=bg_color,
        edgecolor=border_color,
        linewidth=2.5,
        transform=ax.transAxes,
        zorder=2
    )
    ax.add_patch(box)
    
    ax.text(
        xy[0] + width/2, xy[1] + height/2,
        text,
        ha='center', va='center',
        fontsize=fontsize,
        fontweight=fontweight,
        color=text_color,
        transform=ax.transAxes,
        zorder=3
    )


def draw_arrow(ax, x1, y1, x2, y2, color, label='', style='->', width=2.5):
    """Draw an arrow."""
    arrow = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle=style + ',head_width=0.4,head_length=0.3',
        color=color,
        linewidth=width,
        transform=ax.transAxes,
        zorder=1,
        mutation_scale=25
    )
    ax.add_patch(arrow)
    
    if label:
        mid_x = (x1 + x2) / 2
        mid_y = (y1 + y2) / 2
        ax.text(
            mid_x, mid_y + 0.015,
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


def create_architecture_flowchart(output_path='export/architecture_flowchart.png'):
    """Create architecture flowchart with transfer learning."""
    
    print("\n" + "="*70)
    print("CREATING ARCHITECTURE FLOWCHART")
    print("="*70 + "\n")
    
    # Create figure
    fig = plt.figure(figsize=(18, 14))
    
    # Main title
    fig.text(0.5, 0.98, 'Visual-Only Lip Reading: Architecture Flowchart with Transfer Learning',
             ha='center', va='top', fontsize=18, fontweight='bold', color=COLORS['text'])
    
    # Create main axis
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # =========================================================================
    # TRANSFER LEARNING PHASES (TOP)
    # =========================================================================
    
    phase_y = 0.88
    
    # Phase 1
    draw_box(ax, (0.05, phase_y), 0.28, 0.08,
             'PHASE 1: PRE-TRAINING\nLRS2/LRS3 (English) - 150K samples',
             COLORS['light_blue'], COLORS['phase1'], fontsize=9, fontweight='bold')
    
    # Arrow
    draw_arrow(ax, 0.33, phase_y + 0.04, 0.39, phase_y + 0.04, 
              COLORS['phase1'], '', width=3)
    
    # Middle box
    draw_box(ax, (0.39, phase_y), 0.22, 0.08,
             'TRAIN ALL LAYERS\nLearn Visual Features',
             COLORS['light_green'], COLORS['trainable'], fontsize=9, fontweight='bold')
    
    # Arrow
    draw_arrow(ax, 0.61, phase_y + 0.04, 0.67, phase_y + 0.04, 
              COLORS['phase2'], 'Transfer', width=3)
    
    # Phase 2
    draw_box(ax, (0.67, phase_y), 0.28, 0.08,
             'PHASE 2: FINE-TUNING\nBengali/Arabic - 1K-5K samples',
             COLORS['light_orange'], COLORS['phase2'], fontsize=9, fontweight='bold')
    
    # =========================================================================
    # MAIN ARCHITECTURE FLOW
    # =========================================================================
    
    # Starting y position for architecture
    arch_y_start = 0.76
    
    # INPUT DATA
    ax.text(0.5, arch_y_start, '▼ INPUT VIDEO SEQUENCE ▼',
           ha='center', va='center', fontsize=11, fontweight='bold',
           color=COLORS['data'], transform=ax.transAxes)
    
    # Raw video box
    input_y = arch_y_start - 0.08
    draw_box(ax, (0.35, input_y), 0.30, 0.06,
             'Raw Video Frames\n(T × H × W × 3)',
             COLORS['light_purple'], COLORS['data'], fontsize=9)
    
    # Arrow down
    draw_arrow(ax, 0.5, input_y, 0.5, input_y - 0.04, COLORS['arrow'])
    
    # PREPROCESSING
    preproc_y = input_y - 0.10
    draw_box(ax, (0.30, preproc_y), 0.40, 0.06,
             'PREPROCESSING\nFace Detection → Landmarks → Lip Crop (88×88) → Normalize',
             COLORS['light_purple'], COLORS['data'], fontsize=8)
    
    # Arrow down
    draw_arrow(ax, 0.5, preproc_y, 0.5, preproc_y - 0.04, COLORS['arrow'])
    
    # Preprocessed data
    prep_data_y = preproc_y - 0.08
    draw_box(ax, (0.35, prep_data_y), 0.30, 0.05,
             'Lip Sequence (T × 88 × 88 × 1)',
             COLORS['light_purple'], COLORS['data'], fontsize=9)
    
    # Arrow down to model
    draw_arrow(ax, 0.5, prep_data_y, 0.5, prep_data_y - 0.04, COLORS['arrow'])
    
    # =========================================================================
    # VISUAL ENCODER ARCHITECTURE
    # =========================================================================
    
    encoder_y_start = prep_data_y - 0.10
    
    ax.text(0.5, encoder_y_start + 0.02, '━━━━━━ VISUAL ENCODER ━━━━━━',
           ha='center', va='center', fontsize=12, fontweight='bold',
           color=COLORS['text'], transform=ax.transAxes)
    
    # Component positions
    comp_y = encoder_y_start - 0.06
    comp_height = 0.10
    comp_width = 0.16
    x_positions = [0.08, 0.28, 0.52, 0.76]
    
    components = [
        {
            'name': '3D CNN',
            'details': 'Conv3D\n(5×7×7)\n↓\nMaxPool3D\n(1×3×3)',
            'frozen': True,
            'output': '(T, 44, 44, 64)'
        },
        {
            'name': 'ResNet-18',
            'details': 'Residual\nBlocks\n↓\nGlobal\nAvg Pool',
            'frozen': True,
            'output': '(T, 256)'
        },
        {
            'name': 'Conformer Encoder',
            'details': 'Multi-Head\nAttention\n+\nConv Module\n+\nFeed Forward',
            'frozen': False,
            'output': '(T/2, 360)'
        },
        {
            'name': 'CTC Head',
            'details': 'Linear\nProjection\n↓\nSoftmax\n↓\nCTC Loss',
            'frozen': False,
            'output': '(T/2, vocab_size)'
        }
    ]
    
    for i, (x_pos, comp) in enumerate(zip(x_positions, components)):
        # Determine colors
        if comp['frozen']:
            bg_color = COLORS['light_gray']
            border_color = COLORS['frozen']
            status_icon = '❄️'
            status_text = 'FROZEN'
            status_color = COLORS['phase1']
        else:
            bg_color = COLORS['light_green']
            border_color = COLORS['trainable']
            status_icon = '🔥'
            status_text = 'FINE-TUNED'
            status_color = COLORS['phase2']
        
        # Component box
        draw_box(ax, (x_pos, comp_y), comp_width, comp_height,
                comp['name'] + '\n\n' + comp['details'],
                bg_color, border_color, fontsize=8, fontweight='bold')
        
        # Status label
        ax.text(x_pos + comp_width/2, comp_y + comp_height + 0.01,
               f'{status_icon} {status_text}',
               ha='center', va='bottom', fontsize=7, fontweight='bold',
               color=status_color, transform=ax.transAxes,
               bbox=dict(boxstyle='round,pad=0.3', 
                        facecolor='white' if comp['frozen'] else COLORS['light_orange'],
                        edgecolor=status_color, linewidth=1.5))
        
        # Output shape
        ax.text(x_pos + comp_width/2, comp_y - 0.01,
               comp['output'],
               ha='center', va='top', fontsize=7,
               color=COLORS['text'], style='italic',
               transform=ax.transAxes)
        
        # Arrow to next component
        if i < len(components) - 1:
            arrow_x1 = x_pos + comp_width + 0.01
            arrow_x2 = x_positions[i+1] - 0.01
            arrow_y = comp_y + comp_height/2
            draw_arrow(ax, arrow_x1, arrow_y, arrow_x2, arrow_y, 
                      COLORS['arrow'], '', width=2.5)
    
    # =========================================================================
    # OUTPUT PROCESSING
    # =========================================================================
    
    output_y = comp_y - 0.08
    
    # Arrow down
    draw_arrow(ax, 0.84, comp_y, 0.84, output_y + 0.06, COLORS['arrow'])
    
    # CTC Decoding
    draw_box(ax, (0.68, output_y), 0.32, 0.06,
             'CTC DECODING\nBeam Search / Greedy Decoding',
             COLORS['light_red'], COLORS['output'], fontsize=9, fontweight='bold')
    
    # Arrow down
    draw_arrow(ax, 0.84, output_y, 0.84, output_y - 0.04, COLORS['arrow'])
    
    # Final output
    final_y = output_y - 0.08
    draw_box(ax, (0.70, final_y), 0.28, 0.06,
             'PREDICTED TEXT\n(Transliterated)',
             COLORS['light_red'], COLORS['output'], fontsize=9, fontweight='bold')
    
    # =========================================================================
    # DETAILED ARCHITECTURE SPECS (LEFT SIDE)
    # =========================================================================
    
    specs_y = comp_y - 0.02
    
    ax.text(0.02, specs_y, 'ARCHITECTURE\nSPECIFICATIONS:',
           ha='left', va='top', fontsize=9, fontweight='bold',
           color=COLORS['text'], transform=ax.transAxes)
    
    specs_text = '''
3D CNN:
• Input: (B, 1, T, 88, 88)
• Conv3D: 64 filters
• Kernel: 5×7×7
• MaxPool3D: 1×3×3

ResNet-18:
• 4 residual blocks
• Batch normalization
• Output: 256-dim features

Conformer:
• 6+6 blocks (2 stages)
• Dim: 256 → 360
• Heads: 4
• InterCTC at blocks 3,6

CTC Head:
• Linear: 360 → vocab_size
• Loss: CTC Loss
• Decoding: Beam search
'''
    
    ax.text(0.02, specs_y - 0.02, specs_text,
           ha='left', va='top', fontsize=7,
           color=COLORS['text'], transform=ax.transAxes,
           linespacing=1.4,
           family='monospace')
    
    # =========================================================================
    # TRAINING STRATEGY (RIGHT SIDE)
    # =========================================================================
    
    strategy_y = comp_y - 0.02
    
    ax.text(0.98, strategy_y, 'TRAINING\nSTRATEGY:',
           ha='right', va='top', fontsize=9, fontweight='bold',
           color=COLORS['text'], transform=ax.transAxes)
    
    strategy_text = '''
PHASE 1 (Pre-training):
• Dataset: LRS2 + LRS3
• Samples: ~150K
• Language: English
• Strategy: Train all layers
• Epochs: 100
• Optimizer: AdamW
• LR: 5e-4

PHASE 2 (Fine-tuning):
• Dataset: LipBengal / LRW-AR
• Samples: 1K-5K
• Language: Bengali / Arabic
• Strategy:
  ❄️ Freeze: 3D CNN + ResNet
  🔥 Train: Conformer + CTC Head
• Epochs: 50
• Optimizer: AdamW
• LR: 1e-4 (lower)
'''
    
    ax.text(0.98, strategy_y - 0.02, strategy_text,
           ha='right', va='top', fontsize=7,
           color=COLORS['text'], transform=ax.transAxes,
           linespacing=1.4,
           family='monospace')
    
    # =========================================================================
    # LEGEND
    # =========================================================================
    
    legend_y = 0.04
    
    legend_items = [
        ('Pre-training Phase', COLORS['light_blue'], COLORS['phase1']),
        ('Fine-tuning Phase', COLORS['light_orange'], COLORS['phase2']),
        ('Frozen Layers (❄️)', COLORS['light_gray'], COLORS['frozen']),
        ('Trainable Layers (🔥)', COLORS['light_green'], COLORS['trainable']),
        ('Data/Input', COLORS['light_purple'], COLORS['data']),
        ('Output', COLORS['light_red'], COLORS['output'])
    ]
    
    legend_x_start = 0.15
    legend_spacing = 0.12
    
    ax.text(0.5, legend_y + 0.03, 'LEGEND',
           ha='center', va='center', fontsize=9, fontweight='bold',
           color=COLORS['text'], transform=ax.transAxes)
    
    for i, (label, bg, border) in enumerate(legend_items):
        x = legend_x_start + i * legend_spacing
        
        rect = Rectangle((x, legend_y), 0.02, 0.02,
                        facecolor=bg, edgecolor=border,
                        linewidth=2, transform=ax.transAxes)
        ax.add_patch(rect)
        
        ax.text(x + 0.025, legend_y + 0.01, label,
               ha='left', va='center', fontsize=7,
               color=COLORS['text'], transform=ax.transAxes)
    
    # Save
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Architecture flowchart saved to: {output_file}")
    
    # PDF version
    pdf_file = output_file.with_suffix('.pdf')
    plt.savefig(pdf_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ PDF version saved to: {pdf_file}")
    
    print("\n" + "="*70)
    print("✅ ARCHITECTURE FLOWCHART COMPLETE")
    print("="*70 + "\n")
    
    plt.close()
    
    return output_file


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Create architecture flowchart')
    parser.add_argument('--output', type=str,
                       default='export/architecture_flowchart.png')
    
    args = parser.parse_args()
    
    try:
        create_architecture_flowchart(args.output)
        print("Architecture flowchart created successfully!")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)





