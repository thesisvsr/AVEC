#!/usr/bin/env python3
"""
Create a unified flowchart with transfer learning integrated.
Single vertical flow showing everything.
"""

import sys
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Polygon
from matplotlib.gridspec import GridSpec

# Colors
COLORS = {
    'pretrain': '#3498DB',
    'finetune': '#E67E22',
    'frozen': '#95A5A6',
    'active': '#27AE60',
    'data': '#9B59B6',
    'output': '#E74C3C',
    'arrow': '#2C3E50',
    'text': '#2C3E50'
}


def draw_box(ax, xy, w, h, text, bg, border, fs=9, fw='normal'):
    """Draw a box."""
    box = FancyBboxPatch(xy, w, h, boxstyle="round,pad=0.012",
                         facecolor=bg, edgecolor=border, linewidth=3,
                         transform=ax.transAxes, zorder=2)
    ax.add_patch(box)
    ax.text(xy[0]+w/2, xy[1]+h/2, text, ha='center', va='center',
           fontsize=fs, fontweight=fw, color=COLORS['text'],
           transform=ax.transAxes, zorder=3, linespacing=1.3)


def draw_arrow(ax, x1, y1, x2, y2, color, label='', lw=3.5):
    """Draw arrow."""
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                           arrowstyle='->,head_width=0.4,head_length=0.3',
                           color=color, linewidth=lw,
                           transform=ax.transAxes, zorder=1, mutation_scale=25)
    ax.add_patch(arrow)
    if label:
        ax.text((x1+x2)/2+0.05, (y1+y2)/2, label, ha='left', va='center',
               fontsize=8, fontweight='bold', color=color,
               transform=ax.transAxes, zorder=3,
               bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                        edgecolor=color, linewidth=2))


def create_unified_flowchart(output_path='export/unified_flowchart.png'):
    """Create unified flowchart."""
    
    print("\n" + "="*70)
    print("CREATING UNIFIED FLOWCHART")
    print("="*70 + "\n")
    
    fig = plt.figure(figsize=(14, 16))
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    # Title
    fig.text(0.5, 0.985, 'Visual-Only Lip Reading with Transfer Learning',
             ha='center', va='top', fontsize=18, fontweight='bold', color=COLORS['text'])
    fig.text(0.5, 0.97, 'Unified Architecture Flowchart',
             ha='center', va='top', fontsize=14, color=COLORS['text'])
    
    y = 0.92
    
    # =====================================================================
    # TRANSFER LEARNING PHASES
    # =====================================================================
    
    # Phase indicator background
    phase1_box = Rectangle((0.05, y-0.01), 0.40, 0.09,
                           facecolor='#EBF5FB', edgecolor=COLORS['pretrain'],
                           linewidth=2, transform=ax.transAxes, zorder=0, alpha=0.3)
    ax.add_patch(phase1_box)
    
    phase2_box = Rectangle((0.55, y-0.01), 0.40, 0.09,
                           facecolor='#FEF5E7', edgecolor=COLORS['finetune'],
                           linewidth=2, transform=ax.transAxes, zorder=0, alpha=0.3)
    ax.add_patch(phase2_box)
    
    # Phase 1
    ax.text(0.25, y+0.06, 'PHASE 1: PRE-TRAINING', ha='center', va='center',
           fontsize=11, fontweight='bold', color=COLORS['pretrain'],
           transform=ax.transAxes)
    ax.text(0.25, y+0.03, 'LRS2/LRS3 (English)', ha='center', va='center',
           fontsize=9, color=COLORS['text'], transform=ax.transAxes)
    ax.text(0.25, y, '150,000 samples', ha='center', va='center',
           fontsize=9, fontweight='bold', color=COLORS['text'],
           transform=ax.transAxes)
    
    # Arrow
    arrow_y = y+0.03
    draw_arrow(ax, 0.45, arrow_y, 0.55, arrow_y, COLORS['arrow'], 'Transfer\nWeights')
    
    # Phase 2
    ax.text(0.75, y+0.06, 'PHASE 2: FINE-TUNING', ha='center', va='center',
           fontsize=11, fontweight='bold', color=COLORS['finetune'],
           transform=ax.transAxes)
    ax.text(0.75, y+0.03, 'Bengali / Arabic', ha='center', va='center',
           fontsize=9, color=COLORS['text'], transform=ax.transAxes)
    ax.text(0.75, y, '1,000-5,000 samples', ha='center', va='center',
           fontsize=9, fontweight='bold', color=COLORS['text'],
           transform=ax.transAxes)
    
    y -= 0.14
    
    # =====================================================================
    # MAIN PIPELINE
    # =====================================================================
    
    # INPUT
    ax.text(0.5, y+0.02, '▼ DATA PIPELINE ▼', ha='center', va='center',
           fontsize=12, fontweight='bold', color=COLORS['text'],
           transform=ax.transAxes)
    
    y -= 0.05
    
    draw_box(ax, (0.20, y), 0.60, 0.05,
             'INPUT: Raw Video Frames (T × H × W × 3)',
             '#F4ECF7', COLORS['data'], 10, 'bold')
    
    draw_arrow(ax, 0.5, y, 0.5, y-0.04, COLORS['arrow'])
    
    y -= 0.08
    
    # PREPROCESSING
    draw_box(ax, (0.15, y), 0.70, 0.06,
             'PREPROCESSING\nFace Detection → Facial Landmarks → Lip Region Extraction → Normalization',
             '#F4ECF7', COLORS['data'], 9)
    
    draw_arrow(ax, 0.5, y, 0.5, y-0.04, COLORS['arrow'])
    
    y -= 0.08
    
    draw_box(ax, (0.25, y), 0.50, 0.04,
             'Preprocessed Lip Sequence (T × 88 × 88 × 1)',
             '#F4ECF7', COLORS['data'], 9)
    
    draw_arrow(ax, 0.5, y, 0.5, y-0.04, COLORS['arrow'])
    
    y -= 0.07
    
    # =====================================================================
    # VISUAL ENCODER TITLE
    # =====================================================================
    
    ax.text(0.5, y, '━━━━━━━━━━ VISUAL ENCODER NETWORK ━━━━━━━━━━',
           ha='center', va='center', fontsize=12, fontweight='bold',
           color=COLORS['text'], transform=ax.transAxes)
    
    y -= 0.05
    
    # =====================================================================
    # 3D CNN (FROZEN)
    # =====================================================================
    
    draw_box(ax, (0.25, y), 0.50, 0.09,
             '3D CONVOLUTIONAL NEURAL NETWORK\n\nConv3D (5×7×7, 64 filters) + Batch Norm + ReLU\nMaxPool3D (1×3×3)\n\nOutput: (T, 44, 44, 64)',
             '#ECF0F1', COLORS['frozen'], 8)
    
    # Frozen indicator
    ax.text(0.12, y+0.045, '❄\nFROZEN', ha='center', va='center',
           fontsize=9, fontweight='bold', color=COLORS['pretrain'],
           transform=ax.transAxes,
           bbox=dict(boxstyle='round,pad=0.5', facecolor='#EBF5FB',
                    edgecolor=COLORS['pretrain'], linewidth=2.5))
    
    draw_arrow(ax, 0.5, y, 0.5, y-0.04, COLORS['arrow'])
    
    y -= 0.12
    
    # =====================================================================
    # RESNET-18 (FROZEN)
    # =====================================================================
    
    draw_box(ax, (0.25, y), 0.50, 0.09,
             'RESNET-18 (2D CNN)\n\n4 Residual Blocks + Batch Normalization\nGlobal Average Pooling\n\nOutput: (T, 256)',
             '#ECF0F1', COLORS['frozen'], 8)
    
    # Frozen indicator
    ax.text(0.12, y+0.045, '❄\nFROZEN', ha='center', va='center',
           fontsize=9, fontweight='bold', color=COLORS['pretrain'],
           transform=ax.transAxes,
           bbox=dict(boxstyle='round,pad=0.5', facecolor='#EBF5FB',
                    edgecolor=COLORS['pretrain'], linewidth=2.5))
    
    draw_arrow(ax, 0.5, y, 0.5, y-0.04, COLORS['arrow'])
    
    y -= 0.12
    
    # =====================================================================
    # CONFORMER ENCODER (FINE-TUNED)
    # =====================================================================
    
    draw_box(ax, (0.25, y), 0.50, 0.10,
             'CONFORMER ENCODER\n\nMulti-Head Attention (4 heads) + Relative Positional Encoding\nConvolution Module (kernel=15) + Depthwise Conv\nFeed-Forward Network (×4 expansion)\n6+6 Blocks (2 stages) | InterCTC at blocks 3, 6\n\nOutput: (T/2, 360)',
             '#E8F8F5', COLORS['active'], 8)
    
    # Fine-tuned indicator
    ax.text(0.88, y+0.05, 'FINE-TUNED', ha='center', va='center',
           fontsize=9, fontweight='bold', color=COLORS['finetune'],
           transform=ax.transAxes,
           bbox=dict(boxstyle='round,pad=0.5', facecolor='#FEF5E7',
                    edgecolor=COLORS['finetune'], linewidth=2.5))
    
    draw_arrow(ax, 0.5, y, 0.5, y-0.04, COLORS['arrow'])
    
    y -= 0.13
    
    # =====================================================================
    # CTC HEAD (FINE-TUNED)
    # =====================================================================
    
    draw_box(ax, (0.25, y), 0.50, 0.08,
             'CTC HEAD\n\nLinear Projection (360 → vocab_size)\nSoftmax Activation\nCTC Loss Function\n\nOutput: (T/2, vocab_size)',
             '#E8F8F5', COLORS['active'], 8)
    
    # Fine-tuned indicator
    ax.text(0.88, y+0.04, 'FINE-TUNED', ha='center', va='center',
           fontsize=9, fontweight='bold', color=COLORS['finetune'],
           transform=ax.transAxes,
           bbox=dict(boxstyle='round,pad=0.5', facecolor='#FEF5E7',
                    edgecolor=COLORS['finetune'], linewidth=2.5))
    
    draw_arrow(ax, 0.5, y, 0.5, y-0.04, COLORS['arrow'])
    
    y -= 0.10
    
    # =====================================================================
    # DECODING
    # =====================================================================
    
    draw_box(ax, (0.20, y), 0.60, 0.05,
             'CTC DECODING (Beam Search / Greedy Decoding)',
             '#FADBD8', COLORS['output'], 9, 'bold')
    
    draw_arrow(ax, 0.5, y, 0.5, y-0.04, COLORS['arrow'])
    
    y -= 0.08
    
    # =====================================================================
    # OUTPUT
    # =====================================================================
    
    draw_box(ax, (0.22, y), 0.56, 0.06,
             'FINAL OUTPUT\nTransliterated Text Prediction',
             '#FADBD8', COLORS['output'], 10, 'bold')
    
    # =====================================================================
    # KEY INFORMATION
    # =====================================================================
    
    y -= 0.10
    
    # Training strategy box
    info_box = Rectangle((0.08, y-0.06), 0.84, 0.08,
                         facecolor='#F8F9F9', edgecolor=COLORS['text'],
                         linewidth=2, transform=ax.transAxes, zorder=0)
    ax.add_patch(info_box)
    
    ax.text(0.5, y-0.01, 'TRANSFER LEARNING STRATEGY', ha='center', va='center',
           fontsize=10, fontweight='bold', color=COLORS['text'],
           transform=ax.transAxes)
    
    strategy_text = (
        'Phase 1: Train entire network (3D CNN + ResNet + Conformer + CTC) on 150K English samples\n'
        'Phase 2: Freeze feature extractors (3D CNN + ResNet), Fine-tune high-level layers (Conformer + CTC) on 1-5K target language samples\n'
        'Benefit: Leverage large-scale English data to overcome low-resource constraints in Bengali/Arabic'
    )
    
    ax.text(0.5, y-0.045, strategy_text, ha='center', va='center',
           fontsize=7.5, color=COLORS['text'], transform=ax.transAxes,
           linespacing=1.6)
    
    # =====================================================================
    # LEGEND
    # =====================================================================
    
    legend_y = 0.02
    
    ax.text(0.5, legend_y+0.025, 'LEGEND', ha='center', va='center',
           fontsize=9, fontweight='bold', color=COLORS['text'],
           transform=ax.transAxes)
    
    legend_items = [
        ('Pre-training Phase', '#EBF5FB', COLORS['pretrain']),
        ('Fine-tuning Phase', '#FEF5E7', COLORS['finetune']),
        ('Frozen Layers', '#ECF0F1', COLORS['frozen']),
        ('Trainable Layers', '#E8F8F5', COLORS['active']),
        ('Data/Processing', '#F4ECF7', COLORS['data']),
        ('Output', '#FADBD8', COLORS['output'])
    ]
    
    x_start = 0.15
    x_spacing = 0.12
    
    for i, (label, bg, border) in enumerate(legend_items):
        x = x_start + i * x_spacing
        
        rect = Rectangle((x, legend_y), 0.018, 0.018,
                        facecolor=bg, edgecolor=border,
                        linewidth=2, transform=ax.transAxes)
        ax.add_patch(rect)
        
        ax.text(x+0.022, legend_y+0.009, label, ha='left', va='center',
               fontsize=6.5, color=COLORS['text'], transform=ax.transAxes)
    
    # Save
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Unified flowchart saved to: {output_file}")
    
    pdf_file = output_file.with_suffix('.pdf')
    plt.savefig(pdf_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ PDF version saved to: {pdf_file}")
    
    print("\n" + "="*70)
    print("✅ UNIFIED FLOWCHART COMPLETE")
    print("="*70 + "\n")
    
    plt.close()
    return output_file


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Create unified flowchart')
    parser.add_argument('--output', type=str,
                       default='export/unified_flowchart.png')
    
    args = parser.parse_args()
    
    try:
        create_unified_flowchart(args.output)
        print("Unified flowchart created successfully!")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)





