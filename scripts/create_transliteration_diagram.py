#!/usr/bin/env python3
"""
Create a diagram showing the transliteration process from Bengali to English.
Shows character-by-character mapping.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.gridspec import GridSpec
import torch
from PIL import Image
import json

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.banglish_lookup import to_banglish


# Color scheme
COLORS = {
    'bengali': '#E74C3C',      # Red for Bengali
    'transliteration': '#3498DB', # Blue for transliteration
    'mapping': '#27AE60',      # Green for mapping
    'arrow': '#555555',
    'text': '#2C3E50',
    'light_red': '#FADBD8',
    'light_blue': '#EBF5FB',
    'light_green': '#E8F8F5'
}


def load_sample(dataset_root, num_frames=5):
    """Load a sample with video frames and lip crops."""
    root = Path(dataset_root)
    
    for speaker_dir in sorted(root.iterdir()):
        if not speaker_dir.is_dir() or speaker_dir.name.startswith('.'):
            continue
        
        speaker = speaker_dir.name
        
        for word_dir in sorted(speaker_dir.iterdir()):
            if not word_dir.is_dir():
                continue
            
            bengali_word = word_dir.name
            frame_files = sorted(list(word_dir.glob("*.jpg")))
            
            if len(frame_files) < num_frames:
                continue
            
            # Load video frames
            video_indices = np.linspace(0, len(frame_files)-1, num_frames).astype(int)
            video_frames = []
            for idx in video_indices:
                img = Image.open(frame_files[idx])
                video_frames.append(np.array(img))
            
            # Try to find lip crops
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
                            
                            lip_indices = np.linspace(0, lip_frames.shape[0]-1, num_frames).astype(int)
                            lip_frames = [lip_frames[idx] for idx in lip_indices]
                            
                            return video_frames, lip_frames, bengali_word, speaker
                        except:
                            continue
    
    return None, None, None, None


def create_transliteration_diagram(dataset_root='/home/thesis/Thesis/AVEC/datasets/LipBengal',
                                   output_path='export/transliteration_process.png'):
    """Create transliteration process diagram."""
    
    print("\n" + "="*70)
    print("CREATING TRANSLITERATION PROCESS DIAGRAM")
    print("="*70 + "\n")
    
    # Load sample
    print("Loading sample...")
    video_frames, lip_frames, bengali_word, speaker = load_sample(dataset_root)
    
    if video_frames is None:
        print("✗ No suitable sample found!")
        return None
    
    # Get transliteration
    transliteration = to_banglish(bengali_word)
    
    print(f"✓ Sample found:")
    print(f"  Speaker: {speaker}")
    print(f"  Bengali Word: {bengali_word}")
    print(f"  Transliteration: {transliteration}")
    print()
    
    # Split into characters
    bengali_chars = list(bengali_word)
    english_chars = list(transliteration.lower())
    
    # Create figure
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(5, 1, figure=fig, hspace=0.4,
                  left=0.05, right=0.95, top=0.94, bottom=0.05)
    
    # Title
    fig.suptitle('Transliteration Process: Bengali → English (Latin Script)',
                 fontsize=18, fontweight='bold', color=COLORS['text'], y=0.97)
    
    # =========================================================================
    # SECTION 1: VISUAL INPUT
    # =========================================================================
    
    ax_input = fig.add_subplot(gs[0])
    ax_input.set_xlim(0, 1)
    ax_input.set_ylim(0, 1)
    ax_input.axis('off')
    
    ax_input.text(0.5, 0.95, '① VISUAL INPUT: Lip Reading Frames',
                  ha='center', va='top', fontsize=14, fontweight='bold',
                  color=COLORS['text'], transform=ax_input.transAxes)
    
    # Display lip crops
    num_display = min(5, len(lip_frames))
    lip_size = 0.12
    lip_spacing = 0.15
    start_x = 0.5 - (num_display * lip_spacing - (lip_spacing - lip_size)) / 2
    
    for i in range(num_display):
        x_pos = start_x + i * lip_spacing
        extent = [x_pos, x_pos + lip_size, 0.15, 0.75]
        ax_input.imshow(lip_frames[i], extent=extent, aspect='auto',
                       cmap='gray', transform=ax_input.transAxes)
        
        rect = Rectangle((x_pos, 0.15), lip_size, 0.6,
                        linewidth=2.5, edgecolor=COLORS['mapping'],
                        facecolor='none', transform=ax_input.transAxes)
        ax_input.add_patch(rect)
        
        ax_input.text(x_pos + lip_size/2, 0.08, f't={i+1}',
                     ha='center', va='center', fontsize=9,
                     color=COLORS['text'], transform=ax_input.transAxes)
    
    # =========================================================================
    # SECTION 2: MODEL OUTPUT (BENGALI)
    # =========================================================================
    
    ax_bengali = fig.add_subplot(gs[1])
    ax_bengali.set_xlim(0, 1)
    ax_bengali.set_ylim(0, 1)
    ax_bengali.axis('off')
    
    ax_bengali.text(0.5, 0.95, '② MODEL OUTPUT: Bengali Word',
                    ha='center', va='top', fontsize=14, fontweight='bold',
                    color=COLORS['text'], transform=ax_bengali.transAxes)
    
    # Bengali word box
    box = FancyBboxPatch((0.25, 0.35), 0.5, 0.3,
                         boxstyle="round,pad=0.02",
                         facecolor=COLORS['light_red'],
                         edgecolor=COLORS['bengali'],
                         linewidth=4,
                         transform=ax_bengali.transAxes)
    ax_bengali.add_patch(box)
    
    ax_bengali.text(0.5, 0.5, bengali_word,
                    ha='center', va='center', fontsize=48, fontweight='bold',
                    color=COLORS['bengali'], transform=ax_bengali.transAxes)
    
    ax_bengali.text(0.5, 0.2, 'Bengali Script (UTF-8)',
                    ha='center', va='center', fontsize=10, style='italic',
                    color=COLORS['text'], transform=ax_bengali.transAxes)
    
    # =========================================================================
    # SECTION 3: TRANSLITERATION MAPPING
    # =========================================================================
    
    ax_mapping = fig.add_subplot(gs[2])
    ax_mapping.set_xlim(0, 1)
    ax_mapping.set_ylim(0, 1)
    ax_mapping.axis('off')
    
    ax_mapping.text(0.5, 0.95, '③ TRANSLITERATION MAPPING: Character-by-Character',
                    ha='center', va='top', fontsize=14, fontweight='bold',
                    color=COLORS['text'], transform=ax_mapping.transAxes)
    
    # Character mapping
    num_chars = len(bengali_chars)
    char_width = min(0.08, 0.7 / num_chars)
    char_spacing = min(0.10, 0.8 / num_chars)
    start_x_chars = 0.5 - (num_chars * char_spacing - (char_spacing - char_width)) / 2
    
    bengali_y = 0.65
    english_y = 0.25
    
    for i, (beng_char, eng_char) in enumerate(zip(bengali_chars, english_chars)):
        x_pos = start_x_chars + i * char_spacing
        
        # Bengali character box
        bengali_box = FancyBboxPatch((x_pos, bengali_y), char_width, 0.15,
                                     boxstyle="round,pad=0.01",
                                     facecolor=COLORS['light_red'],
                                     edgecolor=COLORS['bengali'],
                                     linewidth=2.5,
                                     transform=ax_mapping.transAxes)
        ax_mapping.add_patch(bengali_box)
        
        ax_mapping.text(x_pos + char_width/2, bengali_y + 0.075,
                       beng_char, ha='center', va='center',
                       fontsize=20, fontweight='bold',
                       color=COLORS['bengali'],
                       transform=ax_mapping.transAxes)
        
        # Arrow
        arrow = FancyArrowPatch(
            (x_pos + char_width/2, bengali_y),
            (x_pos + char_width/2, english_y + 0.15),
            arrowstyle='->,head_width=0.3,head_length=0.2',
            color=COLORS['mapping'],
            linewidth=2.5,
            transform=ax_mapping.transAxes,
            mutation_scale=20
        )
        ax_mapping.add_patch(arrow)
        
        # English character box
        english_box = FancyBboxPatch((x_pos, english_y), char_width, 0.15,
                                     boxstyle="round,pad=0.01",
                                     facecolor=COLORS['light_blue'],
                                     edgecolor=COLORS['transliteration'],
                                     linewidth=2.5,
                                     transform=ax_mapping.transAxes)
        ax_mapping.add_patch(english_box)
        
        # Handle multiple English chars for one Bengali char
        display_eng = eng_char if i < len(english_chars) else ''
        ax_mapping.text(x_pos + char_width/2, english_y + 0.075,
                       display_eng, ha='center', va='center',
                       fontsize=16, fontweight='bold',
                       color=COLORS['transliteration'],
                       transform=ax_mapping.transAxes)
    
    # Labels
    ax_mapping.text(start_x_chars - 0.05, bengali_y + 0.075, 'Bengali:',
                    ha='right', va='center', fontsize=10, fontweight='bold',
                    color=COLORS['text'], transform=ax_mapping.transAxes)
    
    ax_mapping.text(start_x_chars - 0.05, english_y + 0.075, 'Latin:',
                    ha='right', va='center', fontsize=10, fontweight='bold',
                    color=COLORS['text'], transform=ax_mapping.transAxes)
    
    # =========================================================================
    # SECTION 4: FINAL OUTPUT
    # =========================================================================
    
    ax_output = fig.add_subplot(gs[3])
    ax_output.set_xlim(0, 1)
    ax_output.set_ylim(0, 1)
    ax_output.axis('off')
    
    ax_output.text(0.5, 0.95, '④ FINAL OUTPUT: Transliterated Text',
                   ha='center', va='top', fontsize=14, fontweight='bold',
                   color=COLORS['text'], transform=ax_output.transAxes)
    
    # Transliteration box
    box = FancyBboxPatch((0.25, 0.35), 0.5, 0.3,
                         boxstyle="round,pad=0.02",
                         facecolor=COLORS['light_blue'],
                         edgecolor=COLORS['transliteration'],
                         linewidth=4,
                         transform=ax_output.transAxes)
    ax_output.add_patch(box)
    
    ax_output.text(0.5, 0.5, transliteration,
                   ha='center', va='center', fontsize=44, fontweight='bold',
                   color=COLORS['transliteration'], transform=ax_output.transAxes)
    
    ax_output.text(0.5, 0.2, 'Latin Script (English Alphabet)',
                   ha='center', va='center', fontsize=10, style='italic',
                   color=COLORS['text'], transform=ax_output.transAxes)
    
    # =========================================================================
    # SECTION 5: EXPLANATION
    # =========================================================================
    
    ax_explain = fig.add_subplot(gs[4])
    ax_explain.set_xlim(0, 1)
    ax_explain.set_ylim(0, 1)
    ax_explain.axis('off')
    
    explanation_text = (
        f"TRANSLITERATION PROCESS:\n\n"
        f"Bengali Word: {bengali_word} → Transliteration: {transliteration}\n\n"
        f"• Bengali uses a complex script with {len(bengali_chars)} characters\n"
        f"• Transliteration converts to {len(english_chars)} Latin/English characters\n"
        f"• Enables cross-language representation and pronunciation\n"
        f"• Essential for low-resource language processing and transfer learning"
    )
    
    box = FancyBboxPatch((0.1, 0.15), 0.8, 0.7,
                         boxstyle="round,pad=0.02",
                         facecolor=COLORS['light_green'],
                         edgecolor=COLORS['mapping'],
                         linewidth=2.5,
                         transform=ax_explain.transAxes)
    ax_explain.add_patch(box)
    
    ax_explain.text(0.5, 0.5, explanation_text,
                    ha='center', va='center', fontsize=11,
                    color=COLORS['text'], transform=ax_explain.transAxes,
                    linespacing=1.8)
    
    # Save
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ Transliteration diagram saved to: {output_file}")
    
    # PDF version
    pdf_file = output_file.with_suffix('.pdf')
    plt.savefig(pdf_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✓ PDF version saved to: {pdf_file}")
    
    # Save metadata
    metadata = {
        'bengali_word': bengali_word,
        'transliteration': transliteration,
        'speaker': speaker,
        'num_bengali_chars': len(bengali_chars),
        'num_english_chars': len(english_chars),
        'bengali_characters': bengali_chars,
        'english_characters': english_chars
    }
    
    metadata_file = output_file.parent / 'transliteration_metadata.json'
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Metadata saved to: {metadata_file.name}")
    
    print("\n" + "="*70)
    print("✅ TRANSLITERATION DIAGRAM COMPLETE")
    print("="*70 + "\n")
    
    print(f"Word: {bengali_word} → {transliteration}")
    print(f"Characters: {len(bengali_chars)} Bengali → {len(english_chars)} Latin\n")
    
    plt.close()
    
    return output_file


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Create transliteration diagram')
    parser.add_argument('--dataset', type=str,
                       default='/home/thesis/Thesis/AVEC/datasets/LipBengal')
    parser.add_argument('--output', type=str,
                       default='export/transliteration_process.png')
    
    args = parser.parse_args()
    
    try:
        create_transliteration_diagram(args.dataset, args.output)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)





