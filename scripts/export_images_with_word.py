#!/usr/bin/env python3
"""
Export images with the Bengali word displayed.
Creates a nice visualization showing the word along with the frames.
"""

import sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import json

sys.path.insert(0, str(Path(__file__).parent.parent))


def create_composite_with_word(video_frames, lip_frames, bengali_word, 
                                transliteration, output_path):
    """Create a composite image showing frames with the word."""
    
    num_frames = len(video_frames)
    
    # Create figure
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(4, num_frames, figure=fig, hspace=0.3, wspace=0.2,
                  left=0.05, right=0.95, top=0.92, bottom=0.05)
    
    # Title with Bengali word
    fig.suptitle(f'Bengali Word: {bengali_word}  |  Transliteration: {transliteration}',
                 fontsize=20, fontweight='bold', y=0.97)
    
    # Row 1: Video frames
    for i, frame in enumerate(video_frames):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(frame)
        ax.set_title(f'Frame {i+1}', fontsize=10, fontweight='bold')
        ax.axis('off')
    
    # Add label
    ax_label1 = fig.add_subplot(gs[1, :])
    ax_label1.text(0.5, 0.5, '↓ Preprocessing: Face Detection → Lip Extraction ↓',
                   ha='center', va='center', fontsize=12, fontweight='bold',
                   transform=ax_label1.transAxes)
    ax_label1.axis('off')
    
    # Row 2: Lip crops
    for i, lip_frame in enumerate(lip_frames):
        ax = fig.add_subplot(gs[2, i])
        ax.imshow(lip_frame, cmap='gray')
        ax.set_title(f'Lip {i+1}', fontsize=10, fontweight='bold')
        ax.axis('off')
    
    # Row 3: Word display
    ax_word = fig.add_subplot(gs[3, :])
    ax_word.text(0.5, 0.7, f'{bengali_word}',
                ha='center', va='center', fontsize=48, fontweight='bold',
                color='#E74C3C', transform=ax_word.transAxes)
    
    ax_word.text(0.5, 0.3, f'({transliteration})',
                ha='center', va='center', fontsize=24,
                color='#555555', style='italic', transform=ax_word.transAxes)
    ax_word.axis('off')
    
    # Save
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return output_path


def load_and_export_with_word(dataset_root='/home/thesis/Thesis/AVEC/datasets/LipBengal',
                              output_dir='export/diagram_images_with_word',
                              num_frames=5,
                              specific_word=None,
                              specific_speaker=None):
    """Load and export images with word display."""
    
    print("\n" + "="*70)
    print("EXPORTING IMAGES WITH WORD DISPLAY")
    print("="*70 + "\n")
    
    root = Path(dataset_root)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find sample
    sample_found = False
    video_frames = None
    lip_frames = None
    bengali_word = None
    speaker = None
    
    speakers_to_check = [specific_speaker] if specific_speaker else sorted(root.iterdir())
    
    for speaker_dir in speakers_to_check:
        if isinstance(speaker_dir, str):
            speaker_dir = root / speaker_dir
        
        if not speaker_dir.is_dir() or speaker_dir.name.startswith('.'):
            continue
        
        speaker = speaker_dir.name
        
        words_to_check = [specific_word] if specific_word else sorted(speaker_dir.iterdir())
        
        for word_item in words_to_check:
            if isinstance(word_item, str):
                word_dir = speaker_dir / word_item
            else:
                word_dir = word_item
            
            if not word_dir.is_dir():
                continue
            
            bengali_word = word_dir.name
            
            # Check if specific word matches
            if specific_word and bengali_word != specific_word:
                continue
            
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
                            
                            # Sample lip frames
                            lip_indices = np.linspace(0, lip_frames.shape[0]-1, num_frames).astype(int)
                            lip_frames = [lip_frames[idx] for idx in lip_indices]
                            
                            sample_found = True
                            break
                        except Exception as e:
                            continue
                
                if sample_found:
                    break
            
            if sample_found:
                break
        
        if sample_found:
            break
    
    if not sample_found:
        print("✗ No suitable samples found!")
        return None
    
    # Get transliteration
    try:
        from scripts.banglish_lookup import to_banglish
        transliteration = to_banglish(bengali_word)
    except:
        transliteration = "transliteration-unavailable"
    
    print(f"✓ Found sample:")
    print(f"  Speaker: {speaker}")
    print(f"  Bengali Word: {bengali_word}")
    print(f"  Transliteration: {transliteration}")
    print(f"  Frames: {num_frames}")
    print()
    
    # Save individual frames
    print("Saving individual frames...")
    video_dir = output_path / 'video_frames'
    lip_dir = output_path / 'lip_crops'
    video_dir.mkdir(exist_ok=True)
    lip_dir.mkdir(exist_ok=True)
    
    for i, frame in enumerate(video_frames):
        img = Image.fromarray(frame)
        img.save(video_dir / f'frame_{i+1:02d}.jpg', quality=95)
        print(f"  ✓ video_frames/frame_{i+1:02d}.jpg")
    
    print()
    for i, lip in enumerate(lip_frames):
        img = Image.fromarray(lip)
        img.save(lip_dir / f'lip_{i+1:02d}.png')
        print(f"  ✓ lip_crops/lip_{i+1:02d}.png")
    
    # Create composite image
    print("\nCreating composite image with word...")
    composite_file = output_path / f'composite_{bengali_word}.png'
    create_composite_with_word(video_frames, lip_frames, bengali_word,
                              transliteration, composite_file)
    print(f"  ✓ {composite_file.name}")
    
    # Save metadata
    metadata = {
        'speaker': speaker,
        'word_bengali': bengali_word,
        'word_transliteration': transliteration,
        'num_frames': num_frames,
        'video_frames': [f'frame_{i+1:02d}.jpg' for i in range(num_frames)],
        'lip_crops': [f'lip_{i+1:02d}.png' for i in range(num_frames)],
        'composite_image': f'composite_{bengali_word}.png'
    }
    
    metadata_file = output_path / 'metadata.json'
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"  ✓ metadata.json")
    
    # Create README
    readme_file = output_path / 'README.txt'
    with open(readme_file, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("IMAGES WITH WORD DISPLAY\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"Bengali Word: {bengali_word}\n")
        f.write(f"Transliteration: {transliteration}\n")
        f.write(f"Speaker: {speaker}\n")
        f.write(f"Frames: {num_frames}\n\n")
        
        f.write("FILES:\n")
        f.write("-" * 70 + "\n")
        f.write("video_frames/       - Original video frames\n")
        f.write("lip_crops/          - Preprocessed lip crops (88x88)\n")
        f.write(f"composite_{bengali_word}.png - Complete visualization\n")
        f.write("metadata.json       - Sample information\n")
        f.write("README.txt          - This file\n\n")
        
        f.write("COMPOSITE IMAGE:\n")
        f.write("-" * 70 + "\n")
        f.write("The composite image shows:\n")
        f.write("  • Row 1: Original video frames\n")
        f.write("  • Row 2: Processing indicator\n")
        f.write("  • Row 3: Preprocessed lip crops\n")
        f.write("  • Row 4: Bengali word with transliteration\n\n")
        
        f.write("="*70 + "\n")
    
    print(f"  ✓ README.txt")
    
    print("\n" + "="*70)
    print(f"✅ EXPORT COMPLETE: {output_path}")
    print("="*70 + "\n")
    
    print(f"Bengali Word: {bengali_word} ({transliteration})")
    print(f"Composite image: composite_{bengali_word}.png")
    print()
    
    return output_path


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Export images with word display')
    parser.add_argument('--dataset', type=str,
                       default='/home/thesis/Thesis/AVEC/datasets/LipBengal')
    parser.add_argument('--output', type=str,
                       default='export/images_with_word')
    parser.add_argument('--frames', type=int, default=5)
    parser.add_argument('--word', type=str, default=None,
                       help='Specific Bengali word to export')
    parser.add_argument('--speaker', type=str, default=None,
                       help='Specific speaker (e.g., s1, s100)')
    
    args = parser.parse_args()
    
    try:
        load_and_export_with_word(args.dataset, args.output, args.frames,
                                  args.word, args.speaker)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)





