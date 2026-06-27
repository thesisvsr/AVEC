#!/usr/bin/env python3
"""
Extract Bengali lip reading data for manual visualization creation.
Saves: video frames, lip crops, Bengali word, transliteration, character mappings
"""

import sys
from pathlib import Path
import torch
import numpy as np
from PIL import Image
import json
import shutil
import random

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.banglish_lookup import to_banglish


def load_lipbengal_sample_correct(dataset_root, speaker='s100', target_word=None):
    """Load a complete sample."""
    root = Path(dataset_root)
    speaker_dir = root / speaker
    
    if not speaker_dir.exists():
        raise ValueError(f"Speaker {speaker} not found")
    
    word_dirs = [d for d in speaker_dir.iterdir() if d.is_dir()]
    valid_samples = []
    
    for word_dir in word_dirs:
        bengali_word = word_dir.name
        frame_files = list(word_dir.glob("*.jpg"))
        
        if len(frame_files) == 0:
            continue
        
        for split in ['train', 'val', 'test']:
            prepared_dir = root / 'prepared' / split / speaker / bengali_word
            if prepared_dir.exists():
                pt_files = list(prepared_dir.glob("*.pt"))
                if pt_files:
                    for pt_file in pt_files:
                        try:
                            obj = torch.load(pt_file, map_location='cpu')
                            if isinstance(obj, dict) and ('frames' in obj or 'x' in obj):
                                valid_samples.append({
                                    'word_dir': word_dir,
                                    'bengali_word': bengali_word,
                                    'frame_files': frame_files,
                                    'prepared_pt': pt_file,
                                    'split': split
                                })
                                break
                        except:
                            continue
                if valid_samples and valid_samples[-1]['bengali_word'] == bengali_word:
                    break
    
    if not valid_samples:
        raise ValueError(f"No valid samples found")
    
    if target_word:
        matching = [s for s in valid_samples if s['bengali_word'] == target_word]
        if not matching:
            raise ValueError(f"Word {target_word} not found")
        sample = matching[0]
    else:
        sample = random.choice(valid_samples)
    
    return sample


def load_prepared_mouth_crop(pt_path, num_frames=7):
    """Load frames from prepared .pt file."""
    obj = torch.load(pt_path, map_location='cpu')
    
    if isinstance(obj, dict) and 'frames' in obj:
        frames_tensor = obj['frames']
    elif isinstance(obj, dict) and 'x' in obj:
        frames_tensor = obj['x']
    else:
        raise ValueError(f"Unknown format")
    
    if isinstance(frames_tensor, torch.Tensor):
        frames_np = frames_tensor.numpy()
    else:
        frames_np = frames_tensor
    
    if frames_np.dtype != np.uint8:
        if frames_np.max() <= 1.0:
            frames_np = (frames_np * 255).astype(np.uint8)
        else:
            frames_np = frames_np.astype(np.uint8)
    
    T = frames_np.shape[0]
    if T > num_frames:
        indices = np.linspace(0, T-1, num_frames).astype(int)
        frames_np = frames_np[indices]
    elif T < num_frames:
        padding = np.repeat(frames_np[-1:], num_frames - T, axis=0)
        frames_np = np.concatenate([frames_np, padding], axis=0)
    
    return frames_np


def create_ctc_alignment(transliterated_word, num_frames):
    """Create realistic CTC alignment."""
    chars = list(transliterated_word.lower())
    
    if num_frames <= len(chars):
        alignment = chars[:num_frames]
        while len(alignment) < num_frames:
            alignment.append('-')
    else:
        alignment = []
        positions = np.linspace(0, num_frames-1, len(chars)).astype(int)
        
        for i in range(num_frames):
            if i in positions:
                char_idx = list(positions).index(i)
                alignment.append(chars[char_idx])
            else:
                alignment.append('-')
    
    return alignment


def extract_data_for_visualization(
    dataset_root='/home/thesis/Thesis/AVEC/datasets/LipBengal',
    speaker='s100',
    target_word=None,
    output_dir='export/paper_samples/bengali_ctc_data'
):
    """Extract all data needed for visualization."""
    
    print(f"\n{'='*60}")
    print("Extracting Bengali Lip Reading Data")
    print(f"{'='*60}\n")
    
    # Load sample
    sample = load_lipbengal_sample_correct(dataset_root, speaker, target_word)
    
    bengali_word = sample['bengali_word']
    transliterated = to_banglish(bengali_word)
    frame_files = sample['frame_files']
    prepared_pt = sample['prepared_pt']
    
    print(f"Bengali Word: {bengali_word}")
    print(f"Transliterated: {transliterated}")
    print(f"Speaker: {speaker}")
    print(f"Split: {sample['split']}")
    print(f"Total frames available: {len(frame_files)}")
    
    # Create output directory
    output_path = Path(output_dir)
    if output_path.exists():
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Determine number of frames to use
    num_frames = min(7, len(frame_files))
    
    # Load preprocessed lip crops
    mouth_frames = load_prepared_mouth_crop(prepared_pt, num_frames)
    
    # Sample original frames to match
    indices = np.linspace(0, len(frame_files)-1, num_frames).astype(int)
    
    # Create CTC alignment
    alignment = create_ctc_alignment(transliterated, num_frames)
    chars = list(transliterated.lower())
    
    # Save data
    print(f"\n{'='*60}")
    print("Saving Data")
    print(f"{'='*60}\n")
    
    # 1. Save original video frames
    frames_dir = output_path / '1_video_frames'
    frames_dir.mkdir(exist_ok=True)
    
    for i, idx in enumerate(indices):
        src = frame_files[idx]
        dst = frames_dir / f'frame_{i+1:02d}.jpg'
        shutil.copy(src, dst)
        print(f"✓ Video frame {i+1}: {dst.name}")
    
    # 2. Save preprocessed lip crops
    lips_dir = output_path / '2_lip_crops_preprocessed'
    lips_dir.mkdir(exist_ok=True)
    
    for i, lip_frame in enumerate(mouth_frames):
        img = Image.fromarray(lip_frame)
        dst = lips_dir / f'lip_{i+1:02d}.png'
        img.save(dst)
        print(f"✓ Lip crop {i+1}: {dst.name}")
    
    # 3. Save text information as JSON
    text_data = {
        'bengali_word': bengali_word,
        'transliterated_word': transliterated,
        'characters': chars,
        'num_frames': num_frames,
        'ctc_alignment': alignment,
        'frame_to_character_mapping': []
    }
    
    # Create detailed mapping
    for i in range(num_frames):
        aligned_char = alignment[i]
        mapping = {
            'frame_number': i + 1,
            'video_frame_file': f'frame_{i+1:02d}.jpg',
            'lip_crop_file': f'lip_{i+1:02d}.png',
            'aligned_character': aligned_char if aligned_char != '-' else 'blank',
            'is_blank': aligned_char == '-'
        }
        
        # Find which character in the word this aligns to
        if aligned_char != '-' and aligned_char in chars:
            char_position = chars.index(aligned_char) + 1
            mapping['character_position_in_word'] = char_position
        else:
            mapping['character_position_in_word'] = None
        
        text_data['frame_to_character_mapping'].append(mapping)
    
    # Save JSON
    json_file = output_path / '3_text_mapping.json'
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(text_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Text mapping: {json_file.name}")
    
    # 4. Create README
    readme = output_path / 'README.txt'
    with open(readme, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("BENGALI LIP READING DATA FOR VISUALIZATION\n")
        f.write("="*70 + "\n\n")
        
        f.write("WORD INFORMATION:\n")
        f.write(f"  Bengali: {bengali_word}\n")
        f.write(f"  Transliterated (English): {transliterated}\n")
        f.write(f"  Characters: {' '.join(chars)}\n\n")
        
        f.write("DIRECTORY STRUCTURE:\n")
        f.write("  1_video_frames/          - Original video frames (7 frames)\n")
        f.write("  2_lip_crops_preprocessed/ - Preprocessed lip region crops\n")
        f.write("  3_text_mapping.json      - Complete text and alignment info\n\n")
        
        f.write("VISUALIZATION FLOW:\n")
        f.write("  Step 1: Show video frames → lip crops (cropping process)\n")
        f.write("  Step 2: Show Bengali word → Transliteration → English characters\n")
        f.write("  Step 3: Show CTC alignment (lip frames ↔ characters)\n")
        f.write("  Step 4: Show final output word\n\n")
        
        f.write("FRAME-TO-CHARACTER MAPPING:\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'Frame':<8} {'Video File':<20} {'Lip File':<20} {'Character':<12}\n")
        f.write("-" * 70 + "\n")
        
        for i in range(num_frames):
            aligned_char = alignment[i]
            char_display = aligned_char if aligned_char != '-' else '∅ (blank)'
            f.write(f"{i+1:<8} frame_{i+1:02d}.jpg{'':<9} lip_{i+1:02d}.png{'':<9} {char_display:<12}\n")
        
        f.write("\n" + "="*70 + "\n")
        f.write("CTC ALIGNMENT EXPLANATION:\n")
        f.write("="*70 + "\n")
        f.write(f"Word: {transliterated}\n")
        f.write(f"Characters: {' '.join(chars)}\n")
        f.write(f"Alignment: {' '.join(alignment)}\n")
        f.write("\nNote: '∅' or 'blank' means CTC blank token (no character output)\n")
    
    print(f"✓ README: {readme.name}\n")
    
    # Print summary table
    print(f"{'='*60}")
    print("FRAME-TO-CHARACTER MAPPING")
    print(f"{'='*60}")
    print(f"{'Frame':<8} {'Video':<18} {'Lip':<18} {'Char':<10}")
    print(f"{'-'*60}")
    for i in range(num_frames):
        aligned_char = alignment[i]
        char_display = aligned_char if aligned_char != '-' else '∅'
        print(f"{i+1:<8} frame_{i+1:02d}.jpg{'':<7} lip_{i+1:02d}.png{'':<7} {char_display:<10}")
    
    print(f"\n{'='*60}")
    print(f"✅ All data saved to: {output_path}")
    print(f"{'='*60}\n")
    
    print("You now have:")
    print(f"  • {num_frames} video frames")
    print(f"  • {num_frames} preprocessed lip crops")
    print(f"  • Complete text mappings in JSON")
    print(f"  • Detailed README with alignment info")
    print("\nUse this data to create your own visualization!\n")
    
    return output_path


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract Bengali lip reading data')
    parser.add_argument('--dataset', type=str, 
                       default='/home/thesis/Thesis/AVEC/datasets/LipBengal')
    parser.add_argument('--speaker', type=str, default='s100')
    parser.add_argument('--word', type=str, default=None)
    parser.add_argument('--output', type=str, 
                       default='export/paper_samples/bengali_ctc_data')
    
    args = parser.parse_args()
    
    try:
        extract_data_for_visualization(
            dataset_root=args.dataset,
            speaker=args.speaker,
            target_word=args.word,
            output_dir=args.output
        )
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

