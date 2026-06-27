#!/usr/bin/env python3
"""
Extract lip frames and text information for manual visualization creation.
Saves individual frame images and a text file with all alignment information.
"""

import sys
from pathlib import Path
import torch
import numpy as np
from PIL import Image
import json
import shutil
import random

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.banglish_lookup import to_banglish


def load_lipbengal_sample(dataset_root, speaker='s100', target_word=None):
    """Load a complete sample from LipBengal dataset."""
    root = Path(dataset_root)
    speaker_dir = root / speaker
    
    if not speaker_dir.exists():
        raise ValueError(f"Speaker {speaker} not found")
    
    word_dirs = [d for d in speaker_dir.iterdir() if d.is_dir()]
    valid_samples = []
    
    for word_dir in word_dirs:
        bengali_word = word_dir.name
        frame_files = sorted(list(word_dir.glob("*.jpg")))
        
        if len(frame_files) == 0:
            continue
        
        # Find prepared file
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
        raise ValueError(f"No valid samples found for speaker {speaker}")
    
    if target_word:
        matching = [s for s in valid_samples if s['bengali_word'] == target_word]
        if not matching:
            available = [s['bengali_word'] for s in valid_samples[:10]]
            raise ValueError(f"Word '{target_word}' not found. Available samples: {available}")
        sample = matching[0]
    else:
        sample = random.choice(valid_samples)
    
    return sample


def load_prepared_frames(pt_path):
    """Load all frames from prepared .pt file."""
    obj = torch.load(pt_path, map_location='cpu')
    
    if isinstance(obj, dict) and 'frames' in obj:
        frames_tensor = obj['frames']
    elif isinstance(obj, dict) and 'x' in obj:
        frames_tensor = obj['x']
    else:
        raise ValueError(f"Unknown format in {pt_path}")
    
    if isinstance(frames_tensor, torch.Tensor):
        frames_np = frames_tensor.numpy()
    else:
        frames_np = frames_tensor
    
    # Convert to uint8 if needed
    if frames_np.dtype != np.uint8:
        if frames_np.max() <= 1.0:
            frames_np = (frames_np * 255).astype(np.uint8)
        else:
            frames_np = frames_np.astype(np.uint8)
    
    return frames_np


def sample_frames_with_indices(frames, num_samples=None):
    """Sample frames evenly or return all."""
    total_frames = len(frames)
    
    if num_samples is None or num_samples >= total_frames:
        # Return all frames
        return frames, list(range(total_frames))
    
    # Sample evenly across the sequence
    indices = np.linspace(0, total_frames - 1, num_samples).astype(int)
    sampled_frames = [frames[i] for i in indices]
    
    return sampled_frames, indices.tolist()


def create_character_alignment(word, num_frames):
    """Create alignment between characters and frames."""
    chars = list(word.lower())
    
    if num_frames <= len(chars):
        # More characters than frames
        alignment = chars[:num_frames]
        while len(alignment) < num_frames:
            alignment.append('')
    else:
        # More frames than characters - spread characters evenly
        alignment = [''] * num_frames
        
        # Distribute characters evenly
        positions = np.linspace(0, num_frames - 1, len(chars)).astype(int)
        
        for i, char in enumerate(chars):
            pos = positions[i]
            alignment[pos] = char
    
    return alignment


def extract_frames(
    dataset_root,
    speaker,
    target_word,
    output_dir,
    num_frames=None
):
    """Extract frames and create text files with all information."""
    
    print("\n" + "="*70)
    print("EXTRACTING FRAMES FOR MANUAL VISUALIZATION")
    print("="*70)
    
    # Load sample
    print(f"\n📂 Loading sample from speaker: {speaker}")
    if target_word:
        print(f"   Target word: {target_word}")
    
    sample = load_lipbengal_sample(dataset_root, speaker, target_word)
    
    bengali_word = sample['bengali_word']
    transliterated = to_banglish(bengali_word)
    
    print(f"\n📝 Word Information:")
    print(f"   Bengali: {bengali_word}")
    print(f"   English: {transliterated}")
    print(f"   Split: {sample['split']}")
    
    # Load frames
    print(f"\n🎬 Loading preprocessed frames...")
    all_frames = load_prepared_frames(sample['prepared_pt'])
    total_frames = len(all_frames)
    print(f"   Total frames available: {total_frames}")
    
    # Sample or use all frames
    if num_frames and num_frames < total_frames:
        print(f"\n✂️  Sampling {num_frames} frames...")
        frames, frame_indices = sample_frames_with_indices(all_frames, num_frames)
    else:
        print(f"\n📦 Using all {total_frames} frames...")
        frames = all_frames
        frame_indices = list(range(total_frames))
    
    print(f"   Selected frame indices: {frame_indices}")
    
    # Create alignment
    alignment = create_character_alignment(transliterated, len(frames))
    
    # Create output directory
    output_path = Path(output_dir)
    if output_path.exists():
        print(f"\n🗑️  Removing existing directory: {output_path}")
        shutil.rmtree(output_path)
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create frames subdirectory
    frames_dir = output_path / 'frames'
    frames_dir.mkdir(exist_ok=True)
    
    # Save individual frames
    print(f"\n💾 Saving {len(frames)} frames...")
    for i, (frame, orig_idx) in enumerate(zip(frames, frame_indices)):
        frame_img = Image.fromarray(frame)
        frame_path = frames_dir / f'frame_{i+1:03d}.png'
        frame_img.save(frame_path)
        if (i + 1) % 10 == 0 or i == len(frames) - 1:
            print(f"   Saved {i+1}/{len(frames)} frames...")
    
    # Create detailed text file
    info_file = output_path / 'alignment_info.txt'
    print(f"\n📄 Creating alignment information file...")
    
    with open(info_file, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("LIP ALIGNMENT DATA FOR MANUAL VISUALIZATION\n")
        f.write("="*70 + "\n\n")
        
        f.write("WORD INFORMATION:\n")
        f.write("-"*70 + "\n")
        f.write(f"Bengali Word:       {bengali_word}\n")
        f.write(f"English (Banglish): {transliterated}\n")
        f.write(f"Speaker:            {speaker}\n")
        f.write(f"Split:              {sample['split']}\n")
        f.write(f"Total Frames:       {total_frames}\n")
        f.write(f"Displayed Frames:   {len(frames)}\n")
        f.write("\n")
        
        f.write("CHARACTERS IN WORD:\n")
        f.write("-"*70 + "\n")
        chars = list(transliterated.lower())
        for i, char in enumerate(chars, 1):
            f.write(f"  {i}. '{char}'\n")
        f.write(f"\nTotal: {len(chars)} characters\n\n")
        
        f.write("FRAME-TO-CHARACTER ALIGNMENT:\n")
        f.write("="*70 + "\n")
        f.write(f"{'Frame':<8} {'File':<25} {'Original':<12} {'Character':<15}\n")
        f.write("-"*70 + "\n")
        
        for i, (orig_idx, char) in enumerate(zip(frame_indices, alignment)):
            frame_file = f'frame_{i+1:03d}.png'
            char_display = f"'{char}'" if char else "(blank)"
            f.write(f"{i+1:<8} {frame_file:<25} Frame {orig_idx+1:<8} {char_display:<15}\n")
        
        f.write("\n" + "="*70 + "\n")
        f.write("USAGE INSTRUCTIONS:\n")
        f.write("="*70 + "\n")
        f.write("1. All frames are in the 'frames/' subdirectory\n")
        f.write("2. Frame files are numbered: frame_001.png, frame_002.png, etc.\n")
        f.write("3. Use the alignment table above to match characters to frames\n")
        f.write("4. Blank frames have no character (CTC blank token)\n")
        f.write("5. Original frame numbers show position in source video\n")
        f.write("\n")
        f.write("LAYOUT SUGGESTION:\n")
        f.write("  Top:    Bengali word: " + bengali_word + "\n")
        f.write("          ↓\n")
        f.write("  Middle: English transliteration: \"" + transliterated + "\"\n")
        f.write("          ↓\n")
        f.write("  Bottom: Frames with characters above each frame\n")
        f.write("\n")
    
    # Save as JSON as well for programmatic access
    json_file = output_path / 'alignment_info.json'
    metadata = {
        'bengali_word': bengali_word,
        'transliterated_word': transliterated,
        'speaker': speaker,
        'split': sample['split'],
        'total_frames': total_frames,
        'displayed_frames': len(frames),
        'characters': chars,
        'frames': [
            {
                'display_number': i + 1,
                'file_name': f'frame_{i+1:03d}.png',
                'original_frame_number': frame_indices[i] + 1,
                'character': alignment[i] if alignment[i] else None,
                'is_blank': not alignment[i]
            }
            for i in range(len(frames))
        ]
    }
    
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    # Create simple character mapping file
    char_file = output_path / 'characters.txt'
    with open(char_file, 'w', encoding='utf-8') as f:
        f.write(f"Bengali: {bengali_word}\n")
        f.write(f"English: {transliterated}\n")
        f.write(f"Characters: {' '.join(chars)}\n\n")
        f.write("Frame-Character Mapping:\n")
        for i, char in enumerate(alignment, 1):
            f.write(f"Frame {i}: {char if char else '(blank)'}\n")
    
    # Print summary
    print("\n" + "="*70)
    print("✅ EXTRACTION COMPLETE!")
    print("="*70)
    print(f"\nOutput directory: {output_path}")
    print(f"\nFiles created:")
    print(f"  📁 frames/              - {len(frames)} frame images (PNG)")
    print(f"  📄 alignment_info.txt   - Detailed alignment information")
    print(f"  📄 alignment_info.json  - Machine-readable metadata")
    print(f"  📄 characters.txt       - Simple character mapping")
    
    print(f"\n📝 Word: {bengali_word} → {transliterated}")
    print(f"🎬 Frames: {len(frames)}")
    print(f"🔤 Characters: {' '.join(chars)}")
    
    print("\n" + "="*70)
    print("You can now create your visualization manually using these files!")
    print("="*70 + "\n")
    
    return output_path


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Extract lip frames for manual visualization creation'
    )
    parser.add_argument('--dataset', type=str,
                       default='/home/thesis/Thesis/AVEC/datasets/LipBengal',
                       help='Path to LipBengal dataset')
    parser.add_argument('--speaker', type=str, default='s100',
                       help='Speaker ID (e.g., s100)')
    parser.add_argument('--word', type=str, required=True,
                       help='Specific Bengali word to extract')
    parser.add_argument('--output', type=str,
                       default='export/manual_viz_data',
                       help='Output directory path')
    parser.add_argument('--num-frames', type=int, default=None,
                       help='Number of frames to extract (default: all frames)')
    
    args = parser.parse_args()
    
    try:
        extract_frames(
            dataset_root=args.dataset,
            speaker=args.speaker,
            target_word=args.word,
            output_dir=args.output,
            num_frames=args.num_frames
        )
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

