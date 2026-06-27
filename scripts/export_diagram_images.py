#!/usr/bin/env python3
"""
Export the actual images (video frames and lip crops) used in the diagrams.
"""

import sys
from pathlib import Path
import numpy as np
import torch
from PIL import Image
import shutil
import json

sys.path.insert(0, str(Path(__file__).parent.parent))


def load_and_export_samples(dataset_root='/home/thesis/Thesis/AVEC/datasets/LipBengal',
                            output_dir='export/diagram_images',
                            num_frames=5):
    """Load and export sample images used in diagrams."""
    
    print("\n" + "="*70)
    print("EXPORTING DIAGRAM IMAGES")
    print("="*70 + "\n")
    
    root = Path(dataset_root)
    output_path = Path(output_dir)
    
    # Clear and create output directory
    if output_path.exists():
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    video_frames_dir = output_path / '1_video_frames'
    lip_crops_dir = output_path / '2_lip_crops'
    video_frames_dir.mkdir(exist_ok=True)
    lip_crops_dir.mkdir(exist_ok=True)
    
    # Find and load samples
    sample_found = False
    
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
            
            # Try to find corresponding lip crops
            lip_frames = None
            split_found = None
            
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
                            
                            split_found = split
                            break
                        except Exception as e:
                            print(f"  Warning: Could not load {pt_files[0]}: {e}")
                            continue
                
                if lip_frames is not None:
                    break
            
            if lip_frames is not None:
                print(f"✓ Found sample:")
                print(f"  Speaker: {speaker}")
                print(f"  Word: {bengali_word}")
                print(f"  Split: {split_found}")
                print(f"  Video frames: {len(frame_files)}")
                print(f"  Lip frames: {lip_frames.shape[0]}")
                print()
                
                # Sample frames evenly
                video_indices = np.linspace(0, len(frame_files)-1, num_frames).astype(int)
                lip_indices = np.linspace(0, lip_frames.shape[0]-1, num_frames).astype(int)
                
                # Export video frames
                print("Exporting video frames...")
                for i, idx in enumerate(video_indices):
                    src_frame = frame_files[idx]
                    dst_frame = video_frames_dir / f'frame_{i+1:02d}.jpg'
                    
                    # Copy and resize for consistency
                    img = Image.open(src_frame)
                    img.save(dst_frame, quality=95)
                    print(f"  ✓ {dst_frame.name}")
                
                print("\nExporting lip crops...")
                for i, idx in enumerate(lip_indices):
                    lip_frame = lip_frames[idx]
                    dst_lip = lip_crops_dir / f'lip_crop_{i+1:02d}.png'
                    
                    img = Image.fromarray(lip_frame)
                    img.save(dst_lip)
                    print(f"  ✓ {dst_lip.name}")
                
                # Save metadata
                metadata = {
                    'speaker': speaker,
                    'word_bengali': bengali_word,
                    'split': split_found,
                    'num_frames_exported': num_frames,
                    'total_video_frames': len(frame_files),
                    'total_lip_frames': int(lip_frames.shape[0]),
                    'video_frame_size': f"{img.size[0]}x{img.size[1]}",
                    'lip_frame_size': f"{lip_frame.shape[0]}x{lip_frame.shape[1]}",
                    'video_frames': [f'frame_{i+1:02d}.jpg' for i in range(num_frames)],
                    'lip_crops': [f'lip_crop_{i+1:02d}.png' for i in range(num_frames)]
                }
                
                metadata_file = output_path / 'metadata.json'
                with open(metadata_file, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, indent=2, ensure_ascii=False)
                
                print(f"\n✓ Metadata saved to: {metadata_file.name}")
                
                # Create README
                readme_file = output_path / 'README.txt'
                with open(readme_file, 'w', encoding='utf-8') as f:
                    f.write("="*70 + "\n")
                    f.write("DIAGRAM IMAGES EXPORT\n")
                    f.write("="*70 + "\n\n")
                    
                    f.write("This folder contains the actual images used in the system diagrams.\n\n")
                    
                    f.write("CONTENTS:\n")
                    f.write("-" * 70 + "\n")
                    f.write("1_video_frames/     - Original raw video frames from the dataset\n")
                    f.write("2_lip_crops/        - Preprocessed lip region crops (88x88 grayscale)\n")
                    f.write("metadata.json       - Detailed information about the samples\n")
                    f.write("README.txt          - This file\n\n")
                    
                    f.write("SAMPLE INFORMATION:\n")
                    f.write("-" * 70 + "\n")
                    f.write(f"Speaker:        {speaker}\n")
                    f.write(f"Bengali Word:   {bengali_word}\n")
                    f.write(f"Dataset Split:  {split_found}\n")
                    f.write(f"Frames:         {num_frames}\n\n")
                    
                    f.write("VIDEO FRAMES:\n")
                    f.write("-" * 70 + "\n")
                    f.write("These are the original full-resolution video frames showing the\n")
                    f.write("speaker's face. These frames are the input to the preprocessing\n")
                    f.write("pipeline (face detection, landmark detection, lip cropping).\n\n")
                    
                    for i in range(num_frames):
                        f.write(f"  • frame_{i+1:02d}.jpg\n")
                    
                    f.write("\n")
                    f.write("LIP CROPS:\n")
                    f.write("-" * 70 + "\n")
                    f.write("These are the preprocessed 88x88 grayscale lip region crops that\n")
                    f.write("are fed into the 3D CNN visual encoder. These crops are extracted\n")
                    f.write("using facial landmark detection and normalized.\n\n")
                    
                    for i in range(num_frames):
                        f.write(f"  • lip_crop_{i+1:02d}.png\n")
                    
                    f.write("\n")
                    f.write("USAGE:\n")
                    f.write("-" * 70 + "\n")
                    f.write("These images can be used for:\n")
                    f.write("  • Creating custom diagrams and figures\n")
                    f.write("  • Presentations and slides\n")
                    f.write("  • Paper figures and illustrations\n")
                    f.write("  • Documentation and tutorials\n")
                    f.write("  • Demonstrating the preprocessing pipeline\n\n")
                    
                    f.write("="*70 + "\n")
                
                print(f"✓ README saved to: {readme_file.name}")
                
                sample_found = True
                break
        
        if sample_found:
            break
    
    if not sample_found:
        print("✗ No suitable samples found!")
        return None
    
    print("\n" + "="*70)
    print(f"✅ EXPORT COMPLETE: {output_path}")
    print("="*70 + "\n")
    
    print("Exported:")
    print(f"  • {num_frames} video frames (JPG)")
    print(f"  • {num_frames} lip crops (PNG)")
    print(f"  • 1 metadata file (JSON)")
    print(f"  • 1 README file (TXT)")
    print()
    
    return output_path


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Export diagram images')
    parser.add_argument('--dataset', type=str,
                       default='/home/thesis/Thesis/AVEC/datasets/LipBengal',
                       help='Path to dataset')
    parser.add_argument('--output', type=str,
                       default='export/diagram_images',
                       help='Output directory')
    parser.add_argument('--frames', type=int, default=5,
                       help='Number of frames to export')
    
    args = parser.parse_args()
    
    try:
        load_and_export_samples(args.dataset, args.output, args.frames)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)





