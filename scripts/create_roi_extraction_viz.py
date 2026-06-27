#!/usr/bin/env python3
"""
Create publication-quality ROI extraction visualization for paper.
Shows the complete pipeline: Original Frame → Face Detection → Landmarks → ROI → Final Crop

This visualization demonstrates the 5-step process used to extract mouth regions
from video frames for audio-visual speech recognition.
"""

import sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle, Circle, FancyBboxPatch
import cv2
import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Try to import detection libraries
try:
    import mediapipe as mp
    _MP_OK = True
except Exception:
    mp = None
    _MP_OK = False

try:
    from ibug.face_alignment import FANPredictor
    _FAN_OK = True
except Exception:
    FANPredictor = None
    _FAN_OK = False

try:
    import torch
    _TORCH_OK = True
except Exception:
    torch = None
    _TORCH_OK = False


def detect_face_mediapipe(img_rgb):
    """Detect face using MediaPipe Face Detection."""
    if not _MP_OK:
        return None
    
    try:
        face_detection = mp.solutions.face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.5
        )
        results = face_detection.process(img_rgb)
        
        if results.detections:
            detection = results.detections[0]
            bbox = detection.location_data.relative_bounding_box
            
            h, w = img_rgb.shape[:2]
            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            width = int(bbox.width * w)
            height = int(bbox.height * h)
            
            face_detection.close()
            return (x, y, x + width, y + height)
        
        face_detection.close()
    except Exception as e:
        print(f"MediaPipe face detection failed: {e}")
    
    return None


def detect_landmarks_mediapipe(img_rgb):
    """Detect facial landmarks using MediaPipe FaceMesh."""
    if not _MP_OK:
        return None
    
    try:
        face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )
        
        results = face_mesh.process(img_rgb)
        
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]
            h, w = img_rgb.shape[:2]
            
            # Convert to pixel coordinates
            points = []
            for lm in landmarks.landmark:
                x = int(lm.x * w)
                y = int(lm.y * h)
                points.append((x, y))
            
            face_mesh.close()
            return np.array(points)
        
        face_mesh.close()
    except Exception as e:
        print(f"MediaPipe landmarks detection failed: {e}")
    
    return None


def detect_landmarks_fan(img_rgb, device='cpu'):
    """Detect 68 facial landmarks using FAN."""
    if not _FAN_OK:
        return None
    
    try:
        fan = FANPredictor(device=device, model=None)
        result = fan(img_rgb)
        
        if result is not None:
            if isinstance(result, tuple) and len(result) >= 1:
                landmarks_list = result[0]
            else:
                landmarks_list = result
            
            if landmarks_list is not None and len(landmarks_list) > 0:
                pts = landmarks_list[0]
                pts_array = np.array(pts, dtype=np.float32)
                
                if pts_array.ndim == 2 and pts_array.shape[0] >= 68:
                    return pts_array
    except Exception as e:
        print(f"FAN landmarks detection failed: {e}")
    
    return None


def get_mouth_landmarks_indices(landmark_type='fan'):
    """Get indices for mouth landmarks."""
    if landmark_type == 'fan':
        # FAN 68-point: mouth is landmarks 48-67
        return list(range(48, 68))
    elif landmark_type == 'mediapipe':
        # MediaPipe FaceMesh: specific lip indices
        return [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 
                78, 191, 80, 81, 82, 13, 312, 311, 310, 415]
    return []


def compute_mouth_bbox(landmarks, mouth_indices, enlarge_factor=1.6):
    """Compute mouth bounding box from landmarks."""
    if landmarks is None or len(mouth_indices) == 0:
        return None
    
    try:
        mouth_points = landmarks[mouth_indices]
        x_coords = mouth_points[:, 0]
        y_coords = mouth_points[:, 1]
        
        x_min, x_max = x_coords.min(), x_coords.max()
        y_min, y_max = y_coords.min(), y_coords.max()
        
        # Enlarge the bounding box
        cx = (x_min + x_max) / 2
        cy = (y_min + y_max) / 2
        w = (x_max - x_min) * enlarge_factor
        h = (y_max - y_min) * enlarge_factor
        
        x0 = int(cx - w / 2)
        y0 = int(cy - h / 2)
        x1 = int(cx + w / 2)
        y1 = int(cy + h / 2)
        
        return (x0, y0, x1, y1)
    except Exception as e:
        print(f"Error computing mouth bbox: {e}")
        return None


def extract_and_resize_roi(img_rgb, bbox, output_size=88):
    """Extract ROI and resize to standard size."""
    try:
        x0, y0, x1, y1 = bbox
        
        # Clamp to image boundaries
        h, w = img_rgb.shape[:2]
        x0 = max(0, x0)
        y0 = max(0, y0)
        x1 = min(w, x1)
        y1 = min(h, y1)
        
        # Extract ROI
        roi = img_rgb[y0:y1, x0:x1]
        
        if roi.size == 0:
            return None
        
        # Convert to grayscale
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
        
        # Resize to standard size
        roi_resized = cv2.resize(roi_gray, (output_size, output_size), 
                                  interpolation=cv2.INTER_AREA)
        
        return roi_resized
    except Exception as e:
        print(f"Error extracting ROI: {e}")
        return None


def create_roi_extraction_visualization(
    img_rgb,
    face_bbox,
    landmarks,
    mouth_bbox,
    final_crop,
    output_path,
    landmark_type='fan',
    enlarge_factor=1.6,
    output_size=88
):
    """Create 5-panel visualization showing complete ROI extraction pipeline."""
    
    fig = plt.figure(figsize=(20, 5))
    gs = gridspec.GridSpec(1, 5, wspace=0.15)
    
    # Colors
    color_face = '#2E86AB'      # Blue for face bbox
    color_landmarks = '#A23B72'  # Purple for landmarks
    color_mouth = '#F18F01'      # Orange for mouth landmarks
    color_roi = '#C73E1D'        # Red for ROI bbox
    
    # --------------------- PANEL 1: Original Frame ---------------------
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(img_rgb)
    ax1.set_title('Step 1: Input Frame\n(Original Video)', 
                  fontsize=14, fontweight='bold', pad=10)
    ax1.axis('off')
    
    # Add specifications
    h, w = img_rgb.shape[:2]
    ax1.text(0.5, -0.08, f'{w}×{h} RGB',
             transform=ax1.transAxes, ha='center', fontsize=10,
             color='#555')
    
    # --------------------- PANEL 2: Face Detection ---------------------
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(img_rgb)
    
    if face_bbox is not None:
        x0, y0, x1, y1 = face_bbox
        rect = Rectangle((x0, y0), x1 - x0, y1 - y0,
                        linewidth=3, edgecolor=color_face,
                        facecolor='none', linestyle='-')
        ax2.add_patch(rect)
        
        # Add corner markers
        marker_size = 15
        for corner_x, corner_y in [(x0, y0), (x1, y0), (x0, y1), (x1, y1)]:
            ax2.plot(corner_x, corner_y, 's', color=color_face, 
                    markersize=marker_size, markeredgewidth=2,
                    markerfacecolor='none')
    
    ax2.set_title('Step 2: Face Detection\n(RetinaFace/MediaPipe)', 
                  fontsize=14, fontweight='bold', pad=10)
    ax2.axis('off')
    ax2.text(0.5, -0.08, 'Face Bounding Box',
             transform=ax2.transAxes, ha='center', fontsize=10,
             color=color_face, fontweight='600')
    
    # --------------------- PANEL 3: Landmark Detection ---------------------
    ax3 = fig.add_subplot(gs[0, 1])
    ax3.imshow(img_rgb)
    
    if landmarks is not None:
        # Draw all landmarks (smaller)
        ax3.scatter(landmarks[:, 0], landmarks[:, 1], 
                   c=color_landmarks, s=15, alpha=0.6, zorder=2)
        
        # Highlight mouth landmarks (larger)
        mouth_indices = get_mouth_landmarks_indices(landmark_type)
        if len(mouth_indices) > 0 and len(landmarks) > max(mouth_indices):
            mouth_points = landmarks[mouth_indices]
            ax3.scatter(mouth_points[:, 0], mouth_points[:, 1],
                       c=color_mouth, s=40, alpha=1.0, zorder=3,
                       edgecolors='white', linewidths=1)
            
            # Draw mouth contour
            mouth_points_closed = np.vstack([mouth_points, mouth_points[0]])
            ax3.plot(mouth_points_closed[:, 0], mouth_points_closed[:, 1],
                    color=color_mouth, linewidth=2, alpha=0.7, zorder=1)
    
    ax3.set_title('Step 3: Landmark Detection\n(FAN 68-point)', 
                  fontsize=14, fontweight='bold', pad=10)
    ax3.axis('off')
    
    num_landmarks = len(landmarks) if landmarks is not None else 0
    num_mouth = len(get_mouth_landmarks_indices(landmark_type))
    ax3.text(0.5, -0.08, f'{num_landmarks} landmarks ({num_mouth} mouth)',
             transform=ax3.transAxes, ha='center', fontsize=10,
             color=color_mouth, fontweight='600')
    
    # --------------------- PANEL 4: ROI Localization ---------------------
    ax4 = fig.add_subplot(gs[0, 3])
    ax4.imshow(img_rgb)
    
    if mouth_bbox is not None:
        x0, y0, x1, y1 = mouth_bbox
        
        # Draw ROI rectangle with rounded corners
        rect = FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                             boxstyle="round,pad=5",
                             linewidth=4, edgecolor=color_roi,
                             facecolor='none', linestyle='-')
        ax4.add_patch(rect)
        
        # Draw center crosshair
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        crosshair_size = 20
        ax4.plot([cx - crosshair_size, cx + crosshair_size], [cy, cy],
                color=color_roi, linewidth=2, alpha=0.8)
        ax4.plot([cx, cx], [cy - crosshair_size, cy + crosshair_size],
                color=color_roi, linewidth=2, alpha=0.8)
        ax4.plot(cx, cy, 'o', color=color_roi, markersize=8,
                markerfacecolor='white', markeredgewidth=2)
        
        # Draw corner markers
        marker_size = 15
        for corner_x, corner_y in [(x0, y0), (x1, y0), (x0, y1), (x1, y1)]:
            ax4.plot(corner_x, corner_y, 'D', color=color_roi,
                    markersize=marker_size, markeredgewidth=2,
                    markerfacecolor='none')
    
    ax4.set_title('Step 4: ROI Localization\n(Enlarge & Center)', 
                  fontsize=14, fontweight='bold', pad=10)
    ax4.axis('off')
    ax4.text(0.5, -0.08, f'Enlarge Factor: {enlarge_factor}×',
             transform=ax4.transAxes, ha='center', fontsize=10,
             color=color_roi, fontweight='600')
    
    # --------------------- PANEL 5: Final Crop ---------------------
    ax5 = fig.add_subplot(gs[0, 4])
    
    if final_crop is not None:
        ax5.imshow(final_crop, cmap='gray', vmin=0, vmax=255)
        
        # Add grid overlay for technical appearance
        for i in range(0, output_size, output_size // 4):
            ax5.axhline(i, color='cyan', linewidth=0.5, alpha=0.3)
            ax5.axvline(i, color='cyan', linewidth=0.5, alpha=0.3)
    else:
        ax5.text(0.5, 0.5, 'Extraction\nFailed',
                ha='center', va='center', fontsize=16, color='red')
    
    ax5.set_title('Step 5: Preprocessed ROI\n(Model Input)', 
                  fontsize=14, fontweight='bold', pad=10)
    ax5.axis('off')
    ax5.text(0.5, -0.08, f'{output_size}×{output_size} Grayscale',
             transform=ax5.transAxes, ha='center', fontsize=10,
             color='#555', fontweight='600')
    
    # Add overall title
    fig.suptitle('ROI Extraction Pipeline for Audio-Visual Speech Recognition',
                fontsize=16, fontweight='bold', y=0.98)
    
    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none', pad_inches=0.2)
    plt.close()
    
    print(f"✅ Visualization saved: {output_path}")


def process_sample(image_path, output_dir, method='mediapipe+fan', 
                   enlarge_factor=1.6, output_size=88, device='cpu'):
    """Process a single image and create ROI extraction visualization."""
    
    print(f"\n{'='*70}")
    print(f"Processing: {Path(image_path).name}")
    print(f"{'='*70}")
    
    # Load image
    print("📷 Loading image...")
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        print(f"❌ Failed to load image: {image_path}")
        return None
    
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    print(f"   Image size: {img_rgb.shape[1]}×{img_rgb.shape[0]}")
    
    # Step 1: Face Detection
    print("\n🔍 Step 1: Face Detection...")
    face_bbox = detect_face_mediapipe(img_rgb)
    if face_bbox:
        print(f"   ✓ Face detected: {face_bbox}")
    else:
        print(f"   ⚠ Face detection failed, using full image")
        h, w = img_rgb.shape[:2]
        face_bbox = (0, 0, w, h)
    
    # Step 2: Landmark Detection
    print("\n📍 Step 2: Landmark Detection...")
    landmarks = None
    landmark_type = 'fan'
    
    if 'fan' in method.lower() and _FAN_OK:
        print(f"   Trying FAN (device: {device})...")
        landmarks = detect_landmarks_fan(img_rgb, device=device)
        if landmarks is not None:
            print(f"   ✓ FAN detected {len(landmarks)} landmarks")
            landmark_type = 'fan'
    
    if landmarks is None and 'mediapipe' in method.lower() and _MP_OK:
        print(f"   Trying MediaPipe FaceMesh...")
        landmarks = detect_landmarks_mediapipe(img_rgb)
        if landmarks is not None:
            print(f"   ✓ MediaPipe detected {len(landmarks)} landmarks")
            landmark_type = 'mediapipe'
    
    if landmarks is None:
        print(f"   ❌ Landmark detection failed")
        return None
    
    # Step 3: Compute Mouth ROI
    print("\n📏 Step 3: Computing Mouth ROI...")
    mouth_indices = get_mouth_landmarks_indices(landmark_type)
    mouth_bbox = compute_mouth_bbox(landmarks, mouth_indices, enlarge_factor)
    
    if mouth_bbox:
        x0, y0, x1, y1 = mouth_bbox
        print(f"   ✓ Mouth ROI: ({x0}, {y0}) → ({x1}, {y1})")
        print(f"   Size: {x1-x0}×{y1-y0} pixels")
        print(f"   Enlarge factor: {enlarge_factor}×")
    else:
        print(f"   ❌ Failed to compute mouth ROI")
        return None
    
    # Step 4: Extract and Resize
    print("\n✂️  Step 4: Extracting and Resizing ROI...")
    final_crop = extract_and_resize_roi(img_rgb, mouth_bbox, output_size)
    
    if final_crop is not None:
        print(f"   ✓ Final crop: {final_crop.shape[0]}×{final_crop.shape[1]}")
        print(f"   Value range: [{final_crop.min()}, {final_crop.max()}]")
    else:
        print(f"   ❌ Failed to extract ROI")
        return None
    
    # Create visualization
    print("\n🎨 Step 5: Creating Visualization...")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    stem = Path(image_path).stem
    output_path = output_dir / f"roi_extraction_{stem}.png"
    
    create_roi_extraction_visualization(
        img_rgb=img_rgb,
        face_bbox=face_bbox,
        landmarks=landmarks,
        mouth_bbox=mouth_bbox,
        final_crop=final_crop,
        output_path=output_path,
        landmark_type=landmark_type,
        enlarge_factor=enlarge_factor,
        output_size=output_size
    )
    
    # Save metadata
    metadata = {
        'input_image': str(image_path),
        'image_size': {'width': img_rgb.shape[1], 'height': img_rgb.shape[0]},
        'face_bbox': {'x0': int(face_bbox[0]), 'y0': int(face_bbox[1]),
                      'x1': int(face_bbox[2]), 'y1': int(face_bbox[3])},
        'landmark_method': landmark_type,
        'num_landmarks': int(len(landmarks)),
        'num_mouth_landmarks': len(mouth_indices),
        'mouth_bbox': {'x0': int(mouth_bbox[0]), 'y0': int(mouth_bbox[1]),
                       'x1': int(mouth_bbox[2]), 'y1': int(mouth_bbox[3])},
        'enlarge_factor': float(enlarge_factor),
        'output_size': int(output_size),
        'final_crop_stats': {
            'mean': float(final_crop.mean()),
            'std': float(final_crop.std()),
            'min': int(final_crop.min()),
            'max': int(final_crop.max())
        }
    }
    
    metadata_path = output_dir / f"roi_extraction_{stem}.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"📄 Metadata saved: {metadata_path}")
    
    return output_path


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Create ROI extraction visualization for paper'
    )
    parser.add_argument('--input', type=str,
                       help='Input image path (or directory for batch processing)')
    parser.add_argument('--output-dir', type=str,
                       default='export/paper_samples/roi_extraction',
                       help='Output directory for visualizations')
    parser.add_argument('--method', type=str,
                       default='mediapipe+fan',
                       choices=['mediapipe', 'fan', 'mediapipe+fan'],
                       help='Detection method')
    parser.add_argument('--enlarge', type=float, default=1.6,
                       help='ROI enlargement factor (default: 1.6)')
    parser.add_argument('--output-size', type=int, default=88,
                       help='Output crop size (default: 88)')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda'],
                       help='Device for FAN (default: cpu)')
    parser.add_argument('--use-samples', action='store_true',
                       help='Process existing LipBengal sample frames')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("ROI EXTRACTION VISUALIZATION FOR PAPER")
    print("="*70)
    
    # Check available libraries
    print("\n📦 Available Libraries:")
    print(f"   MediaPipe: {'✓' if _MP_OK else '✗'}")
    print(f"   FAN (ibug): {'✓' if _FAN_OK else '✗'}")
    print(f"   PyTorch: {'✓' if _TORCH_OK else '✗'}")
    print(f"   OpenCV: ✓")
    
    if not _MP_OK and not _FAN_OK:
        print("\n❌ Error: No detection library available!")
        print("   Install MediaPipe: pip install mediapipe")
        print("   Install FAN: pip install ibug.face_alignment")
        return
    
    # Determine input images
    images_to_process = []
    
    if args.use_samples:
        # Use existing LipBengal sample frames
        samples_dir = Path('export/dataset_samples/lipbengal')
        if samples_dir.exists():
            images_to_process = list(samples_dir.glob('*.jpg'))[:3]  # Process first 3
            print(f"\n📂 Using {len(images_to_process)} sample frames from {samples_dir}")
        else:
            print(f"\n⚠ Sample directory not found: {samples_dir}")
            print("   Using default image if provided")
    
    if args.input:
        input_path = Path(args.input)
        if input_path.is_file():
            images_to_process = [input_path]
        elif input_path.is_dir():
            images_to_process = list(input_path.glob('*.jpg')) + list(input_path.glob('*.png'))
        print(f"\n📂 Processing {len(images_to_process)} images from: {args.input}")
    
    if not images_to_process:
        print("\n❌ No input images specified!")
        print("   Use --input <path> or --use-samples")
        return
    
    # Process each image
    print(f"\n🚀 Processing {len(images_to_process)} image(s)...")
    results = []
    
    for img_path in images_to_process:
        try:
            result = process_sample(
                image_path=img_path,
                output_dir=args.output_dir,
                method=args.method,
                enlarge_factor=args.enlarge,
                output_size=args.output_size,
                device=args.device
            )
            if result:
                results.append(result)
        except Exception as e:
            print(f"\n❌ Error processing {img_path}: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print("\n" + "="*70)
    print(f"✅ COMPLETE! Processed {len(results)}/{len(images_to_process)} images")
    print("="*70)
    
    if results:
        print("\n📁 Output files:")
        for result in results:
            print(f"   - {result}")
            json_file = Path(str(result).replace('.png', '.json'))
            if json_file.exists():
                print(f"   - {json_file}")
    
    print(f"\n📂 Output directory: {args.output_dir}")
    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠ Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)





