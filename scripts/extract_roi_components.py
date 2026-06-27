#!/usr/bin/env python3
"""
Extract ROI components as separate images for manual visualization.

Outputs:
  1. Original video frame (original.png)
  2. Frame with landmarks overlaid (landmarks.png)
  3. Final cropped lip region (crop.png)
"""

import sys
from pathlib import Path
import numpy as np
import cv2
import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Try to import detection libraries
try:
    from ibug.face_alignment import FANPredictor
    _FAN_OK = True
except Exception:
    FANPredictor = None
    _FAN_OK = False

try:
    import mediapipe as mp
    _MP_OK = True
except Exception:
    mp = None
    _MP_OK = False


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
        print(f"FAN detection failed: {e}")
    
    return None


def detect_landmarks_mediapipe(img_rgb):
    """Detect facial landmarks using MediaPipe FaceMesh."""
    if not _MP_OK:
        return None
    
    try:
        face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True
        )
        
        results = face_mesh.process(img_rgb)
        
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]
            h, w = img_rgb.shape[:2]
            
            points = []
            for lm in landmarks.landmark:
                x = int(lm.x * w)
                y = int(lm.y * h)
                points.append((x, y))
            
            face_mesh.close()
            return np.array(points)
        
        face_mesh.close()
    except Exception as e:
        print(f"MediaPipe detection failed: {e}")
    
    return None


def get_mouth_landmarks_indices(landmark_type='fan'):
    """Get indices for mouth landmarks."""
    if landmark_type == 'fan':
        return list(range(48, 68))
    elif landmark_type == 'mediapipe':
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
    """Extract ROI and resize."""
    try:
        x0, y0, x1, y1 = bbox
        
        h, w = img_rgb.shape[:2]
        x0 = max(0, x0)
        y0 = max(0, y0)
        x1 = min(w, x1)
        y1 = min(h, y1)
        
        roi = img_rgb[y0:y1, x0:x1]
        
        if roi.size == 0:
            return None
        
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
        roi_resized = cv2.resize(roi_gray, (output_size, output_size), 
                                  interpolation=cv2.INTER_AREA)
        
        return roi_resized
    except Exception as e:
        print(f"Error extracting ROI: {e}")
        return None


def draw_landmarks_on_frame(img_rgb, landmarks, landmark_type='fan'):
    """Draw landmarks on image."""
    img_with_landmarks = img_rgb.copy()
    
    if landmarks is None:
        return img_with_landmarks
    
    # Get mouth indices
    mouth_indices = get_mouth_landmarks_indices(landmark_type)
    
    # Draw all landmarks (small, purple)
    for i, (x, y) in enumerate(landmarks):
        cv2.circle(img_with_landmarks, (int(x), int(y)), 2, (147, 59, 114), -1)
    
    # Draw mouth landmarks (larger, orange)
    if len(mouth_indices) > 0 and len(landmarks) > max(mouth_indices):
        mouth_points = landmarks[mouth_indices]
        
        # Draw mouth contour
        for i in range(len(mouth_points)):
            pt1 = tuple(mouth_points[i].astype(int))
            pt2 = tuple(mouth_points[(i + 1) % len(mouth_points)].astype(int))
            cv2.line(img_with_landmarks, pt1, pt2, (241, 143, 1), 2)
        
        # Draw mouth points
        for x, y in mouth_points:
            cv2.circle(img_with_landmarks, (int(x), int(y)), 4, (241, 143, 1), -1)
            cv2.circle(img_with_landmarks, (int(x), int(y)), 4, (255, 255, 255), 1)
    
    return img_with_landmarks


def process_image(image_path, output_dir, method='fan', enlarge_factor=1.6, 
                  output_size=88, device='cpu'):
    """Process single image and extract components."""
    
    print(f"\n{'='*70}")
    print(f"Processing: {Path(image_path).name}")
    print(f"{'='*70}")
    
    # Load image
    print("\n📷 Loading image...")
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        print(f"❌ Failed to load: {image_path}")
        return False
    
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    print(f"   Size: {img_rgb.shape[1]}×{img_rgb.shape[0]}")
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save original frame
    original_path = output_dir / "1_original_frame.png"
    cv2.imwrite(str(original_path), img_bgr)
    print(f"\n✅ Saved: {original_path}")
    
    # Detect landmarks
    print("\n📍 Detecting landmarks...")
    landmarks = None
    landmark_type = 'fan'
    
    if 'fan' in method.lower() and _FAN_OK:
        print(f"   Trying FAN (device: {device})...")
        landmarks = detect_landmarks_fan(img_rgb, device=device)
        if landmarks is not None:
            print(f"   ✓ FAN detected {len(landmarks)} landmarks")
            landmark_type = 'fan'
    
    if landmarks is None and 'mediapipe' in method.lower() and _MP_OK:
        print(f"   Trying MediaPipe...")
        landmarks = detect_landmarks_mediapipe(img_rgb)
        if landmarks is not None:
            print(f"   ✓ MediaPipe detected {len(landmarks)} landmarks")
            landmark_type = 'mediapipe'
    
    if landmarks is None:
        print(f"   ❌ Landmark detection failed")
        return False
    
    # Draw landmarks on frame
    print("\n🎨 Drawing landmarks...")
    img_with_landmarks = draw_landmarks_on_frame(img_rgb, landmarks, landmark_type)
    landmarks_path = output_dir / "2_landmarks_detected.png"
    img_with_landmarks_bgr = cv2.cvtColor(img_with_landmarks, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(landmarks_path), img_with_landmarks_bgr)
    print(f"✅ Saved: {landmarks_path}")
    
    # Compute mouth ROI
    print("\n📏 Computing mouth ROI...")
    mouth_indices = get_mouth_landmarks_indices(landmark_type)
    mouth_bbox = compute_mouth_bbox(landmarks, mouth_indices, enlarge_factor)
    
    if mouth_bbox is None:
        print(f"   ❌ Failed to compute mouth ROI")
        return False
    
    x0, y0, x1, y1 = mouth_bbox
    print(f"   ROI: ({x0}, {y0}) → ({x1}, {y1})")
    print(f"   Size: {x1-x0}×{y1-y0} pixels")
    print(f"   Enlarge factor: {enlarge_factor}×")
    
    # Extract and resize ROI
    print("\n✂️  Extracting final crop...")
    final_crop = extract_and_resize_roi(img_rgb, mouth_bbox, output_size)
    
    if final_crop is None:
        print(f"   ❌ Failed to extract ROI")
        return False
    
    crop_path = output_dir / "3_final_crop.png"
    cv2.imwrite(str(crop_path), final_crop)
    print(f"✅ Saved: {crop_path}")
    print(f"   Size: {final_crop.shape[1]}×{final_crop.shape[0]}")
    print(f"   Value range: [{final_crop.min()}, {final_crop.max()}]")
    
    # Save metadata
    metadata = {
        'input_image': str(image_path),
        'image_size': {'width': img_rgb.shape[1], 'height': img_rgb.shape[0]},
        'landmark_method': landmark_type,
        'num_landmarks': int(len(landmarks)),
        'num_mouth_landmarks': len(mouth_indices),
        'mouth_bbox': {'x0': int(x0), 'y0': int(y0), 'x1': int(x1), 'y1': int(y1)},
        'enlarge_factor': float(enlarge_factor),
        'output_size': int(output_size),
        'final_crop_stats': {
            'mean': float(final_crop.mean()),
            'std': float(final_crop.std()),
            'min': int(final_crop.min()),
            'max': int(final_crop.max())
        },
        'output_files': {
            'original': str(original_path.name),
            'landmarks': str(landmarks_path.name),
            'crop': str(crop_path.name)
        }
    }
    
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n📄 Saved metadata: {metadata_path}")
    
    print(f"\n{'='*70}")
    print("✅ SUCCESS!")
    print(f"{'='*70}")
    print(f"\nOutput directory: {output_dir}")
    print(f"\nFiles created:")
    print(f"  1. {original_path.name} - Original frame")
    print(f"  2. {landmarks_path.name} - Frame with landmarks")
    print(f"  3. {crop_path.name} - Final {output_size}×{output_size} crop")
    print(f"  4. {metadata_path.name} - Metadata\n")
    
    return True


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Extract ROI components as separate images'
    )
    parser.add_argument('--input', type=str, required=True,
                       help='Input image path')
    parser.add_argument('--output-dir', type=str,
                       default='export/roi_components',
                       help='Output directory (default: export/roi_components)')
    parser.add_argument('--method', type=str,
                       default='fan',
                       choices=['fan', 'mediapipe', 'fan+mediapipe'],
                       help='Detection method (default: fan)')
    parser.add_argument('--enlarge', type=float, default=1.6,
                       help='ROI enlargement factor (default: 1.6)')
    parser.add_argument('--output-size', type=int, default=88,
                       help='Output crop size (default: 88)')
    parser.add_argument('--device', type=str, default='cpu',
                       choices=['cpu', 'cuda'],
                       help='Device for FAN (default: cpu)')
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("ROI COMPONENT EXTRACTION")
    print("="*70)
    
    # Check libraries
    print("\n📦 Available Libraries:")
    print(f"   FAN: {'✓' if _FAN_OK else '✗'}")
    print(f"   MediaPipe: {'✓' if _MP_OK else '✗'}")
    
    if not _FAN_OK and not _MP_OK:
        print("\n❌ Error: No detection library available!")
        return 1
    
    # Check input file
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"\n❌ Error: Input file not found: {args.input}")
        return 1
    
    # Process image
    success = process_image(
        image_path=args.input,
        output_dir=args.output_dir,
        method=args.method,
        enlarge_factor=args.enlarge,
        output_size=args.output_size,
        device=args.device
    )
    
    return 0 if success else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n⚠ Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)





