"""
Compatibility wrapper for ibug.face_alignment.FANPredictor using the
`face-alignment` package (FAN 2D landmarks).

This provides a minimal drop-in API:

    predictor = FANPredictor(device='cuda'|'cpu', model=None)
    landmarks, scores = predictor(image, detected_faces=None, rgb=True)

Where:
  - image: numpy.ndarray of shape (H, W, 3), RGB
  - landmarks: list of (68, 2) numpy arrays
  - scores: list of floats (dummy 1.0 values)

If you need GPU support, ensure PyTorch with CUDA is installed and pass device='cuda'.
"""
from __future__ import annotations

from typing import Any, List, Tuple

import numpy as np


class FANPredictor:  # type: ignore
    def __init__(self, device: str = 'cpu', model: Any | None = None):
        try:
            import face_alignment  # type: ignore
            from face_alignment import FaceAlignment, LandmarksType  # type: ignore
        except Exception as e:
            raise ImportError(
                "face-alignment is required for FANPredictor wrapper.\n"
                "Install with: pip install face-alignment"
            ) from e

        # Map device string to expected format by face_alignment
        dev = 'cuda' if device.startswith('cuda') else 'cpu'
        # Initialize FaceAlignment with 2D landmarks (enum name varies across versions)
        ltype = None
        for name in ('_2D', 'TWO_D', 'POINTS_2D', 'L2D', 'LANDMARKS_2D'):
            ltype = getattr(LandmarksType, name, None)
            if ltype is not None:
                break
        if ltype is None:
            # Fallback: try accessing via module attributes
            ltype = getattr(face_alignment, 'LandmarksType', None)
            ltype = getattr(ltype, '_2D', None)
        if ltype is None:
            raise RuntimeError("Could not find a 2D LandmarksType enum in face-alignment")

        self._fa = FaceAlignment(
            ltype,
            device=dev,
            flip_input=False,
        )

    def __call__(self, image: np.ndarray, detected_faces: Any | None = None, rgb: bool = True) -> Tuple[List[np.ndarray], List[float]]:
        """
        Predict 68-point landmarks for faces in the image.
        Returns (landmarks_list, scores_list).

        If detected_faces are provided (from an external detector like RetinaFace),
        they will be used to guide the landmark detection for improved robustness.
        Supported formats include:
          - numpy array of shape (N,4) or (N,5) where [:4] are [x1,y1,x2,y2]
          - list of [x1,y1,x2,y2]
          - list of objects with attributes .bbox (x1,y1,x2,y2) or .x1/.y1/.x2/.y2
        """
        if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Expected image ndarray HxWx3, got {type(image)} shape={getattr(image, 'shape', None)}")

        H, W = image.shape[:2]

        # Normalize detected face boxes to a list of [x1,y1,x2,y2] in pixel coords
        boxes: List[List[int]] | None = None
        if detected_faces is not None:
            boxes = []
            def clamp_box(x0, y0, x1, y1):
                x0 = max(0, int(round(x0))); y0 = max(0, int(round(y0)))
                x1 = min(W - 1, int(round(x1))); y1 = min(H - 1, int(round(y1)))
                if x1 > x0 and y1 > y0:
                    boxes.append([x0, y0, x1, y1])

            try:
                import numpy as _np
                if isinstance(detected_faces, _np.ndarray) and detected_faces.ndim == 2 and detected_faces.shape[1] >= 4:
                    for r in detected_faces:
                        x0, y0, x1, y1 = float(r[0]), float(r[1]), float(r[2]), float(r[3])
                        clamp_box(x0, y0, x1, y1)
                elif isinstance(detected_faces, (list, tuple)):
                    for f in detected_faces:
                        if isinstance(f, (list, tuple)) and len(f) >= 4:
                            x0, y0, x1, y1 = float(f[0]), float(f[1]), float(f[2]), float(f[3])
                            clamp_box(x0, y0, x1, y1)
                        else:
                            # Object with attributes
                            x0 = getattr(f, 'x1', None); y0 = getattr(f, 'y1', None)
                            x1 = getattr(f, 'x2', None); y1 = getattr(f, 'y2', None)
                            if x0 is None or y0 is None or x1 is None or y1 is None:
                                b = getattr(f, 'bbox', None)
                                if b is not None and len(b) >= 4:
                                    x0, y0, x1, y1 = float(b[0]), float(b[1]), float(b[2]), float(b[3])
                                else:
                                    continue
                            clamp_box(x0, y0, x1, y1)
                # Else: ignore unknown formats
            except Exception:
                boxes = []

            if boxes is not None and len(boxes) == 0:
                boxes = None

        # face_alignment expects RGB; if input is BGR, caller should convert
        if boxes is not None:
            lms_list = self._fa.get_landmarks_from_image(image, detected_faces=boxes)
        else:
            lms_list = self._fa.get_landmarks_from_image(image)

        if lms_list is None:
            return [], []
        # Ensure numpy arrays of shape (68, 2)
        out_lms: List[np.ndarray] = []
        for lm in lms_list:
            arr = np.array(lm, dtype=np.float32)
            if arr.ndim != 2 or arr.shape[1] != 2:
                continue
            out_lms.append(arr)
        scores = [1.0] * len(out_lms)
        return out_lms, scores
