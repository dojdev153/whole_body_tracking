# src/detectors/face_embedder.py
from __future__ import annotations

import os
from typing import Optional, Tuple
import cv2
import numpy as np
import onnxruntime as ort

from src.detectors.face_detector import FaceDetection

# Standard 112x112 ArcFace facial landmarks reference
ARCFACE_REFERENCE_POINTS = np.array(
    [
        [38.2946, 51.6963],  # right eye
        [73.5318, 51.5014],  # left eye
        [56.0252, 71.7366],  # nose
        [41.5493, 92.3655],  # right mouth corner
        [70.7299, 92.2041],  # left mouth corner
    ],
    dtype=np.float32,
)


class FaceEmbedder:
    """
    ArcFace ONNX 512-D Face Feature Extractor running on CPU via ONNXRuntime.
    Performs similarity alignment using 5 facial landmarks before embedding.
    """

    def __init__(self, model_path: str = "models/embedder_arcface.onnx"):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"ArcFace ONNX model not found at '{model_path}'")

        self.model_path = model_path
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(
            model_path,
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def align_face(self, frame: np.ndarray, landmarks: Optional[np.ndarray], bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """Align face using 5-point similarity transformation, or square crop fallback."""
        x, y, w, h = bbox
        if landmarks is not None and len(landmarks) == 5:
            # Estimate partial affine (similarity transform: translation, rotation, uniform scale)
            M, _ = cv2.estimateAffinePartial2D(landmarks.astype(np.float32), ARCFACE_REFERENCE_POINTS)
            if M is not None:
                aligned = cv2.warpAffine(frame, M, (112, 112), borderValue=0.0)
                return aligned

        # Fallback: square crop around bounding box
        img_h, img_w = frame.shape[:2]
        cx, cy = x + w / 2.0, y + h / 2.0
        side = max(w, h) * 1.2
        x1 = int(max(0, cx - side / 2.0))
        y1 = int(max(0, cy - side / 2.0))
        x2 = int(min(img_w, cx + side / 2.0))
        y2 = int(min(img_h, cy + side / 2.0))

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return np.zeros((112, 112, 3), dtype=np.uint8)

        return cv2.resize(crop, (112, 112), interpolation=cv2.INTER_LINEAR)

    def extract_embedding(self, frame: np.ndarray, face: FaceDetection) -> np.ndarray:
        """
        Extract L2-normalized 512-D embedding from a detected face.
        Returns a 1D numpy array of shape (512,), normalized to unit length.
        """
        aligned_bgr = self.align_face(frame, face.landmarks, face.bbox)
        rgb = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB)

        # Normalize to [-1, 1]
        normed = (rgb.astype(np.float32) - 127.5) / 128.0
        blob = np.transpose(normed, (2, 0, 1))[np.newaxis, ...]  # 1, 3, 112, 112

        output = self.session.run([self.output_name], {self.input_name: blob})[0]
        raw_feat = output[0]  # shape (512,)

        # L2 normalize
        norm = np.linalg.norm(raw_feat)
        if norm > 1e-6:
            feat = raw_feat / norm
        else:
            feat = raw_feat

        return feat.astype(np.float32)
