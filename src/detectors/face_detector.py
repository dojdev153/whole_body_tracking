# src/detectors/face_detector.py
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional, Tuple
import cv2
import numpy as np


@dataclass
class FaceDetection:
    """Represents a detected face with bounding box, score, and 5 facial landmarks."""
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    score: float
    landmarks: np.ndarray  # shape (5, 2): [right_eye, left_eye, nose, right_mouth, left_mouth]
    center: Tuple[float, float]

    @property
    def x(self) -> int:
        return self.bbox[0]

    @property
    def y(self) -> int:
        return self.bbox[1]

    @property
    def w(self) -> int:
        return self.bbox[2]

    @property
    def h(self) -> int:
        return self.bbox[3]

    @property
    def area(self) -> float:
        return float(self.w * self.h)


class FaceDetector:
    """
    High-speed CPU Face Detector using OpenCV YuNet ONNX model.
    Detects faces and extracts 5 facial landmarks for alignment.
    """

    def __init__(
        self,
        model_path: str = "models/face_detection_yunet_2023mar.onnx",
        score_threshold: float = 0.6,
        nms_threshold: float = 0.3,
        top_k: int = 5000,
    ):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"YuNet model not found at '{model_path}'")

        self.model_path = model_path
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.top_k = top_k
        self._current_size = (320, 320)

        self.detector = cv2.FaceDetectorYN.create(
            model_path,
            "",
            self._current_size,
            self.score_threshold,
            self.nms_threshold,
            self.top_k,
        )

    def detect(self, frame: np.ndarray) -> List[FaceDetection]:
        """
        Detect faces in a BGR frame.
        Returns a list of FaceDetection dataclass instances.
        """
        if frame is None or frame.size == 0:
            return []

        h, w = frame.shape[:2]
        if (w, h) != self._current_size:
            self.detector.setInputSize((w, h))
            self._current_size = (w, h)

        _, faces = self.detector.detect(frame)
        if faces is None or len(faces) == 0:
            return []

        detections: List[FaceDetection] = []
        for face in faces:
            # Face format: [x, y, w, h, x_re, y_re, x_le, y_le, x_nt, y_nt, x_rc, y_rc, x_lc, y_lc, score]
            x, y, bw, bh = int(face[0]), int(face[1]), int(face[2]), int(face[3])
            # Clamp to frame bounds
            x = max(0, x)
            y = max(0, y)
            bw = min(w - x, bw)
            bh = min(h - y, bh)

            if bw <= 0 or bh <= 0:
                continue

            score = float(face[14])
            landmarks = np.array(
                [
                    [face[4], face[5]],   # right eye
                    [face[6], face[7]],   # left eye
                    [face[8], face[9]],   # nose tip
                    [face[10], face[11]], # right mouth corner
                    [face[12], face[13]], # left mouth corner
                ],
                dtype=np.float32,
            )

            center = (float(x + bw / 2.0), float(y + bh / 2.0))
            detections.append(
                FaceDetection(
                    bbox=(x, y, bw, bh),
                    score=score,
                    landmarks=landmarks,
                    center=center,
                )
            )

        return detections
