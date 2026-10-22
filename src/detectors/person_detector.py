# src/detectors/person_detector.py
"""
YOLOX-Nano CPU-only Person Detector with HOG fallback.
Output format: [x_center, y_center, width, height, obj_conf, cls_conf x 80]
YOLOX decoding: obj_score = sigmoid(pred[4]), class_score = sigmoid(pred[5:])
Final score = obj_score * class_score[0]  (COCO class 0 = person)
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import cv2
import numpy as np
import onnxruntime as ort


@dataclass
class PersonDetection:
    """Represents a detected person body bounding box and confidence score."""
    bbox: Tuple[int, int, int, int]  # (x, y, w, h) — pixel coords, clamped
    score: float
    landmarks: Optional[np.ndarray] = None
    center: Tuple[float, float] = field(default=(0.0, 0.0))

    def __post_init__(self):
        if self.center == (0.0, 0.0):
            x, y, w, h = self.bbox
            object.__setattr__(self, "center", (float(x + w / 2.0), float(y + h / 2.0)))

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


# YOLOX-Nano strides and grid anchor tables (pre-built for input 416x416)
_STRIDES = [8, 16, 32]


def _make_grids(input_h: int = 416, input_w: int = 416):
    """Build flat anchor grid arrays for YOLOX detection decoding."""
    grids, strides_arr = [], []
    for stride in _STRIDES:
        gh, gw = input_h // stride, input_w // stride
        yv, xv = np.meshgrid(np.arange(gh), np.arange(gw), indexing="ij")
        grid = np.stack([xv, yv], axis=-1).reshape(-1, 2).astype(np.float32)
        grids.append(grid)
        strides_arr.append(np.full((gh * gw, 1), stride, dtype=np.float32))
    return np.concatenate(grids, axis=0), np.concatenate(strides_arr, axis=0)


_GRIDS, _STRIDE_ARR = _make_grids(416, 416)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -88.0, 88.0)))


class PersonDetector:
    """
    CPU-optimised YOLOX-Nano Person Detector via ONNXRuntime.
    Falls back to OpenCV HOG if the ONNX model is not found.
    """

    INPUT_SIZE = 416     # YOLOX-Nano is trained at 416x416
    PERSON_CLASS = 0     # COCO label 0 = person

    def __init__(
        self,
        model_path: str = "models/yolox_nano.onnx",
        score_threshold: float = 0.35,
        nms_threshold: float = 0.45,
        top_k: int = 50,
    ):
        self.model_path = model_path
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.top_k = top_k
        self._use_onnx = False

        if os.path.exists(model_path):
            try:
                opts = ort.SessionOptions()
                opts.intra_op_num_threads = 2
                opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                self._sess = ort.InferenceSession(
                    model_path, sess_options=opts,
                    providers=["CPUExecutionProvider"],
                )
                self._input_name = self._sess.get_inputs()[0].name
                self._use_onnx = True
                print(f"[PersonDetector] YOLOX-Nano loaded from '{model_path}'")
            except Exception as e:
                print(f"[PersonDetector] YOLOX load error: {e}. Falling back to HOG.")

        if not self._use_onnx:
            self._hog = cv2.HOGDescriptor()
            self._hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
            print("[PersonDetector] Using HOG fallback.")

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------
    def _preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, float, float, float]:
        """
        Letterbox-resize the frame to 416x416.
        Returns blob (1,3,416,416), ratio, pad_x, pad_y.
        """
        h, w = image.shape[:2]
        ratio = min(self.INPUT_SIZE / h, self.INPUT_SIZE / w)
        new_w, new_h = int(w * ratio), int(h * ratio)
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        pad_x = (self.INPUT_SIZE - new_w) / 2.0
        pad_y = (self.INPUT_SIZE - new_h) / 2.0
        canvas = np.full((self.INPUT_SIZE, self.INPUT_SIZE, 3), 114, dtype=np.uint8)
        canvas[int(pad_y):int(pad_y) + new_h, int(pad_x):int(pad_x) + new_w] = resized
        # BGR -> RGB, HWC -> CHW, normalise
        rgb = canvas[:, :, ::-1].astype(np.float32)
        blob = np.transpose(rgb, (2, 0, 1))[np.newaxis, ...]   # 1,3,416,416
        return blob, ratio, pad_x, pad_y

    # ------------------------------------------------------------------
    # Postprocessing
    # ------------------------------------------------------------------
    def _postprocess(
        self,
        raw: np.ndarray,          # shape (1, 3549, 85)
        orig_w: int,
        orig_h: int,
        ratio: float,
        pad_x: float,
        pad_y: float,
    ) -> List[PersonDetection]:
        pred = raw[0]   # (3549, 85)

        # YOLOX ONNX export already contains sigmoid-activated probabilities
        obj = pred[:, 4]
        cls_person = pred[:, 5 + self.PERSON_CLASS]
        person_score = obj * cls_person

        # Score threshold gate FIRST - avoid decoding thousands of non-person boxes
        mask = person_score >= self.score_threshold
        if not np.any(mask):
            return []

        person_score = person_score[mask]
        pred_passing = pred[mask]
        strides_passing = _STRIDE_ARR[mask]
        grids_passing = _grids(pred)[mask]

        # Decode box centres and dimensions only for passing boxes
        box_xy = (grids_passing + pred_passing[:, :2]) * strides_passing
        box_wh = np.exp(pred_passing[:, 2:4]) * strides_passing

        # Convert YOLOX [cx, cy, w, h] (in 416-space) to [x1, y1, x2, y2] orig-image space
        x1_416 = box_xy[:, 0] - box_wh[:, 0] / 2.0
        y1_416 = box_xy[:, 1] - box_wh[:, 1] / 2.0
        x2_416 = box_xy[:, 0] + box_wh[:, 0] / 2.0
        y2_416 = box_xy[:, 1] + box_wh[:, 1] / 2.0

        # Remove letterbox padding and rescale to original frame
        x1 = np.clip((x1_416 - pad_x) / ratio, 0, orig_w - 1)
        y1 = np.clip((y1_416 - pad_y) / ratio, 0, orig_h - 1)
        x2 = np.clip((x2_416 - pad_x) / ratio, 0, orig_w - 1)
        y2 = np.clip((y2_416 - pad_y) / ratio, 0, orig_h - 1)

        bw = x2 - x1
        bh = y2 - y1

        # NMS
        boxes_nms = np.stack([x1, y1, bw, bh], axis=1)
        indices = cv2.dnn.NMSBoxes(
            boxes_nms.tolist(),
            person_score.tolist(),
            self.score_threshold,
            self.nms_threshold,
        )
        if len(indices) == 0:
            return []

        detections: List[PersonDetection] = []
        for idx in indices:
            i = int(idx)
            ibx, iby, ibw, ibh = int(x1[i]), int(y1[i]), int(bw[i]), int(bh[i])
            if ibw < 20 or ibh < 30:
                continue
            detections.append(
                PersonDetection(
                    bbox=(ibx, iby, ibw, ibh),
                    score=float(person_score[i]),
                )
            )
        return detections

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def detect(self, frame: np.ndarray) -> List[PersonDetection]:
        """Detect persons in a BGR frame. Returns list of PersonDetection."""
        if frame is None or frame.size == 0:
            return []
        orig_h, orig_w = frame.shape[:2]

        if self._use_onnx:
            blob, ratio, pad_x, pad_y = self._preprocess(frame)
            raw = self._sess.run(None, {self._input_name: blob})[0]
            return self._postprocess(raw, orig_w, orig_h, ratio, pad_x, pad_y)

        # HOG fallback
        rects, weights = self._hog.detectMultiScale(
            frame, winStride=(8, 8), padding=(4, 4), scale=1.05,
        )
        return [
            PersonDetection(bbox=(int(rx), int(ry), int(rw), int(rh)), score=float(w))
            for (rx, ry, rw, rh), w in zip(rects, weights)
            if w >= self.score_threshold
        ]


# ----- helpers (module-level grids for efficiency) --------------------
def _grids(pred: np.ndarray) -> np.ndarray:
    """Return the pre-built grid matching pred rows."""
    n = pred.shape[0]
    return _GRIDS[:n]
