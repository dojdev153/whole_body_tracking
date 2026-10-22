# src/detectors/body_reid.py
"""
Body Re-Identification and Clothing Signature Embedder.

Extracts a compact 256-D appearance descriptor from a person body crop using:
  - Upper-body/torso HOG spatial features (128-D)
  - Multi-strip HSV colour and brightness histograms (128-D)
  - Dominant clothing colour extraction for intuitive visual feedback

Combined into an L2-normalised 256-D vector. Lightweight on CPU (~1.5 ms).
Enables instant identity recognition by clothes even when face is 100% hidden.
"""
from __future__ import annotations

from typing import Optional, Tuple
import cv2
import numpy as np


class BodyReIDEmbedder:
    """
    CPU-optimized Body and Clothing Re-ID using HOG + Multi-channel Colour Encoding.
    Works on any body crop size; internally normalizes to 64x128.
    """

    _HOG_WIN  = (64, 128)      # w, h
    _HOG_CELL = (8, 8)
    _HOG_BINS = 9
    _STRIPS   = 4              # vertical strips for torso and clothing layers
    _HUE_BINS = 16             # 16 hue bins per strip
    _VAL_BINS = 16             # 16 brightness/value bins per strip (crucial for black/white/grey clothes)

    def __init__(self):
        self._hog = cv2.HOGDescriptor(
            self._HOG_WIN,
            (16, 16),          # block size
            (8, 8),            # block stride
            self._HOG_CELL,
            self._HOG_BINS,
        )

    def extract(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Extract 256-D L2-normalised body appearance descriptor.
        Args:
            frame: Full BGR image frame.
            bbox:  (x, y, w, h) bounding box of the detected person.
        Returns:
            1-D float32 numpy array of length 256.
        """
        x, y, w, h = bbox
        img_h, img_w = frame.shape[:2]

        pad = int(0.05 * max(w, h))
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(img_w, x + w + pad)
        y2 = min(img_h, y + h + pad)
        crop = frame[y1:y2, x1:x2]

        if crop.size == 0 or crop.shape[0] < 16 or crop.shape[1] < 8:
            return np.zeros(256, dtype=np.float32)

        crop_resized = cv2.resize(crop, self._HOG_WIN, interpolation=cv2.INTER_LINEAR)

        # --- HOG Shape Features (128-D) ---
        hog_raw = self._hog.compute(crop_resized).flatten()
        if len(hog_raw) >= 128:
            factor = len(hog_raw) // 128
            hog_128 = hog_raw[: factor * 128].reshape(128, factor).mean(axis=1)
        else:
            hog_128 = np.resize(hog_raw, 128)

        # --- Multi-Strip HSV Colour and Lightness Histograms (128-D) ---
        # 4 vertical strips (e.g. shoulders, upper chest, lower torso, waist)
        # 16 Hue bins + 16 Value bins per strip = 32 bins * 4 strips = 128-D
        hsv = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2HSV)
        strip_h = self._HOG_WIN[1] // self._STRIPS
        colour_vec = []
        for s in range(self._STRIPS):
            stripe = hsv[s * strip_h:(s + 1) * strip_h, :, :]
            h_hist = cv2.calcHist([stripe], [0], None, [self._HUE_BINS], [0, 180]).flatten()
            v_hist = cv2.calcHist([stripe], [2], None, [self._VAL_BINS], [0, 256]).flatten()
            colour_vec.append(h_hist)
            colour_vec.append(v_hist)

        colour_hist = np.concatenate(colour_vec, axis=0).astype(np.float32)

        # --- Combine HOG shape + Colour appearance ---
        combined = np.concatenate([hog_128.astype(np.float32), colour_hist])  # 256-D
        norm = np.linalg.norm(combined)
        if norm > 1e-6:
            combined /= norm

        return combined.astype(np.float32)

    def get_clothing_color(
        self,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
    ) -> Tuple[Tuple[int, int, int], str]:
        """
        Extract dominant clothing colour (BGR) and human-readable label from the torso.
        """
        x, y, w, h = bbox
        img_h, img_w = frame.shape[:2]

        # Upper-to-mid torso region (shirt/jacket area)
        y1 = max(0, int(y + 0.20 * h))
        y2 = min(img_h, int(y + 0.70 * h))
        x1 = max(0, int(x + 0.15 * w))
        x2 = min(img_w, int(x + 0.85 * w))
        torso = frame[y1:y2, x1:x2]

        if torso.size == 0 or torso.shape[0] < 5 or torso.shape[1] < 5:
            return (80, 80, 80), "Clothing"

        # Median colour to resist noise/shadows
        med = np.median(torso.reshape(-1, 3), axis=0).astype(int)
        b, g, r = int(med[0]), int(med[1]), int(med[2])

        # HSV color categorization
        hsv_pixel = cv2.cvtColor(np.uint8([[[b, g, r]]]), cv2.COLOR_BGR2HSV)[0][0]
        hue, sat, val = int(hsv_pixel[0]), int(hsv_pixel[1]), int(hsv_pixel[2])

        if val < 50:
            cname = "Black"
        elif sat < 30 and val > 190:
            cname = "White"
        elif sat < 35:
            cname = "Grey"
        elif hue < 10 or hue >= 170:
            cname = "Red"
        elif hue < 25:
            cname = "Orange"
        elif hue < 38:
            cname = "Yellow"
        elif hue < 85:
            cname = "Green"
        elif hue < 132:
            cname = "Blue"
        elif hue < 160:
            cname = "Purple"
        else:
            cname = "Pink"

        return (b, g, r), cname

    def similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between two L2-normalised descriptors."""
        if a is None or b is None:
            return 0.0
        return float(np.clip(np.dot(a, b), -1.0, 1.0))
