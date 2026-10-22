# src/tracker/kalman_filter.py
from __future__ import annotations

from typing import Tuple
import numpy as np


class KalmanBoxTracker:
    """
    2D Constant Velocity Kalman Filter for tracking bounding boxes:
    State: [xc, yc, s, r, v_xc, v_yc, v_s]^T
    where:
        xc, yc: center coordinates
        s = w * h: box area (scale)
        r = w / h: aspect ratio (constant)
        v_xc, v_yc, v_s: respective velocities
    """

    def __init__(self, bbox: Tuple[int, int, int, int]):
        # State dimension: 7, Measurement dimension: 4
        self.dim_x = 7
        self.dim_z = 4

        # State transition matrix F
        self.F = np.eye(self.dim_x, dtype=np.float32)
        for i in range(3):
            self.F[i, i + 4] = 1.0  # position += velocity * dt (dt = 1)

        # Measurement matrix H
        self.H = np.zeros((self.dim_z, self.dim_x), dtype=np.float32)
        for i in range(self.dim_z):
            self.H[i, i] = 1.0

        # Measurement noise covariance R
        self.R = np.eye(self.dim_z, dtype=np.float32)
        self.R[0, 0] *= 1.0    # x
        self.R[1, 1] *= 1.0    # y
        self.R[2, 2] *= 10.0   # scale
        self.R[3, 3] *= 10.0   # ratio

        # State covariance P
        self.P = np.eye(self.dim_x, dtype=np.float32) * 10.0
        self.P[4:, 4:] *= 100.0  # high uncertainty on initial velocities

        # Process noise covariance Q
        self.Q = np.eye(self.dim_x, dtype=np.float32)
        self.Q[:2, :2] *= 1.0
        self.Q[2, 2] *= 1.0
        self.Q[3, 3] *= 0.01  # aspect ratio changes slowly
        self.Q[4:6, 4:6] *= 1.0
        self.Q[6, 6] *= 0.01

        # Initialize state x
        self.x = np.zeros((self.dim_x, 1), dtype=np.float32)
        z = self._bbox_to_z(bbox)
        self.x[:4] = z

    @staticmethod
    def _bbox_to_z(bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """Convert [x, y, w, h] to [xc, yc, s, r]^T."""
        x, y, w, h = bbox
        w = max(1.0, float(w))
        h = max(1.0, float(h))
        xc = float(x) + w / 2.0
        yc = float(y) + h / 2.0
        s = w * h
        r = w / h
        return np.array([[xc], [yc], [s], [r]], dtype=np.float32)

    @staticmethod
    def _x_to_bbox(x: np.ndarray) -> Tuple[int, int, int, int]:
        """Convert [xc, yc, s, r]^T to [x, y, w, h]."""
        xc = float(x[0, 0])
        yc = float(x[1, 0])
        s = max(1.0, float(x[2, 0]))
        r = max(0.05, min(20.0, float(x[3, 0])))

        w = np.sqrt(s * r)
        h = s / max(1e-6, w)

        bx = int(round(xc - w / 2.0))
        by = int(round(yc - h / 2.0))
        bw = int(round(w))
        bh = int(round(h))
        return (bx, by, bw, bh)

    def predict(self) -> Tuple[int, int, int, int]:
        """Advance state by one time step and return predicted bounding box."""
        if (self.x[6, 0] + self.x[2, 0]) <= 0:
            self.x[6, 0] = 0.0

        self.x = np.dot(self.F, self.x)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        return self._x_to_bbox(self.x)

    def update(self, bbox: Tuple[int, int, int, int]):
        """Update Kalman filter with observed bounding box measurement."""
        z = self._bbox_to_z(bbox)
        y = z - np.dot(self.H, self.x)
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))

        self.x = self.x + np.dot(K, y)
        I = np.eye(self.dim_x, dtype=np.float32)
        self.P = np.dot((I - np.dot(K, self.H)), self.P)

    def get_bbox(self) -> Tuple[int, int, int, int]:
        """Get current estimated bounding box."""
        return self._x_to_bbox(self.x)

    def get_velocity(self) -> Tuple[float, float]:
        """Get current estimated center velocity (vx, vy)."""
        return (float(self.x[4, 0]), float(self.x[5, 0]))
