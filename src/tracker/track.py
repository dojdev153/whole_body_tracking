# src/tracker/track.py
"""
Track â€” individual person track with Honest Three-State Model.

States
------
CONFIRMED  : Face actively seen and matches enrolled DB. High confidence.
TEMPORARY  : Face hidden (turned away, occluded). Body still tracked by YOLOX
             + Kalman.  Body Re-ID keeps the identity alive.  Confidence
             decays exponentially: C(t) = C0 * exp(-Î» * Î”t).
UNKNOWN    : Re-ID similarity too low OR decay timeout exceeded. Identity is
             honestly reported as unknown.

New in this version
-------------------
* body_reid_embedding â€” compact 256-D appearance vector updated every time the
  body is detected (face present or not).
* face_hidden_since   â€” absolute timestamp when face was last seen.
* face_hidden_duration (property) â€” how long (seconds) the face has been hidden,
  computed live.
* confirmed_total_seen_time â€” total wall-clock seconds the face was ever CONFIRMED.
"""
from __future__ import annotations

from enum import Enum
import time
from typing import List, Optional, Tuple
import numpy as np

from src.detectors.face_detector import FaceDetection
from src.tracker.kalman_filter import KalmanBoxTracker


class TrackState(Enum):
    """Honest Three-State Model for identity continuity."""
    CONFIRMED = "CONFIRMED"
    TEMPORARY = "TEMPORARY"
    UNKNOWN   = "UNKNOWN"


class Track:
    def __init__(
        self,
        track_id: int,
        initial_bbox: Tuple[int, int, int, int],
        timestamp: float,
        decay_lambda: float = 0.30,        # slower decay â†’ more graceful hiding
        temporary_timeout: float = 120.0,  # 2 minutes before dropping to UNKNOWN
        confidence_floor: float = 0.10,    # lower floor: body Re-ID keeps it alive
    ):
        self.track_id = track_id
        self.kf = KalmanBoxTracker(initial_bbox)
        self.bbox = initial_bbox
        self.decay_lambda = decay_lambda
        self.temporary_timeout = temporary_timeout
        self.confidence_floor = confidence_floor

        # â”€â”€ State â”€â”€
        self.state: TrackState = TrackState.UNKNOWN
        self.identity: str = f"Person_{track_id}"
        self.confirmed_name: Optional[str] = None
        self.confidence: float = 0.0
        self.initial_temp_confidence: float = 0.0

        # â”€â”€ Timestamps â”€â”€
        self.created_at: float = timestamp
        self.last_seen_time: float = timestamp
        self.last_confirmed_time: float = 0.0
        self.temporary_start_time: float = 0.0
        self.face_hidden_since: float = 0.0   # 0 means face is currently visible
        self._face_currently_hidden: bool = False
        self._confirmed_face_start: float = 0.0  # when current CONFIRMED span started
        self.confirmed_total_seen_time: float = 0.0  # cumulative seconds face confirmed

        # â”€â”€ Face association â”€â”€
        self.last_face_detection: Optional[FaceDetection] = None
        self.last_face_embedding: Optional[np.ndarray] = None
        self.last_face_embed_time: float = 0.0
        self.face_present: bool = False

        # â”€â”€ Body Re-ID â”€â”€
        self.body_reid_embedding: Optional[np.ndarray] = None  # 256-D
        self.body_reid_confidence: float = 0.0  # last re-id match score
        self.clothing_color: Optional[Tuple[int, int, int]] = None  # (B, G, R)
        self.clothing_name: str = ""  # e.g. 'Black', 'Blue'
        self.clothing_locked: bool = False

        # â”€â”€ Tracker internals â”€â”€
        self.age: int = 1
        self.hits: int = 1
        self.time_since_update: int = 0
        self.trajectory: List[Tuple[float, float]] = []

    # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    @property
    def center(self) -> Tuple[float, float]:
        x, y, w, h = self.bbox
        return (float(x + w / 2.0), float(y + h / 2.0))

    @property
    def face_hidden_duration(self) -> float:
        """Live seconds since the face was last seen (0 if face is visible)."""
        if self._face_currently_hidden and self.face_hidden_since > 0:
            return max(0.0, time.time() - self.face_hidden_since)
        return 0.0

    def face_hidden_duration_at(self, ts: float) -> float:
        """Seconds face has been hidden as of timestamp `ts`."""
        if self._face_currently_hidden and self.face_hidden_since > 0:
            return max(0.0, ts - self.face_hidden_since)
        return 0.0

    # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    def predict(self) -> Tuple[int, int, int, int]:
        if self.time_since_update > 1:
            self.kf.x[4:7] *= 0.80  # damp velocity when coasting without new detections
        self.bbox = self.kf.predict()
        self.age += 1
        self.time_since_update += 1
        return self.bbox

    def update_box(self, bbox: Tuple[int, int, int, int]):
        self.kf.update(bbox)
        self.bbox = self.kf.get_bbox()
        self.hits += 1
        self.time_since_update = 0
        self.trajectory.append(self.center)
        if len(self.trajectory) > 90:
            self.trajectory.pop(0)

    def update_body_reid(
        self,
        embedding: np.ndarray,
        color_bgr: Optional[Tuple[int, int, int]] = None,
        color_name: str = "",
    ):
        """Store / blend body Re-ID & clothing descriptor."""
        if color_bgr is not None:
            self.clothing_color = color_bgr
        if color_name:
            self.clothing_name = color_name
        if embedding is None:
            return
        if self.body_reid_embedding is None:
            self.body_reid_embedding = embedding.copy()
        else:
            alpha = 0.15
            blended = (1.0 - alpha) * self.body_reid_embedding + alpha * embedding
            n = np.linalg.norm(blended)
            self.body_reid_embedding = blended / n if n > 1e-6 else blended
    def confirm_identity(
        self,
        name: str,
        confidence: float,
        timestamp: float,
        face_detection: Optional[FaceDetection] = None,
        embedding: Optional[np.ndarray] = None,
    ):
        """Transition to / maintain CONFIRMED state. Face actively verified."""
        # Accumulate time in CONFIRMED before this call
        if self.state == TrackState.CONFIRMED and self._confirmed_face_start > 0:
            self.confirmed_total_seen_time += timestamp - self._confirmed_face_start
        self._confirmed_face_start = timestamp

        self.state = TrackState.CONFIRMED
        self.identity = name
        self.confirmed_name = name
        self.confidence = float(np.clip(confidence, 0.0, 1.0))
        self.last_confirmed_time = timestamp
        self.last_seen_time = timestamp
        self.face_present = True
        self.last_face_detection = face_detection
        self.last_face_embed_time = timestamp
        if embedding is not None:
            self.last_face_embedding = embedding

        # Mark face as visible
        if self._face_currently_hidden:
            self._face_currently_hidden = False
            self.face_hidden_since = 0.0

    def maintain_confirmed(self, timestamp: float, face_detection: Optional[FaceDetection] = None):
        """Maintain active CONFIRMED state without re-running ArcFace embedding."""
        if self.state == TrackState.CONFIRMED:
            self.last_seen_time = timestamp
            self.last_confirmed_time = timestamp
            self.face_present = True
            if face_detection is not None:
                self.last_face_detection = face_detection
            if self._face_currently_hidden:
                self._face_currently_hidden = False
                self.face_hidden_since = 0.0
                self._confirmed_face_start = timestamp

    # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    def handle_face_loss(self, timestamp: float, reid_boost: float = 0.0):
        """
        Called each frame the face is NOT seen on this body.
        If track was confirmed/locked to an identity, maintain that identity and confidence!
        """
        self.face_present = False
        self.last_face_detection = None
        self.last_seen_time = timestamp

        # Start hidden timer if this is the first frame without a face
        if not self._face_currently_hidden:
            self._face_currently_hidden = True
            self.face_hidden_since = timestamp
            if self.state == TrackState.CONFIRMED and self._confirmed_face_start > 0:
                self.confirmed_total_seen_time += timestamp - self._confirmed_face_start
                self._confirmed_face_start = 0.0

        is_locked = (self.confirmed_name is not None)

        # Only truly "lock" a track if it has been confirmed for multiple frames (not a drive-by clothes match)
        is_face_confirmed = is_locked and self.hits > 10

        if self.state == TrackState.CONFIRMED:
            self.state = TrackState.TEMPORARY
            self.initial_temp_confidence = max(self.confidence, 0.95) if is_face_confirmed else self.confidence
            self.temporary_start_time = timestamp

        if self.state == TrackState.TEMPORARY:
            delta_t = max(0.0, timestamp - self.temporary_start_time)

            if is_face_confirmed:
                # LOCKED TARGET (face was confirmed): keep identity name and high confidence
                self.identity = self.confirmed_name
                self.confidence = 0.95
                # Only drop after 15 minutes
                if delta_t > 900.0:
                    self.state = TrackState.UNKNOWN
                    self.identity = f"Unknown_{self.track_id}"
                    self.confidence = 0.0
            else:
                # Not face-confirmed - normal decay
                effective_lambda = self.decay_lambda * (1.0 - 0.80 * reid_boost)
                decayed = self.initial_temp_confidence * np.exp(-effective_lambda * delta_t)
                self.confidence = float(decayed)
                if delta_t > self.temporary_timeout or self.confidence < self.confidence_floor:
                    self.state = TrackState.UNKNOWN
                    self.identity = f"Unknown_{self.track_id}"
                    self.confidence = 0.0

    def handle_reid_identity(self, name: str, reid_sim: float, timestamp: float):
        """
        Called when body/clothing Re-ID matches an enrolled person.
        Instantly maintains or recovers identity based on saved clothes!
        """
        if self.confirmed_name is not None and name == self.confirmed_name:
            self.identity = name
            self.confidence = float(max(0.95, reid_sim))
            self.body_reid_confidence = reid_sim
            self.clothing_locked = True
        elif self.confirmed_name is None and reid_sim >= 0.48:
            # Recognized person by their saved clothes!
            self.confirm_identity(name, reid_sim, timestamp)
            self.body_reid_confidence = reid_sim
            self.clothing_locked = True

    def mark_unknown(self):
        # Only keep locked identity if we were previously confirmed (face was seen).
        # This prevents brand-new unverified tracks from stealing confirmed names.
        if self.confirmed_name is not None and self.hits > 5:
            # Track has been around long enough with confirmed name - keep label
            self.identity = self.confirmed_name
            self.confidence = max(self.confidence, 0.80)
            self.face_present = False
            return
        self.state = TrackState.UNKNOWN
        self.identity = f"Unknown_{self.track_id}"
        self.confidence = 0.0
        self.face_present = False

    def get_temporary_elapsed(self, current_time: float) -> float:
        if self.state == TrackState.TEMPORARY:
            return max(0.0, current_time - self.temporary_start_time)
        return 0.0
