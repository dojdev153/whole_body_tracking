# src/control/target_centering.py
from __future__ import annotations

from dataclasses import dataclass
import time
from typing import List, Optional, Tuple
import numpy as np

from src.tracker.track import Track, TrackState


@dataclass
class CenteringTelemetry:
    """Telemetry data output by the target centering controller."""
    has_target: bool
    target_id: Optional[int]
    target_identity: Optional[str]
    target_state: Optional[TrackState]
    target_point: Optional[Tuple[float, float]]  # (x, y) in frame coords
    pan_error: float   # Normalized [-1.0 (left), +1.0 (right)]
    tilt_error: float  # Normalized [-1.0 (up), +1.0 (down)]
    is_centered: bool  # True if inside deadband
    simulated_pan_angle: float   # Degrees (-90° to +90°)
    simulated_tilt_angle: float  # Degrees (-45° to +45°)
    control_pan_output: float    # Commanded angular velocity (deg/s)
    control_tilt_output: float


class TargetCenteringController:
    """
    Robotics Target Centering & Visual Servoing Controller.
    Computes normalized pan/tilt tracking error and simulates a physical PTZ gimbal
    without requiring physical motor hardware.
    """

    def __init__(
        self,
        deadband: float = 0.05,        # 5% normalized deadband to prevent mechanical jitter
        kp: float = 45.0,              # Proportional gain (deg/s per unit error)
        kd: float = 12.0,              # Derivative gain
        max_pan_angle: float = 90.0,   # Gimbal pan limit (+/- deg)
        max_tilt_angle: float = 45.0,  # Gimbal tilt limit (+/- deg)
        target_name: Optional[str] = None,  # Preferred target identity (e.g. "Duff")
    ):
        self.deadband = deadband
        self.kp = kp
        self.kd = kd
        self.max_pan_angle = max_pan_angle
        self.max_tilt_angle = max_tilt_angle
        self.target_name = target_name

        # State
        self.sim_pan: float = 0.0
        self.sim_tilt: float = 0.0
        self.prev_error_x: float = 0.0
        self.prev_error_y: float = 0.0
        self.last_update_time: float = time.time()
        self.selected_target_id: Optional[int] = None

    def select_target(self, tracks: List[Track]) -> Optional[Track]:
        """
        Target selection hierarchy:
        1. Manually selected target ID (if still alive)
        2. Preferred identity name (e.g. "Duff") in CONFIRMED or TEMPORARY state
        3. Any CONFIRMED track (highest confidence)
        4. Any TEMPORARY track (maintains visual servoing across temporary facial loss!)
        5. Largest UNKNOWN track
        """
        if not tracks:
            return None

        # 1. Manually selected target
        if self.selected_target_id is not None:
            for t in tracks:
                if t.track_id == self.selected_target_id:
                    return t

        # 2. Preferred identity name
        if self.target_name:
            # Check confirmed first
            for t in tracks:
                if t.state == TrackState.CONFIRMED and t.identity.lower() == self.target_name.lower():
                    return t
            # Check temporary
            for t in tracks:
                if t.state == TrackState.TEMPORARY and t.identity.lower() == self.target_name.lower():
                    return t

        # 3. Any CONFIRMED track
        confirmed = [t for t in tracks if t.state == TrackState.CONFIRMED]
        if confirmed:
            return max(confirmed, key=lambda t: t.confidence)

        # 4. Any TEMPORARY track (crucial: maintain centering during temporary head turn/occlusion!)
        temporary = [t for t in tracks if t.state == TrackState.TEMPORARY]
        if temporary:
            return max(temporary, key=lambda t: t.confidence)

        # 5. Fallback: Largest unknown track by area
        return max(tracks, key=lambda t: t.bbox[2] * t.bbox[3])

    def compute(
        self,
        tracks: List[Track],
        frame_width: int,
        frame_height: int,
        timestamp: Optional[float] = None,
    ) -> CenteringTelemetry:
        """
        Calculate target centering error and update simulated PTZ gimbal telemetry.
        """
        curr_time = timestamp if timestamp is not None else time.time()
        dt = max(1e-3, curr_time - self.last_update_time)
        self.last_update_time = curr_time

        target = self.select_target(tracks)

        if target is None:
            # No target available
            self.prev_error_x = 0.0
            self.prev_error_y = 0.0
            return CenteringTelemetry(
                has_target=False,
                target_id=None,
                target_identity=None,
                target_state=None,
                target_point=None,
                pan_error=0.0,
                tilt_error=0.0,
                is_centered=False,
                simulated_pan_angle=self.sim_pan,
                simulated_tilt_angle=self.sim_tilt,
                control_pan_output=0.0,
                control_tilt_output=0.0,
            )

        # Optical center of image
        cx_opt = frame_width / 2.0
        cy_opt = frame_height / 2.0

        # Target center point: use upper chest / head position for natural centering
        bx, by, bw, bh = target.bbox
        tx = float(bx + bw / 2.0)
        ty = float(by + 0.30 * bh)  # 30% from top of body box (head/upper chest)

        # Normalized errors in [-1.0, 1.0]
        raw_ex = (tx - cx_opt) / cx_opt
        raw_ey = (ty - cy_opt) / cy_opt

        # Apply deadband
        deadband_ex = 0.0 if abs(raw_ex) < self.deadband else raw_ex
        deadband_ey = 0.0 if abs(raw_ey) < self.deadband else raw_ey

        is_centered = (abs(raw_ex) < self.deadband) and (abs(raw_ey) < self.deadband)

        # Derivative calculation
        d_ex = (deadband_ex - self.prev_error_x) / dt
        d_ey = (deadband_ey - self.prev_error_y) / dt
        self.prev_error_x = deadband_ex
        self.prev_error_y = deadband_ey

        # Control commands (deg/s)
        u_pan = self.kp * deadband_ex + self.kd * d_ex
        u_tilt = self.kp * deadband_ey + self.kd * d_ey

        # Integrate simulated PTZ gimbal angles
        self.sim_pan = float(np.clip(self.sim_pan + u_pan * dt, -self.max_pan_angle, self.max_pan_angle))
        self.sim_tilt = float(np.clip(self.sim_tilt + u_tilt * dt, -self.max_tilt_angle, self.max_tilt_angle))

        return CenteringTelemetry(
            has_target=True,
            target_id=target.track_id,
            target_identity=target.identity,
            target_state=target.state,
            target_point=(tx, ty),
            pan_error=raw_ex,
            tilt_error=raw_ey,
            is_centered=is_centered,
            simulated_pan_angle=self.sim_pan,
            simulated_tilt_angle=self.sim_tilt,
            control_pan_output=u_pan,
            control_tilt_output=u_tilt,
        )

    def reset_gimbal(self):
        """Reset simulated gimbal angles to home (0, 0)."""
        self.sim_pan = 0.0
        self.sim_tilt = 0.0
        self.prev_error_x = 0.0
        self.prev_error_y = 0.0
