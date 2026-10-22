# src/visualization/ui_renderer.py
"""
Aesthetic Robotics HUD Renderer.

New in this version
-------------------
* Per-track identity card with large name, live confidence %, and a face-hidden
  timer displayed as  "HIDDEN: 0m 08s"  (switches to minutes once â‰¥60 s).
* Animated confidence arc (OpenCV ellipse) drawn inside the body bbox.
* Re-ID status badge shown when Re-ID is actively keeping identity alive.
* Three-state colour coding stays: Confirmed=green, Temporary=amber, Unknown=red.
"""
from __future__ import annotations

import math
import time
from typing import List, Optional, Tuple
import cv2
import numpy as np

from src.control.target_centering import CenteringTelemetry
from src.tracker.track import Track, TrackState

# â”€â”€ Colour palette (BGR) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
COLOR_CONFIRMED = (50, 220, 80)      # emerald green
COLOR_TEMPORARY = (0, 170, 255)      # amber/gold
COLOR_UNKNOWN   = (80, 80, 220)      # crimson-rose
COLOR_REID      = (200, 80, 240)     # violet â€” body Re-ID active
COLOR_CYAN      = (230, 200, 0)      # tech cyan
COLOR_DARK_BG   = (18, 18, 24)
COLOR_WHITE     = (245, 245, 250)
COLOR_GRAY      = (130, 130, 140)

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_FONT_BOLD = cv2.FONT_HERSHEY_DUPLEX


def _fmt_duration(seconds: float) -> str:
    """Format a duration (seconds) as  '3m 14s'  or  '47s'."""
    s = int(seconds)
    if s >= 60:
        return f"{s // 60}m {s % 60:02d}s"
    return f"{s}s"


def _draw_rounded_rect(img, pt1, pt2, color, radius=8, alpha=0.72):
    """Draw a semi-transparent filled rounded rectangle using fast ROI blending."""
    x1, y1 = max(0, int(pt1[0])), max(0, int(pt1[1]))
    x2, y2 = min(img.shape[1], int(pt2[0])), min(img.shape[0], int(pt2[1]))
    if x2 <= x1 or y2 <= y1:
        return
    sub = img[y1:y2, x1:x2]
    overlay = sub.copy()
    w, h = x2 - x1, y2 - y1
    r = min(radius, w // 2, h // 2)
    if r > 0:
        cv2.rectangle(overlay, (r, 0), (w - r, h), color, -1)
        cv2.rectangle(overlay, (0, r), (w, h - r), color, -1)
        for cx, cy in [(r, r), (w - r, r), (r, h - r), (w - r, h - r)]:
            cv2.circle(overlay, (cx, cy), r, color, -1)
    else:
        cv2.rectangle(overlay, (0, 0), (w, h), color, -1)
    cv2.addWeighted(overlay, alpha, sub, 1.0 - alpha, 0, sub)


class UIRenderer:
    def __init__(self, show_hud: bool = True, show_trajectories: bool = False):
        self.show_hud = show_hud
        self.show_trajectories = show_trajectories

    # â”€â”€ Corner-bracket bounding box â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    def _corner_box(self, img, bbox, color, thick=2, cl=18):
        x, y, w, h = bbox
        x2, y2 = x + w, y + h
        cv2.rectangle(img, (x, y), (x2, y2), color, 1, cv2.LINE_AA)
        t = thick + 1
        cl = min(cl, w // 3, h // 3)
        for (ax, ay), (bx, by) in [
            ((x, y), (x + cl, y)), ((x, y), (x, y + cl)),
            ((x2, y), (x2 - cl, y)), ((x2, y), (x2, y + cl)),
            ((x, y2), (x + cl, y2)), ((x, y2), (x, y2 - cl)),
            ((x2, y2), (x2 - cl, y2)), ((x2, y2), (x2, y2 - cl)),
        ]:
            cv2.line(img, (ax, ay), (bx, by), color, t, cv2.LINE_AA)

    # â”€â”€ Confidence arc inside box â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    def _conf_arc(self, img, bbox, confidence, color):
        x, y, w, h = bbox
        if w < 30 or h < 30:
            return
        # Small arc in top-right corner of the bbox
        cx, cy = x + w - 20, y + 20
        r = 14
        angle = int(360 * confidence)
        cv2.ellipse(img, (cx, cy), (r, r), -90, 0, 360, (40, 40, 50), 2, cv2.LINE_AA)
        if angle > 0:
            cv2.ellipse(img, (cx, cy), (r, r), -90, 0, angle, color, 2, cv2.LINE_AA)
        pct_str = f"{int(confidence * 100)}"
        (tw, th), _ = cv2.getTextSize(pct_str, _FONT, 0.30, 1)
        cv2.putText(img, pct_str, (cx - tw // 2, cy + th // 2), _FONT, 0.30, color, 1, cv2.LINE_AA)

    # â”€â”€ Per-track identity card â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    def _draw_identity_card(self, img, track: Track, is_target: bool, now: float):
        bx, by, bw, bh = track.bbox
        f_h, f_w = img.shape[:2]

        if track.state == TrackState.CONFIRMED:
            color = COLOR_CONFIRMED
            state_str = "CONFIRMED"
        elif track.state == TrackState.TEMPORARY:
            color = COLOR_TEMPORARY
            state_str = "TEMPORARY"
        else:
            color = COLOR_UNKNOWN
            state_str = "UNKNOWN"

        # â”€â”€ card dimensions â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        card_w = max(bw, 180)
        card_h = 80
        card_x = max(0, min(bx, f_w - card_w - 4))
        card_y = max(0, by - card_h - 4)

        _draw_rounded_rect(img, (card_x, card_y), (card_x + card_w, card_y + card_h),
                           COLOR_DARK_BG, radius=7, alpha=0.80)

        # â”€â”€ State pill â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        pill_color = color
        pill_text  = state_str
        if track.confirmed_name is not None:
            pill_color = COLOR_CONFIRMED
            if not track.face_present:
                pill_text = "LOCKED"
            else:
                pill_text = "CONFIRMED"
        elif track.state == TrackState.TEMPORARY and track.body_reid_confidence >= 0.50:
            pill_color = COLOR_REID
            pill_text  = "RE-ID ACTIVE"

        (pw, ph), _ = cv2.getTextSize(pill_text, _FONT, 0.38, 1)
        pill_x1, pill_y1 = card_x + 6, card_y + 6
        pill_x2, pill_y2 = pill_x1 + pw + 10, pill_y1 + ph + 6
        cv2.rectangle(img, (pill_x1, pill_y1), (pill_x2, pill_y2), pill_color, -1, cv2.LINE_AA)
        cv2.putText(img, pill_text, (pill_x1 + 5, pill_y2 - 4), _FONT, 0.38, (10, 10, 10), 1, cv2.LINE_AA)

        # â”€â”€ Identity name (large) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        name_y = card_y + ph + 24
        cv2.putText(img, track.identity, (card_x + 8, name_y), _FONT_BOLD, 0.55, color, 1, cv2.LINE_AA)

        # â”€â”€ Confidence % â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        conf_str = f"{track.confidence * 100:.0f}%"
        (cw, _), _ = cv2.getTextSize(conf_str, _FONT, 0.45, 1)
        cv2.putText(img, conf_str, (card_x + card_w - cw - 8, name_y),
                    _FONT, 0.45, color, 1, cv2.LINE_AA)

        # â”€â”€ Face-hidden timer (most important new element) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        hidden_s = track.face_hidden_duration_at(now)
        timer_y  = name_y + 20
        if hidden_s > 0.5:
            t_str = f"HIDDEN: {_fmt_duration(hidden_s)}"
            cv2.putText(img, t_str, (card_x + 8, timer_y), _FONT, 0.40,
                        COLOR_TEMPORARY if track.state == TrackState.TEMPORARY else COLOR_GRAY,
                        1, cv2.LINE_AA)
            # Horizontal decay bar
            bar_x, bar_y = card_x + 8, timer_y + 6
            bar_w_total  = card_w - 16
            fill_ratio   = max(0.0, track.confidence)
            cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_w_total, bar_y + 4), (40, 40, 50), -1)
            cv2.rectangle(img, (bar_x, bar_y),
                          (bar_x + int(bar_w_total * fill_ratio), bar_y + 4),
                          color, -1)
        elif track.state == TrackState.CONFIRMED:
            t_str = f"SEEN: {_fmt_duration(track.confirmed_total_seen_time + (now - track._confirmed_face_start if track._confirmed_face_start > 0 else 0))}"
            cv2.putText(img, t_str, (card_x + 8, timer_y), _FONT, 0.40, COLOR_CONFIRMED, 1, cv2.LINE_AA)

        # â”€â”€ TARGET LOCK badge â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        if is_target:
            lock_x = card_x + card_w - 90
            cv2.putText(img, "â—† LOCK", (lock_x, card_y + card_h - 6),
                        _FONT_BOLD, 0.40, color, 1, cv2.LINE_AA)

    # â”€â”€ Main draw-tracks entry point â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    def draw_tracks(
        self,
        frame: np.ndarray,
        tracks: List[Track],
        telemetry: Optional[CenteringTelemetry] = None,
        current_time: Optional[float] = None,
    ):
        now = current_time if current_time is not None else time.time()
        for track in tracks:
            # Skip unknown tracks - only show confirmed and temporary
            if track.state == TrackState.UNKNOWN:
                continue
            if track.state == TrackState.CONFIRMED:
                color = COLOR_CONFIRMED
            elif track.state == TrackState.TEMPORARY:
                # If previously confirmed (locked identity), show in green so user knows
                # "I know who you are, just cant see your face right now"
                if track.confirmed_name is not None:
                    color = COLOR_CONFIRMED  # Green: locked identity, back turned
                else:
                    color = COLOR_REID if track.body_reid_confidence >= 0.50 else COLOR_TEMPORARY
            else:
                color = COLOR_UNKNOWN

            # Trajectory
            if self.show_trajectories and len(track.trajectory) > 1:
                pts = np.array(track.trajectory, dtype=np.int32).reshape(-1, 1, 2)
                cv2.polylines(frame, [pts], False, color, 1, cv2.LINE_AA)

            # Body box
            self._corner_box(frame, track.bbox, color)

            # Confidence arc
            self._conf_arc(frame, track.bbox, track.confidence, color)

            # Identity card
            is_target = bool(
                telemetry and telemetry.has_target and telemetry.target_id == track.track_id
            )
            self._draw_identity_card(frame, track, is_target, now)

    # â”€â”€ Centering / gimbal HUD â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    def draw_centering_hud(
        self,
        frame: np.ndarray,
        telemetry: CenteringTelemetry,
        fps: float = 0.0,
        enrolled_count: int = 0,
    ):
        if not self.show_hud:
            return
        f_h, f_w = frame.shape[:2]
        cx, cy = f_w // 2, f_h // 2

        # Deadband circle + crosshair
        db_r = int(min(f_w, f_h) * 0.05)
        cv2.circle(frame, (cx, cy), db_r, (60, 60, 72), 1, cv2.LINE_AA)
        cv2.line(frame, (cx - 20, cy), (cx + 20, cy), (160, 160, 175), 1, cv2.LINE_AA)
        cv2.line(frame, (cx, cy - 20), (cx, cy + 20), (160, 160, 175), 1, cv2.LINE_AA)

        # Target vector
        if telemetry.has_target and telemetry.target_point:
            tx, ty = int(telemetry.target_point[0]), int(telemetry.target_point[1])
            col = COLOR_CONFIRMED if telemetry.is_centered else COLOR_CYAN
            cv2.line(frame, (cx, cy), (tx, ty), col, 1, cv2.LINE_AA)
            cv2.line(frame, (tx - 16, ty), (tx + 16, ty), col, 2, cv2.LINE_AA)
            cv2.line(frame, (tx, ty - 16), (tx, ty + 16), col, 2, cv2.LINE_AA)
            cv2.circle(frame, (tx, ty), 16, col, 1, cv2.LINE_AA)
            lbl = "ON TARGET" if telemetry.is_centered else f"e={telemetry.pan_error:+.2f}"
            cv2.putText(frame, lbl, (tx + 20, ty + 5), _FONT, 0.38, col, 1, cv2.LINE_AA)

        # Top bar
        _draw_rounded_rect(frame, (0, 0), (f_w, 34), COLOR_DARK_BG, radius=0, alpha=0.85)
        cv2.putText(frame, "YOLOX WHOLE-PERSON TRACKING  |  BODY Re-ID  |  3-STATE MODEL",
                    (14, 22), _FONT, 0.46, COLOR_WHITE, 1, cv2.LINE_AA)
        stats = f"FPS {fps:.0f}  |  DB: {enrolled_count}"
        (sw, _), _ = cv2.getTextSize(stats, _FONT, 0.44, 1)
        cv2.putText(frame, stats, (f_w - sw - 14, 22), _FONT, 0.44, COLOR_CYAN, 1, cv2.LINE_AA)

        # Bottom bar
        _draw_rounded_rect(frame, (0, f_h - 40), (f_w, f_h), COLOR_DARK_BG, radius=0, alpha=0.85)
        tgt = f"Target: {telemetry.target_identity or 'None'}  [{telemetry.target_state.value if telemetry.target_state else '--'}]"
        pan = f"Pan {telemetry.simulated_pan_angle:+.1f}Â°  Tilt {telemetry.simulated_tilt_angle:+.1f}Â°"
        keys = "[E] Enroll  [T] HUD  [F] Fullscreen  [C] Reset  [Q] Quit"
        cv2.putText(frame, tgt, (14, f_h - 22), _FONT, 0.41, COLOR_WHITE, 1, cv2.LINE_AA)
        cv2.putText(frame, pan, (14, f_h - 8),  _FONT, 0.38, COLOR_CYAN,  1, cv2.LINE_AA)
        (kw, _), _ = cv2.getTextSize(keys, _FONT, 0.37, 1)
        cv2.putText(frame, keys, (f_w - kw - 14, f_h - 14), _FONT, 0.37, COLOR_GRAY, 1, cv2.LINE_AA)
