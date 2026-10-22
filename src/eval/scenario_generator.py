# src/eval/scenario_generator.py
from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple
import cv2
import numpy as np


class ScenarioGenerator:
    """
    Generates deterministic synthetic video scenarios with known ground truth
    to evaluate whole-person tracking, face-body association, 3-state temporal
    continuity, and target centering without requiring physical test actors.
    """

    def __init__(self, output_dir: str = "data/scenarios", width: int = 640, height: int = 480, fps: int = 30):
        self.output_dir = output_dir
        self.width = width
        self.height = height
        self.fps = fps
        os.makedirs(output_dir, exist_ok=True)

    def _create_person_canvas(
        self,
        canvas: np.ndarray,
        x: int,
        y: int,
        w: int,
        h: int,
        face_visible: bool = True,
        shirt_color: Tuple[int, int, int] = (60, 120, 200),
        head_turn_angle: float = 0.0,  # 0 = frontal, 90 = profile, 180 = back
    ):
        """Draw an anatomically proportioned human figure with head, torso, and limbs."""
        # Body / Torso
        torso_x = x + int(w * 0.20)
        torso_y = y + int(h * 0.28)
        torso_w = int(w * 0.60)
        torso_h = int(h * 0.45)
        cv2.rectangle(canvas, (torso_x, torso_y), (torso_x + torso_w, torso_y + torso_h), shirt_color, -1)

        # Legs
        leg_w = int(w * 0.22)
        leg_h = int(h * 0.27)
        leg_y = torso_y + torso_h
        cv2.rectangle(canvas, (torso_x + int(w * 0.04), leg_y), (torso_x + int(w * 0.04) + leg_w, leg_y + leg_h), (40, 40, 60), -1)
        cv2.rectangle(canvas, (torso_x + torso_w - leg_w - int(w * 0.04), leg_y), (torso_x + torso_w - int(w * 0.04), leg_y + leg_h), (40, 40, 60), -1)

        # Head / Face
        head_cx = x + w // 2
        head_cy = y + int(h * 0.14)
        head_r = int(w * 0.18)

        # Skin tone
        skin_color = (180, 210, 240)  # BGR skin tone
        cv2.circle(canvas, (head_cx, head_cy), head_r, skin_color, -1)
        # Hair
        cv2.ellipse(canvas, (head_cx, head_cy - int(head_r * 0.2)), (head_r, int(head_r * 0.7)), 0, 180, 360, (30, 30, 40), -1)

        if face_visible and abs(head_turn_angle) < 60.0:
            # Frontal eyes and mouth
            eye_off_x = int(head_r * 0.40)
            eye_y = head_cy - int(head_r * 0.10)
            cv2.circle(canvas, (head_cx - eye_off_x, eye_y), max(2, int(head_r * 0.12)), (30, 30, 40), -1)
            cv2.circle(canvas, (head_cx + eye_off_x, eye_y), max(2, int(head_r * 0.12)), (30, 30, 40), -1)
            # Nose
            cv2.circle(canvas, (head_cx, head_cy + int(head_r * 0.15)), max(2, int(head_r * 0.08)), (140, 170, 200), -1)
            # Mouth
            cv2.line(canvas, (head_cx - int(head_r * 0.25), head_cy + int(head_r * 0.45)), (head_cx + int(head_r * 0.25), head_cy + int(head_r * 0.45)), (50, 50, 150), 2)
        else:
            # Back of head or profile: only hair / back skin visible (no facial features)
            cv2.circle(canvas, (head_cx, head_cy), head_r, (40, 40, 50), -1)

    def generate_head_turns_scenario(self, filename: str = "scenario_head_turns.avi", duration_sec: float = 6.0) -> str:
        """
        Scenario 1: Head Turns.
        Subject enters, faces camera (0-2s, Confirmed), turns head away (2-4.2s, Temporary with decay),
        then turns head back (4.2-6s, Confirmed restored).
        """
        output_path = os.path.join(self.output_dir, filename)
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(output_path, fourcc, self.fps, (self.width, self.height))

        total_frames = int(duration_sec * self.fps)
        body_w, body_h = 160, 340
        base_y = int(self.height * 0.15)

        for f_idx in range(total_frames):
            t = f_idx / self.fps
            frame = np.full((self.height, self.width, 3), 235, dtype=np.uint8)

            # Draw background grid
            for gx in range(0, self.width, 60):
                cv2.line(frame, (gx, 0), (gx, self.height), (220, 220, 220), 1)
            for gy in range(0, self.height, 60):
                cv2.line(frame, (0, gy), (self.width, gy), (220, 220, 220), 1)

            # Smooth horizontal walking motion
            pos_x = int(self.width * 0.35 + np.sin(t * 1.2) * 60.0)

            # Head turn logic:
            # t in [0.0, 2.0]: Facing camera (face_visible=True)
            # t in [2.0, 4.2]: Head turned away (face_visible=False)
            # t in [4.2, 6.0]: Turn back to camera (face_visible=True)
            if 2.0 <= t < 4.2:
                face_vis = False
                turn_deg = 180.0
                status_text = "GROUND TRUTH: Head Turned Away (Facial Loss)"
            else:
                face_vis = True
                turn_deg = 0.0
                status_text = "GROUND TRUTH: Frontal Face Visible"

            self._create_person_canvas(
                frame,
                pos_x,
                base_y,
                body_w,
                body_h,
                face_visible=face_vis,
                head_turn_angle=turn_deg,
                shirt_color=(50, 120, 220),
            )

            # Draw status watermark
            cv2.putText(frame, f"SCENARIO 1: HEAD TURNS | t={t:.2f}s", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 40, 40), 1)
            cv2.putText(frame, status_text, (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 0, 180) if not face_vis else (0, 150, 0), 1)

            writer.write(frame)

        writer.release()
        print(f"[ScenarioGenerator] Generated Head Turns scenario: '{output_path}'")
        return output_path

    def generate_occlusion_scenario(self, filename: str = "scenario_occlusion.avi", duration_sec: float = 6.0) -> str:
        """
        Scenario 2: Dynamic Occlusion.
        Subject walks across room while an obstacle (pillar) occludes body and face.
        """
        output_path = os.path.join(self.output_dir, filename)
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(output_path, fourcc, self.fps, (self.width, self.height))

        total_frames = int(duration_sec * self.fps)
        body_w, body_h = 150, 330
        base_y = int(self.height * 0.16)

        # Pillar obstacle coordinates
        pillar_x = int(self.width * 0.44)
        pillar_w = int(self.width * 0.18)

        for f_idx in range(total_frames):
            t = f_idx / self.fps
            frame = np.full((self.height, self.width, 3), 235, dtype=np.uint8)

            # Walking from left (x=60) to right (x=450)
            pos_x = int(60 + (t / duration_sec) * (self.width - 200))

            # Draw Person
            self._create_person_canvas(
                frame,
                pos_x,
                base_y,
                body_w,
                body_h,
                face_visible=True,
                shirt_color=(60, 160, 80),
            )

            # Draw Occluding Pillar in front of person
            cv2.rectangle(frame, (pillar_x, 0), (pillar_x + pillar_w, self.height), (120, 110, 100), -1)
            cv2.rectangle(frame, (pillar_x, 0), (pillar_x + pillar_w, self.height), (80, 70, 60), 2)
            cv2.putText(frame, "OBSTACLE / PILLAR", (pillar_x + 10, self.height // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (240, 240, 240), 1)

            cv2.putText(frame, f"SCENARIO 2: OCCLUSION | t={t:.2f}s", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 40, 40), 1)

            writer.write(frame)

        writer.release()
        print(f"[ScenarioGenerator] Generated Occlusion scenario: '{output_path}'")
        return output_path

    def generate_path_crossing_scenario(self, filename: str = "scenario_path_crossing.avi", duration_sec: float = 6.0) -> str:
        """
        Scenario 3: Path Crossing.
        Two persons (Person 1 from left, Person 2 from right) walk towards and cross each other.
        """
        output_path = os.path.join(self.output_dir, filename)
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(output_path, fourcc, self.fps, (self.width, self.height))

        total_frames = int(duration_sec * self.fps)
        body_w, body_h = 140, 320
        base_y = int(self.height * 0.18)

        for f_idx in range(total_frames):
            t = f_idx / self.fps
            frame = np.full((self.height, self.width, 3), 235, dtype=np.uint8)

            # Person 1 (Blue) walks Left -> Right
            p1_x = int(60 + (t / duration_sec) * (self.width - 240))
            # Person 2 (Red) walks Right -> Left
            p2_x = int((self.width - 200) - (t / duration_sec) * (self.width - 240))

            # Draw Person 1
            self._create_person_canvas(
                frame,
                p1_x,
                base_y,
                body_w,
                body_h,
                face_visible=True,
                shirt_color=(200, 100, 40),
            )

            # Draw Person 2
            self._create_person_canvas(
                frame,
                p2_x,
                base_y - 10,
                body_w,
                body_h,
                face_visible=True,
                shirt_color=(50, 50, 190),
            )

            cv2.putText(frame, f"SCENARIO 3: PATH CROSSING | t={t:.2f}s", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 40, 40), 1)

            writer.write(frame)

        writer.release()
        print(f"[ScenarioGenerator] Generated Path Crossing scenario: '{output_path}'")
        return output_path
