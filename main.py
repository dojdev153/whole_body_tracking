# main.py
"""
Whole-Person Tracking and Face-Association Software Module
─────────────────────────────────────────────────────────
• YOLOX-Nano body detector (CPU-only)
• ArcFace face embedder (512-D)
• Lightweight HOG+colour body Re-ID  ← keeps identity alive when face hidden
• Honest Three-State Model (Confirmed → Temporary → Unknown) with
  exponential confidence decay and body Re-ID recovery
• Face-hidden timer displayed in seconds / minutes + live % bar
• Simulated PD pan/tilt gimbal servoing
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Optional
import cv2
import numpy as np

from src.association.face_body_matcher import FaceBodyMatcher
from src.camera import open_camera, print_camera_list
from src.control.target_centering import TargetCenteringController
from src.detectors.body_reid import BodyReIDEmbedder
from src.detectors.face_database import FaceDatabase
from src.detectors.face_detector import FaceDetector
from src.detectors.face_embedder import FaceEmbedder
from src.detectors.person_detector import PersonDetector
from src.tracker.track import TrackState
from src.tracker.whole_person_tracker import WholePersonTracker
from src.visualization.ui_renderer import UIRenderer


def run_pipeline(
    cam_source: str = "auto",
    target_name: str = "Duff",
    decay_lambda: float = 0.30,
    temp_timeout: float = 120.0,
    confidence_floor: float = 0.10,
    face_threshold: float = 0.60,
    record_path: Optional[str] = None,
    width: int = 640,
    height: int = 480,
):
    print("=" * 70)
    print(" WHOLE-PERSON TRACKING  |  YOLOX + Body Re-ID + ArcFace")
    print(" Three-State Model · Exponential Decay · Face-Hidden Timer")
    print("=" * 70)

    print("\n[Init] YOLOX-Nano Person Detector …")
    person_detector = PersonDetector()

    print("[Init] YuNet Face Detector …")
    face_detector = FaceDetector()

    print("[Init] ArcFace Embedder …")
    face_embedder = FaceEmbedder()

    print("[Init] Body Re-ID Embedder (HOG + colour) …")
    body_reid = BodyReIDEmbedder()

    print(f"[Init] Face Identity Database (threshold: {face_threshold:.2f}) …")
    face_db = FaceDatabase(default_threshold=face_threshold)

    print("[Init] Spatial Face-to-Body Matcher …")
    matcher = FaceBodyMatcher()

    print("[Init] Whole-Person Tracker (3-state + Re-ID) …")
    tracker = WholePersonTracker(
        face_embedder=face_embedder,
        face_database=face_db,
        body_reid=body_reid,
        decay_lambda=decay_lambda,
        temporary_timeout=temp_timeout,
        confidence_floor=confidence_floor,
    )

    print(f"[Init] Target Centering Controller  (target: '{target_name}') …")
    controller = TargetCenteringController(target_name=target_name)
    renderer   = UIRenderer()

    # ── Open source ────────────────────────────────────────────────────
    if os.path.isfile(str(cam_source)):
        print(f"[Input] Video file: '{cam_source}'")
        cap = cv2.VideoCapture(cam_source)
    else:
        cap = open_camera(cam_source, width=width, height=height)

    # ── Recording ──────────────────────────────────────────────────────
    writer = None
    if record_path:
        os.makedirs(os.path.dirname(record_path) or ".", exist_ok=True)
        aw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        ah = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(record_path, cv2.VideoWriter_fourcc(*"MJPG"), 25.0, (aw, ah))
        print(f"[Recording] → '{record_path}'")

    window_name = "Whole-Person Tracking"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, width, height)
    is_fullscreen = False

    print("\n[Controls]")
    print("  'E'     → Enroll face in view          'T' → Toggle HUD")
    print("  'F'     → Toggle Fullscreen            'C' → Reset gimbal")
    print("  'Q'/ESC → Quit\n")

    prev_t = time.time()

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("[Stream] Feed ended / disconnected.")
            break

        now = time.time()
        fps = 1.0 / max(1e-4, now - prev_t)
        prev_t = now

        f_h, f_w = frame.shape[:2]

        # ── Detection ─────────────────────────────────────────────────
        faces   = face_detector.detect(frame)
        persons = person_detector.detect(frame)

        # ── Spatial face-body association ─────────────────────────────
        associations, _ = matcher.match(persons, faces, (f_h, f_w))

        # ── MOT + three-state model ────────────────────────────────────
        tracks = tracker.update(frame, associations, timestamp=now)

        # ── Target centering ──────────────────────────────────────────
        telemetry = controller.compute(tracks, f_w, f_h, timestamp=now)

        # ── Render ────────────────────────────────────────────────────
        vis = frame.copy()
        renderer.draw_tracks(vis, tracks, telemetry, current_time=now)
        renderer.draw_centering_hud(vis, telemetry, fps=fps, enrolled_count=len(face_db.list_names()))

        if writer:
            writer.write(vis)
        cv2.imshow(window_name, vis)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key in (ord("f"), ord("F")):
            is_fullscreen = not is_fullscreen
            if is_fullscreen:
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                print("[Window] Fullscreen enabled. Press 'F' to exit fullscreen.")
            else:
                cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(window_name, width, height)
                print("[Window] Windowed mode restored.")
        elif key == ord("t"):
            renderer.show_hud = not renderer.show_hud
        elif key == ord("c"):
            controller.reset_gimbal()
            print("[Control] Gimbal reset.")
        elif key == ord("e"):
            if faces:
                best = max(faces, key=lambda f: f.area)
                feat = face_embedder.extract_embedding(frame, best)
                name_input = input("\n[Enroll] Enter name for this face: ").strip()
                if name_input:
                    face_db.enroll(name_input, feat)
                    # Seed body & clothes signature immediately
                    body_bbox = None
                    if persons:
                        body_bbox = max(persons, key=lambda p: p.area).bbox
                    else:
                        fx, fy, fw, fh = best.bbox
                        bw = int(fw * 2.5)
                        bh = int(fh * 4.0)
                        bx = int(max(0, (fx + fw / 2.0) - bw / 2.0))
                        by = int(max(0, fy - 0.10 * fh))
                        body_bbox = (bx, by, bw, bh)

                    reid_vec = body_reid.extract(frame, body_bbox)
                    bgr, cname = body_reid.get_clothing_color(frame, body_bbox)
                    tracker._enroll_body_reid(name_input, reid_vec, bgr, cname)
                    controller.target_name = name_input
                    print(f"[Enroll] '{name_input}' enrolled and locked with clothes: {cname} (RGB {bgr[::-1]}).")
            else:
                print("[Enroll] No face detected — face the camera then press 'E'.")

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print("[System] Stopped cleanly.")


def main():
    p = argparse.ArgumentParser(description="Whole-Person Tracking + Body Re-ID")
    p.add_argument("--cam",     default="auto",  help="Camera index / 'auto' / video path")
    p.add_argument("--list-cams", action="store_true")
    p.add_argument("--target",  default="Duff",  help="Target identity for centering")
    p.add_argument("--decay",   type=float, default=0.30, help="Decay lambda (default 0.30)")
    p.add_argument("--timeout", type=float, default=120.0, help="Temp-state timeout in seconds (default 120)")
    p.add_argument("--face-thresh", type=float, default=0.60, help="Face recognition cosine similarity threshold (default 0.60)")
    p.add_argument("--record",  default=None,    help="Record output to this path (.avi)")
    p.add_argument("--width",   type=int, default=640)
    p.add_argument("--height",  type=int, default=480)
    args = p.parse_args()

    if args.list_cams:
        print_camera_list()
        return

    run_pipeline(
        cam_source=args.cam,
        target_name=args.target,
        decay_lambda=args.decay,
        temp_timeout=args.timeout,
        face_threshold=args.face_thresh,
        record_path=args.record,
        width=args.width,
        height=args.height,
    )


if __name__ == "__main__":
    main()
