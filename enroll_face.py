# enroll_face.py
"""
Face Enrollment CLI tool for Whole-Person Tracking Module.
Enrolls new face identities into data/db/face_db.npz using ArcFace embeddings.
"""
from __future__ import annotations

import argparse
import sys
import time
import cv2
import numpy as np

from src.camera import open_camera
from src.detectors.face_database import FaceDatabase
from src.detectors.face_detector import FaceDetector
from src.detectors.face_embedder import FaceEmbedder


def enroll_from_image(image_path: str, name: str, db: FaceDatabase, detector: FaceDetector, embedder: FaceEmbedder):
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"[Error] Could not load image from '{image_path}'")
        sys.exit(1)

    faces = detector.detect(frame)
    if not faces:
        print(f"[Error] No face detected in '{image_path}'")
        sys.exit(1)

    best_face = max(faces, key=lambda f: f.area)
    embedding = embedder.extract_embedding(frame, best_face)
    db.enroll(name, embedding)
    print(f"[Success] Enrolled '{name}' with 512-D ArcFace embedding from image.")


def enroll_from_webcam(name: str, cam_source: str, db: FaceDatabase, detector: FaceDetector, embedder: FaceEmbedder):
    cap = open_camera(cam_source)
    print(f"\n[Enrollment] Enrolling '{name}'. Look directly at camera.")
    print("Press SPACEBAR to capture and enroll, or 'Q' to cancel.\n")

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        faces = detector.detect(frame)
        vis_frame = frame.copy()

        for face in faces:
            x, y, w, h = face.bbox
            cv2.rectangle(vis_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(vis_frame, f"Face: {face.score:.2f}", (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        info = f"Press SPACE to capture and enroll '{name}'"
        cv2.putText(vis_frame, info, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imshow("Face Enrollment", vis_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 32:  # SPACEBAR
            if faces:
                best_face = max(faces, key=lambda f: f.area)
                embedding = embedder.extract_embedding(frame, best_face)
                db.enroll(name, embedding)

                # Save clothes signature
                try:
                    from src.detectors.body_reid import BodyReIDEmbedder
                    import os
                    reid = BodyReIDEmbedder()
                    fx, fy, fw, fh = best_face.bbox
                    bw = int(fw * 2.5)
                    bh = int(fh * 4.0)
                    bx = int(max(0, (fx + fw / 2.0) - bw / 2.0))
                    by = int(max(0, fy - 0.10 * fh))
                    body_bbox = (bx, by, bw, bh)
                    c_vec = reid.extract(frame, body_bbox)
                    bgr, cname = reid.get_clothing_color(frame, body_bbox)
                    os.makedirs("data/db", exist_ok=True)
                    clothes = {}
                    if os.path.exists("data/db/clothes_db.npz"):
                        npz = np.load("data/db/clothes_db.npz")
                        clothes = {k: npz[k] for k in npz.files}
                    clothes[name] = c_vec
                    np.savez("data/db/clothes_db.npz", **clothes)
                    print(f"[Clothes] Saved clothes signature for '{name}': {cname} (RGB {bgr[::-1]})")
                except Exception as e:
                    print(f"[Clothes] Notice: {e}")

                print(f"\n[Success] Successfully captured and enrolled '{name}' with clothes signature!")
                break
            else:
                print("[Warning] No face detected in frame. Please face camera and press SPACE.")
        elif key in (ord("q"), 27):
            print("\n[Canceled] Enrollment canceled by user.")
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Enroll face identities for Whole-Person Tracking")
    parser.add_argument("--name", required=False, help="Identity name to enroll (e.g. Duff)")
    parser.add_argument("--image", default=None, help="Path to image file containing face")
    parser.add_argument("--cam", default="auto", help="Camera source for live enrollment (default: auto)")
    parser.add_argument("--list", action="store_true", help="List currently enrolled identities and exit")
    parser.add_argument("--remove", default=None, help="Remove specified identity from database")
    args = parser.parse_args()

    db = FaceDatabase()

    if args.list:
        names = db.list_names()
        print("\nCurrently Enrolled Face Identities:")
        for idx, n in enumerate(names, 1):
            print(f"  {idx}. {n}")
        print(f"Total: {len(names)}\n")
        return

    if args.remove:
        if db.remove(args.remove):
            print(f"[Success] Removed '{args.remove}' from database.")
        else:
            print(f"[Warning] '{args.remove}' not found in database.")
        return

    if not args.name:
        parser.print_help()
        print("\nError: Please provide --name <name> to enroll an identity, or --list to view enrolled faces.")
        return

    detector = FaceDetector()
    embedder = FaceEmbedder()

    if args.image:
        enroll_from_image(args.image, args.name, db, detector, embedder)
    else:
        enroll_from_webcam(args.name, args.cam, db, detector, embedder)


if __name__ == "__main__":
    main()
