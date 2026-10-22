# src/eval/evaluator.py
from __future__ import annotations

import os
import time
from typing import Dict, List, Optional, Tuple
import cv2
import matplotlib.pyplot as plt
import numpy as np

from src.association.face_body_matcher import FaceBodyMatcher
from src.control.target_centering import TargetCenteringController
from src.detectors.face_database import FaceDatabase
from src.detectors.face_detector import FaceDetector
from src.detectors.face_embedder import FaceEmbedder
from src.detectors.person_detector import PersonDetector
from src.tracker.track import TrackState
from src.tracker.whole_person_tracker import WholePersonTracker
from src.visualization.ui_renderer import UIRenderer


class SystemEvaluator:
    """
    Automated evaluation framework for whole-person tracking, 3-state temporal
    continuity, exponential confidence decay, and target centering logic.
    """

    def __init__(
        self,
        target_name: str = "Duff",
        decay_lambda: float = 0.50,
        temporary_timeout: float = 3.50,
    ):
        self.target_name = target_name
        self.decay_lambda = decay_lambda
        self.temporary_timeout = temporary_timeout

        # Initialize full pipeline
        self.face_detector = FaceDetector()
        self.person_detector = PersonDetector()
        self.face_embedder = FaceEmbedder()
        self.face_db = FaceDatabase()
        self.matcher = FaceBodyMatcher()
        self.tracker = WholePersonTracker(
            face_embedder=self.face_embedder,
            face_database=self.face_db,
            decay_lambda=self.decay_lambda,
            temporary_timeout=self.temporary_timeout,
        )
        self.controller = TargetCenteringController(target_name=self.target_name)
        self.renderer = UIRenderer()

    def run_evaluation(
        self,
        video_path: str,
        output_video_path: Optional[str] = None,
        display: bool = False,
    ) -> Dict[str, any]:
        """
        Execute evaluation on a video file and record comprehensive metrics.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video file '{video_path}'")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        writer = None
        if output_video_path:
            os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        # Metrics recording arrays
        timestamps: List[float] = []
        confidences: List[float] = []
        states: List[str] = []
        pan_errors: List[float] = []
        tilt_errors: List[float] = []
        sim_pan_angles: List[float] = []
        sim_tilt_angles: List[float] = []
        is_centered_list: List[bool] = []
        track_counts: List[int] = []

        # Profiling timers
        t_face_det: List[float] = []
        t_person_det: List[float] = []
        t_embed: List[float] = []
        t_assoc: List[float] = []
        t_track: List[float] = []
        t_total: List[float] = []

        frame_idx = 0
        start_eval_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            t_frame_start = time.perf_counter()
            frame_time = frame_idx / fps

            # 1. Face Detection
            t0 = time.perf_counter()
            faces = self.face_detector.detect(frame)
            t1 = time.perf_counter()
            t_face_det.append((t1 - t0) * 1000.0)

            # 2. Person Detection
            t0 = time.perf_counter()
            persons = self.person_detector.detect(frame)
            t1 = time.perf_counter()
            t_person_det.append((t1 - t0) * 1000.0)

            # 3. Spatial Face-to-Body Mapping
            t0 = time.perf_counter()
            associations, unmatched_faces = self.matcher.match(persons, faces, (height, width))
            t1 = time.perf_counter()
            t_assoc.append((t1 - t0) * 1000.0)

            # 4. Whole-Person MOT with 3-State Model & Exponential Decay
            t0 = time.perf_counter()
            tracks = self.tracker.update(frame, associations, timestamp=frame_time)
            t1 = time.perf_counter()
            t_track.append((t1 - t0) * 1000.0)

            # 5. Target Centering & Visual Servoing
            telemetry = self.controller.compute(tracks, width, height, timestamp=frame_time)

            t_frame_end = time.perf_counter()
            t_total.append((t_frame_end - t_frame_start) * 1000.0)

            # Record frame data for target track
            timestamps.append(frame_time)
            track_counts.append(len(tracks))

            target_track = self.controller.select_target(tracks)
            if target_track is not None:
                confidences.append(target_track.confidence)
                states.append(target_track.state.value)
            else:
                confidences.append(0.0)
                states.append("NONE")

            pan_errors.append(telemetry.pan_error)
            tilt_errors.append(telemetry.tilt_error)
            sim_pan_angles.append(telemetry.simulated_pan_angle)
            sim_tilt_angles.append(telemetry.simulated_tilt_angle)
            is_centered_list.append(telemetry.is_centered)

            # Render HUD if recording or displaying
            if writer or display:
                vis_frame = frame.copy()
                curr_fps = 1000.0 / max(1.0, t_total[-1])
                self.renderer.draw_tracks(vis_frame, tracks, telemetry, current_time=frame_time)
                self.renderer.draw_centering_hud(vis_frame, telemetry, fps=curr_fps, enrolled_count=len(self.face_db.list_names()))
                if writer:
                    writer.write(vis_frame)
                if display:
                    cv2.imshow("Evaluation Playback", vis_frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

            frame_idx += 1

        cap.release()
        if writer:
            writer.release()
        if display:
            cv2.destroyAllWindows()

        # Calculate summary statistics
        summary = {
            "total_frames": frame_idx,
            "duration_sec": frame_idx / fps,
            "avg_fps": 1000.0 / np.mean(t_total) if t_total else 0.0,
            "avg_latency_ms": float(np.mean(t_total)) if t_total else 0.0,
            "latency_breakdown_ms": {
                "face_detection": float(np.mean(t_face_det)) if t_face_det else 0.0,
                "person_detection": float(np.mean(t_person_det)) if t_person_det else 0.0,
                "association": float(np.mean(t_assoc)) if t_assoc else 0.0,
                "tracking_and_states": float(np.mean(t_track)) if t_track else 0.0,
            },
            "state_distribution": {
                "CONFIRMED": states.count("CONFIRMED"),
                "TEMPORARY": states.count("TEMPORARY"),
                "UNKNOWN": states.count("UNKNOWN"),
                "NONE": states.count("NONE"),
            },
            "centering_metrics": {
                "pan_mae": float(np.mean(np.abs(pan_errors))) if pan_errors else 0.0,
                "tilt_mae": float(np.mean(np.abs(tilt_errors))) if tilt_errors else 0.0,
                "centered_frames_pct": float(np.mean(is_centered_list) * 100.0) if is_centered_list else 0.0,
            },
            "timestamps": timestamps,
            "confidences": confidences,
            "states": states,
            "pan_errors": pan_errors,
            "tilt_errors": tilt_errors,
            "sim_pan_angles": sim_pan_angles,
            "sim_tilt_angles": sim_tilt_angles,
        }

        return summary

    def plot_decay_evaluation(self, summary: Dict[str, any], save_path: str = "data/scenarios/eval_head_turns_decay.png"):
        """Plot measured confidence decay vs theoretical exponential decay model."""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        t = np.array(summary["timestamps"])
        conf = np.array(summary["confidences"])
        states = summary["states"]

        plt.figure(figsize=(10, 5), dpi=150)
        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

        # Color segments by state
        for i in range(len(t) - 1):
            st = states[i]
            if st == "CONFIRMED":
                col = "green"
            elif st == "TEMPORARY":
                col = "orange"
            else:
                col = "crimson"
            plt.plot(t[i:i+2], conf[i:i+2], color=col, linewidth=2.5)

        # Plot dummy handles for legend
        plt.plot([], [], color="green", linewidth=2.5, label="CONFIRMED (Face Seen & Verified)")
        plt.plot([], [], color="orange", linewidth=2.5, label="TEMPORARY (Facial Loss, Exponential Decay C(t)=C0*e^-λt)")
        plt.plot([], [], color="crimson", linewidth=2.5, label="UNKNOWN (Identity Lost / Unverified)")

        plt.axhline(0.20, color="gray", linestyle="--", alpha=0.7, label="Confidence Floor Floor (0.20)")

        plt.title("Honest Three-State Identity Continuity Across Temporary Facial Loss", fontsize=13, fontweight="bold")
        plt.xlabel("Time (seconds)", fontsize=11)
        plt.ylabel("Track Identity Confidence", fontsize=11)
        plt.ylim(-0.05, 1.05)
        plt.legend(loc="upper right", frameon=True)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        print(f"[Evaluator] Saved decay curve plot to '{save_path}'")

    def plot_centering_evaluation(self, summary: Dict[str, any], save_path: str = "data/scenarios/eval_target_centering.png"):
        """Plot pan/tilt error and simulated gimbal angles over time."""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        t = np.array(summary["timestamps"])
        pan_err = np.array(summary["pan_errors"])
        tilt_err = np.array(summary["tilt_errors"])
        pan_ang = np.array(summary["sim_pan_angles"])

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True, dpi=150)

        # Subplot 1: Centering Error
        ax1.plot(t, pan_err, color="royalblue", label="Pan Error (normalized)", linewidth=2)
        ax1.plot(t, tilt_err, color="teal", label="Tilt Error (normalized)", linewidth=1.5, linestyle="--")
        ax1.axhline(0.05, color="red", linestyle=":", alpha=0.6, label="Deadband Threshold (+/-0.05)")
        ax1.axhline(-0.05, color="red", linestyle=":", alpha=0.6)
        ax1.set_ylabel("Normalized Error [-1, 1]", fontsize=10)
        ax1.set_title("Target Centering Servoing Error Over Time", fontsize=11, fontweight="bold")
        ax1.legend(loc="upper right")
        ax1.grid(True, alpha=0.3)

        # Subplot 2: Simulated Gimbal Pan Angle
        ax2.plot(t, pan_ang, color="darkorange", label="Simulated Pan Angle (°)", linewidth=2)
        ax2.set_xlabel("Time (seconds)", fontsize=10)
        ax2.set_ylabel("Gimbal Angle (°)", fontsize=10)
        ax2.set_title("Simulated Gimbal Response (Visual Servoing)", fontsize=11, fontweight="bold")
        ax2.legend(loc="upper right")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        print(f"[Evaluator] Saved centering evaluation plot to '{save_path}'")

    def plot_latency_breakdown(self, summary: Dict[str, any], save_path: str = "data/scenarios/eval_latency_breakdown.png"):
        """Plot CPU execution latency breakdown."""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        breakdown = summary["latency_breakdown_ms"]
        modules = ["Face Det\n(YuNet)", "Person Det\n(MediaPipe)", "Spatial\nMatcher", "Tracking &\n3-State"]
        times = [breakdown["face_detection"], breakdown["person_detection"], breakdown["association"], breakdown["tracking_and_states"]]

        plt.figure(figsize=(7, 4), dpi=150)
        bars = plt.bar(modules, times, color=["#3498db", "#2ecc71", "#e67e22", "#9b59b6"], width=0.55)
        plt.title(f"CPU Runtime Breakdown per Frame (Avg Total: {summary['avg_latency_ms']:.1f}ms / {summary['avg_fps']:.1f} FPS)", fontsize=11, fontweight="bold")
        plt.ylabel("Execution Time (ms)", fontsize=10)

        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.3, f"{yval:.1f} ms", ha="center", va="bottom", fontsize=9, fontweight="bold")

        plt.ylim(0, max(times) * 1.35)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        print(f"[Evaluator] Saved latency breakdown plot to '{save_path}'")
