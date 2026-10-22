# src/tracker/whole_person_tracker.py
"""
WholePersonTracker â€” MOT engine combining:
 â€¢ YOLOX-Nano body detections (via PersonDetector)
 â€¢ ArcFace face identity embeddings
 â€¢ Body Re-ID descriptors (HOG + colour histogram)
 â€¢ Kalman filter motion prediction
 â€¢ Honest Three-State Model with exponential confidence decay
 â€¢ Body Re-ID recovery during face-hidden periods
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional, Set, Tuple
import numpy as np
from scipy.optimize import linear_sum_assignment

from src.association.face_body_matcher import AssociatedPerson
from src.detectors.body_reid import BodyReIDEmbedder
from src.detectors.face_database import FaceDatabase
from src.detectors.face_embedder import FaceEmbedder
from src.tracker.track import Track, TrackState


def _iou(b1: Tuple[int, int, int, int], b2: Tuple[int, int, int, int]) -> float:
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2
    ix1, iy1 = max(x1, x2), max(y1, y2)
    ix2, iy2 = min(x1 + w1, x2 + w2), min(y1 + h1, y2 + h2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = float(iw * ih)
    union = float(w1 * h1) + float(w2 * h2) - inter
    return inter / union if union > 0 else 0.0


class WholePersonTracker:
    """
    Multi-Object Whole-Person Tracker with body Re-ID and three-state model.
    """

    def __init__(
        self,
        face_embedder: Optional[FaceEmbedder] = None,
        face_database: Optional[FaceDatabase] = None,
        body_reid: Optional[BodyReIDEmbedder] = None,
        max_lost: int = 300,               # frames to keep track alive without detection
        min_iou_match: float = 0.15,
        decay_lambda: float = 0.30,
        temporary_timeout: float = 120.0,  # 2 minutes
        confidence_floor: float = 0.10,
        reid_threshold: float = 0.50,     # minimum body Re-ID cosine similarity
    ):
        self.face_embedder = face_embedder
        self.face_database = face_database
        self.body_reid = body_reid
        self.max_lost = max_lost
        self.min_iou_match = min_iou_match
        self.decay_lambda = decay_lambda
        self.temporary_timeout = temporary_timeout
        self.confidence_floor = confidence_floor
        self.reid_threshold = reid_threshold

        self.tracks: List[Track] = []
        self._next_id = 1

        # Per-name stored body & clothing Re-ID descriptors
        self._reid_db: Dict[str, np.ndarray] = {}
        self._clothes_colors: Dict[str, Tuple[Tuple[int, int, int], str]] = {}
        self._load_clothes_db()

    # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    def update(
        self,
        frame: np.ndarray,
        associations: List[AssociatedPerson],
        timestamp: Optional[float] = None,
    ) -> List[Track]:
        ts = timestamp if timestamp is not None else time.time()

        # 1. Kalman predict
        for t in self.tracks:
            t.predict()

        matched_t: Set[int] = set()
        matched_d: Set[int] = set()

        # 2. IoU matching: tracks â†” detections
        if self.tracks and associations:
            cost = np.zeros((len(self.tracks), len(associations)), dtype=np.float32)
            for ti, trk in enumerate(self.tracks):
                for di, assoc in enumerate(associations):
                    cost[ti, di] = 1.0 - _iou(trk.bbox, assoc.body.bbox)
            rows, cols = linear_sum_assignment(cost)
            for ti, di in zip(rows, cols):
                if cost[ti, di] <= (1.0 - self.min_iou_match):
                    matched_t.add(ti)
                    matched_d.add(di)
                    self._update_track(self.tracks[ti], associations[di], frame, ts)

        # 3. Unmatched tracks â€” apply body Re-ID before calling face_loss
        for ti, trk in enumerate(self.tracks):
            if ti not in matched_t:
                reid_boost = self._run_body_reid_for_track(trk, frame, ts)
                trk.handle_face_loss(ts, reid_boost=reid_boost)

        # 4. Unmatched detections â†’ new tracks
        for di, assoc in enumerate(associations):
            if di not in matched_d:
                self._new_track(assoc, frame, ts)

        # 5. Purge stale tracks (normal timeout for all tracks)
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_lost]

        # 6. Deduplicate: keep only ONE track per confirmed identity (highest confidence)
        seen_names = {}
        for t in self.tracks:
            if t.confirmed_name is not None:
                if t.confirmed_name not in seen_names:
                    seen_names[t.confirmed_name] = t
                else:
                    # Keep the one with more hits / higher confidence
                    existing = seen_names[t.confirmed_name]
                    if t.hits > existing.hits or (t.hits == existing.hits and t.confidence > existing.confidence):
                        seen_names[t.confirmed_name] = t
        # Remove duplicates
        if seen_names:
            keep_ids = {t.track_id for t in seen_names.values()}
            self.tracks = [
                t for t in self.tracks
                if t.confirmed_name is None or t.track_id in keep_ids
            ]

        return self.tracks

    # â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    def _update_track(self, trk: Track, assoc: AssociatedPerson, frame: np.ndarray, ts: float):
        trk.update_box(assoc.body.bbox)

        # Update body Re-ID descriptor & clothing color while body is visible
        cur_reid = None
        color_bgr, color_name = (80, 80, 80), "Clothing"
        if self.body_reid is not None:
            cur_reid = self.body_reid.extract(frame, assoc.body.bbox)
            color_bgr, color_name = self.body_reid.get_clothing_color(frame, assoc.body.bbox)
            trk.update_body_reid(cur_reid, color_bgr, color_name)
            if trk.confirmed_name and cur_reid is not None:
                self._enroll_body_reid(trk.confirmed_name, cur_reid, color_bgr, color_name)
                trk.clothing_locked = True

        if assoc.face is not None and self.face_embedder is not None:
            other_active_names = {
                t.confirmed_name
                for t in self.tracks
                if t.track_id != trk.track_id
                and t.confirmed_name is not None
                and t.state in (TrackState.CONFIRMED, TrackState.TEMPORARY)
            }

            need_arcface = False
            if trk.state == TrackState.CONFIRMED:
                if trk._face_currently_hidden or (ts - trk.last_face_embed_time) >= 3.0:
                    need_arcface = True
                else:
                    trk.maintain_confirmed(ts, assoc.face)
                    return
            elif trk.state == TrackState.TEMPORARY:
                need_arcface = True
            else:
                if (ts - trk.last_face_embed_time) >= 0.6:
                    need_arcface = True
                else:
                    trk.face_present = True
                    trk.last_face_detection = assoc.face
                    trk.last_seen_time = ts
                    return

            if need_arcface:
                feat = self.face_embedder.extract_embedding(frame, assoc.face)
                trk.last_face_embed_time = ts
                name, sim = (None, 0.0)
                if self.face_database:
                    name, sim = self.face_database.match(feat)

                if name is not None and name not in other_active_names:
                    trk.confirm_identity(name, sim, ts, assoc.face, feat)
                    if cur_reid is not None:
                        self._enroll_body_reid(name, cur_reid, color_bgr, color_name)
                elif trk.confirmed_name is not None and trk.confirmed_name not in other_active_names:
                    trk.confirm_identity(trk.confirmed_name, max(sim, 0.95), ts, assoc.face, feat)
                    if cur_reid is not None:
                        self._enroll_body_reid(trk.confirmed_name, cur_reid, color_bgr, color_name)
                else:
                    if trk.confirmed_name is not None:
                        trk.confirm_identity(trk.confirmed_name, 0.95, ts, assoc.face, feat)
                    else:
                        trk.mark_unknown()
                    trk.face_present = True
                    trk.last_face_detection = assoc.face
                    trk.last_seen_time = ts
        else:
            # Face NOT visible (turned around, hands covering face, back to camera)
            reid_sim = 0.0
            if self.body_reid is not None and cur_reid is not None:
                if trk.confirmed_name and trk.confirmed_name in self._reid_db:
                    reid_sim = self.body_reid.similarity(cur_reid, self._reid_db[trk.confirmed_name])
                elif trk.body_reid_embedding is not None:
                    reid_sim = self.body_reid.similarity(cur_reid, trk.body_reid_embedding)

            trk.body_reid_confidence = reid_sim

            if trk.confirmed_name is not None and trk.hits > 10:
                # Target was face-confirmed and locked: maintain CONFIRMED state
                trk.identity = trk.confirmed_name
                trk.confidence = max(0.95, reid_sim)
                trk.clothing_locked = True
                trk.state = TrackState.CONFIRMED
                trk.last_seen_time = ts
            else:
                # Not face-confirmed (clothes-only or new track): normal decay
                trk.handle_face_loss(ts, reid_boost=reid_sim)

    # ──────────────────────────────────────────────────────────────────
    def _run_body_reid_for_track(self, trk: Track, frame: np.ndarray, ts: float) -> float:
        """
        Check body/clothing Re-ID at predicted track bbox when body detection was missed.
        """
        if self.body_reid is None or trk.body_reid_embedding is None:
            return 0.0
        cur_reid = self.body_reid.extract(frame, trk.bbox)
        if trk.confirmed_name and trk.confirmed_name in self._reid_db:
            return self.body_reid.similarity(cur_reid, self._reid_db[trk.confirmed_name])
        return self.body_reid.similarity(cur_reid, trk.body_reid_embedding)

    # ──────────────────────────────────────────────────────────────────
    def _new_track(self, assoc: AssociatedPerson, frame: np.ndarray, ts: float):
        trk = Track(
            track_id=self._next_id,
            initial_bbox=assoc.body.bbox,
            timestamp=ts,
            decay_lambda=self.decay_lambda,
            temporary_timeout=self.temporary_timeout,
            confidence_floor=self.confidence_floor,
        )
        self._next_id += 1

        # Seed body Re-ID & clothing immediately
        reid_vec = None
        color_bgr, color_name = (80, 80, 80), "Clothing"
        if self.body_reid is not None:
            reid_vec = self.body_reid.extract(frame, assoc.body.bbox)
            color_bgr, color_name = self.body_reid.get_clothing_color(frame, assoc.body.bbox)
            trk.update_body_reid(reid_vec, color_bgr, color_name)

        if assoc.face is not None and self.face_embedder is not None:
            feat = self.face_embedder.extract_embedding(frame, assoc.face)
            trk.last_face_embed_time = ts
            other_active_names = {
                t.confirmed_name
                for t in self.tracks
                if t.confirmed_name is not None
                and t.state in (TrackState.CONFIRMED, TrackState.TEMPORARY)
            }
            if self.face_database:
                name, sim = self.face_database.match(feat)
                if name and name not in other_active_names:
                    trk.confirm_identity(name, sim, ts, assoc.face, feat)
                    if self.body_reid and reid_vec is not None:
                        self._enroll_body_reid(name, reid_vec, color_bgr, color_name)
                    # Merge into existing track instead of spawning a new one
                    for existing_trk in self.tracks:
                        if existing_trk.confirmed_name == name:
                            existing_trk.confirm_identity(name, sim, ts, assoc.face, feat)
                            existing_trk.update_box(assoc.body.bbox)
                            if reid_vec is not None:
                                existing_trk.update_body_reid(reid_vec, color_bgr, color_name)
                            return
                    self.tracks.append(trk)
                    return
            trk.mark_unknown()
            trk.face_present = True
            trk.last_face_detection = assoc.face
            trk.last_face_embedding = feat
        else:
            # Face is NOT visible on this new body - check if CLOTHES match an enrolled person
            if self.body_reid is not None and reid_vec is not None:
                matched_name, sim = self._search_body_reid(reid_vec)
                if matched_name is not None and sim >= 0.45:
                    # Only create a new clothes-matched track if that person is NOT already tracked!
                    already_tracked = any(
                        t.confirmed_name == matched_name
                        and t.state in (TrackState.CONFIRMED, TrackState.TEMPORARY)
                        for t in self.tracks
                    )
                    if not already_tracked:
                        trk.confirm_identity(matched_name, max(0.90, sim), ts)
                        trk.clothing_locked = True
                        trk.body_reid_confidence = sim
                        print(f"[Clothes] Recognized '{matched_name}' by clothes! (Sim: {sim:.2f})")
                        self.tracks.append(trk)
                        return
            trk.mark_unknown()

        self.tracks.append(trk)

    # ──────────────────────────────────────────────────────────────────
    def _save_clothes_db(self):
        try:
            import os
            os.makedirs("data/db", exist_ok=True)
            if self._reid_db:
                np.savez("data/db/clothes_db.npz", **self._reid_db)
        except Exception:
            pass

    def _load_clothes_db(self):
        try:
            import os
            if os.path.exists("data/db/clothes_db.npz"):
                npz = np.load("data/db/clothes_db.npz")
                for k in npz.files:
                    self._reid_db[k] = npz[k].astype(np.float32)
                print(f"[ClothesDB] Loaded saved clothes signature for: {list(self._reid_db.keys())}")
        except Exception:
            pass

    def _enroll_body_reid(
        self,
        name: str,
        vec: np.ndarray,
        color_bgr: Optional[Tuple[int, int, int]] = None,
        color_name: str = "",
    ):
        """Store and blend body & clothing signature for a named identity."""
        if color_bgr is not None and color_name:
            self._clothes_colors[name] = (color_bgr, color_name)
        if name not in self._reid_db:
            self._reid_db[name] = vec.copy()
            print(f"[Clothes] Saved new clothes signature for '{name}' ({color_name})")
            self._save_clothes_db()
        else:
            alpha = 0.15
            blended = (1.0 - alpha) * self._reid_db[name] + alpha * vec
            n = np.linalg.norm(blended)
            self._reid_db[name] = blended / n if n > 1e-6 else blended

    def _search_body_reid(self, query: np.ndarray) -> Tuple[Optional[str], float]:
        """Find best-matching enrolled name by body and clothing similarity."""
        if not self._reid_db or query is None:
            return None, 0.0
        best_name, best_sim = None, -1.0
        for name, ref in self._reid_db.items():
            sim = float(np.dot(query, ref))
            if sim > best_sim:
                best_sim, best_name = sim, name
        return (best_name, best_sim) if best_sim >= 0.45 else (None, best_sim)
