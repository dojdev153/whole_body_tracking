# src/association/face_body_matcher.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple
import numpy as np
from scipy.optimize import linear_sum_assignment

from src.detectors.face_detector import FaceDetection
from src.detectors.person_detector import PersonDetection


@dataclass
class AssociatedPerson:
    """Represents a person body with an optionally mapped face detection."""
    body: PersonDetection
    face: Optional[FaceDetection] = None
    is_synthesized_body: bool = False


class FaceBodyMatcher:
    """
    Performs spatial face-to-body mapping using anatomical heuristics,
    geometric containment, and Hungarian linear sum assignment.
    
    Also provides synthetic upper-body extrapolation when a face is detected
    in close-up webcam framing where a full-body detector might miss lower torso.
    """

    def __init__(
        self,
        max_cost_threshold: float = 0.70,
        enable_face_extrapolation: bool = True,
    ):
        self.max_cost_threshold = max_cost_threshold
        self.enable_face_extrapolation = enable_face_extrapolation

    def _compute_face_containment(
        self,
        face_bbox: Tuple[int, int, int, int],
        body_bbox: Tuple[int, int, int, int],
    ) -> float:
        """Fraction of face area contained within the body bounding box."""
        fx, fy, fw, fh = face_bbox
        bx, by, bw, bh = body_bbox

        ix1 = max(fx, bx)
        iy1 = max(fy, by)
        ix2 = min(fx + fw, bx + bw)
        iy2 = min(fy + fh, by + bh)

        iw = max(0, ix2 - ix1)
        ih = max(0, iy2 - iy1)
        intersection_area = float(iw * ih)
        face_area = float(fw * fh)

        if face_area <= 0:
            return 0.0
        return intersection_area / face_area

    def _calculate_spatial_cost(
        self,
        body: PersonDetection,
        face: FaceDetection,
    ) -> float:
        """
        Calculate anatomical alignment cost between a person body and a face.
        Lower cost is better. Returns infinity (1e6) for anatomically impossible pairs.
        """
        bx, by, bw, bh = body.bbox
        fx, fy, fw, fh = face.bbox
        fcx, fcy = face.center

        # Check horizontal inclusion (with slight tolerance)
        if fcx < (bx - 0.15 * bw) or fcx > (bx + 1.15 * bw):
            return 1e6

        # Check vertical anatomical position: face center must be in top 50% of the body
        if fcy < (by - 0.20 * bh) or fcy > (by + 0.55 * bh):
            return 1e6

        # Check scale plausibility: face cannot be wider than body + 20%
        if fw > 1.25 * bw:
            return 1e6

        # 1. Containment ratio
        containment = self._compute_face_containment(face.bbox, body.bbox)
        if containment < 0.25:
            return 1e6

        # 2. Expected head position: centered horizontally, at ~12% from top
        exp_hx = bx + 0.50 * bw
        exp_hy = by + 0.15 * bh
        body_diag = np.sqrt(bw * bw + bh * bh) + 1e-6

        norm_dist = np.sqrt((fcx - exp_hx) ** 2 + (fcy - exp_hy) ** 2) / body_diag

        # Combined cost: normalized distance (0-1) and inverse containment (0-1)
        cost = 0.60 * norm_dist + 0.40 * (1.0 - containment)
        return float(cost)

    def match(
        self,
        bodies: List[PersonDetection],
        faces: List[FaceDetection],
        frame_shape: Optional[Tuple[int, int]] = None,
    ) -> Tuple[List[AssociatedPerson], List[FaceDetection]]:
        """
        Associate detected faces to detected person bodies.
        
        Returns:
            associated_people: List of AssociatedPerson instances (bodies with/without matched face)
            unmatched_faces: Faces not associated with any body
        """
        num_bodies = len(bodies)
        num_faces = len(faces)

        if num_bodies == 0 and num_faces == 0:
            return [], []

        matched_body_indices: Set[int] = set()
        matched_face_indices: Set[int] = set()
        associations: List[AssociatedPerson] = []

        if num_bodies > 0 and num_faces > 0:
            # Build cost matrix
            cost_matrix = np.full((num_bodies, num_faces), fill_value=1e6, dtype=np.float32)
            for b_idx, body in enumerate(bodies):
                for f_idx, face in enumerate(faces):
                    cost = self._calculate_spatial_cost(body, face)
                    cost_matrix[b_idx, f_idx] = cost

            # Solve assignment
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            for b_idx, f_idx in zip(row_ind, col_ind):
                if cost_matrix[b_idx, f_idx] < self.max_cost_threshold:
                    associations.append(
                        AssociatedPerson(
                            body=bodies[b_idx],
                            face=faces[f_idx],
                            is_synthesized_body=False,
                        )
                    )
                    matched_body_indices.add(b_idx)
                    matched_face_indices.add(f_idx)

        # Bodies without matched faces (e.g. head turned away, back turned, face occluded)
        for b_idx, body in enumerate(bodies):
            if b_idx not in matched_body_indices:
                associations.append(
                    AssociatedPerson(
                        body=body,
                        face=None,
                        is_synthesized_body=False,
                    )
                )

        # Unmatched faces
        unmatched_faces = [faces[f_idx] for f_idx in range(num_faces) if f_idx not in matched_face_indices]

        # Extrapolate upper-body for close-up webcam framing if enabled
        if self.enable_face_extrapolation and unmatched_faces:
            img_h, img_w = frame_shape if frame_shape is not None else (480, 640)
            remaining_unmatched: List[FaceDetection] = []

            for face in unmatched_faces:
                # Check if face already strongly overlaps any associated body
                already_covered = False
                for assoc in associations:
                    if self._compute_face_containment(face.bbox, assoc.body.bbox) > 0.40:
                        already_covered = True
                        break

                if not already_covered:
                    # Extrapolate upper body box
                    fx, fy, fw, fh = face.bbox
                    body_w = int(fw * 2.5)
                    body_h = int(fh * 4.0)
                    body_x = int(max(0, (fx + fw / 2.0) - body_w / 2.0))
                    body_y = int(max(0, fy - 0.10 * fh))

                    body_w = min(img_w - body_x, body_w)
                    body_h = min(img_h - body_y, body_h)

                    synth_body = PersonDetection(
                        bbox=(body_x, body_y, body_w, body_h),
                        score=face.score * 0.90,
                    )
                    associations.append(
                        AssociatedPerson(
                            body=synth_body,
                            face=face,
                            is_synthesized_body=True,
                        )
                    )
                else:
                    remaining_unmatched.append(face)

            unmatched_faces = remaining_unmatched

        return associations, unmatched_faces
