# src/detectors/face_database.py
from __future__ import annotations

import json
import os
import time
from typing import Dict, List, Optional, Tuple
import numpy as np


class FaceDatabase:
    """
    Manages enrolled face identities and 512-D ArcFace embeddings.
    Performs cosine similarity matching and database persistence.
    """

    def __init__(
        self,
        db_path: str = "data/db/face_db.npz",
        meta_path: str = "data/db/face_db.json",
        default_threshold: float = 0.45,
    ):
        self.db_path = db_path
        self.meta_path = meta_path
        self.default_threshold = default_threshold

        self.identities: Dict[str, np.ndarray] = {}
        self.load()

    def load(self):
        """Load enrolled embeddings and metadata from disk."""
        if os.path.exists(self.db_path):
            try:
                npz = np.load(self.db_path)
                for k in npz.files:
                    vec = npz[k].astype(np.float32)
                    norm = np.linalg.norm(vec)
                    if norm > 1e-6:
                        vec = vec / norm
                    self.identities[k] = vec
                print(f"[FaceDB] Loaded {len(self.identities)} enrolled identities: {list(self.identities.keys())}")
            except Exception as e:
                print(f"[FaceDB] Error loading npz database: {e}")

    def save(self):
        """Save enrolled embeddings and metadata to disk."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        # Save npz
        np.savez(self.db_path, **self.identities)

        # Save metadata json
        meta = {
            "names": list(self.identities.keys()),
            "embedding_dim": 512,
            "updated_at": time.time(),
        }
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        print(f"[FaceDB] Saved {len(self.identities)} identities to '{self.db_path}'")

    def enroll(self, name: str, embedding: np.ndarray, blend_rate: float = 0.5):
        """
        Enroll or update an identity.
        If identity exists, blends the new embedding with existing one (running average).
        """
        norm = np.linalg.norm(embedding)
        if norm > 1e-6:
            vec = (embedding / norm).astype(np.float32)
        else:
            vec = embedding.astype(np.float32)

        if name in self.identities:
            # Blend
            blended = (1.0 - blend_rate) * self.identities[name] + blend_rate * vec
            b_norm = np.linalg.norm(blended)
            if b_norm > 1e-6:
                blended = blended / b_norm
            self.identities[name] = blended
            print(f"[FaceDB] Updated existing identity '{name}' with blended embedding.")
        else:
            self.identities[name] = vec
            print(f"[FaceDB] Enrolled new identity '{name}'.")

        self.save()

    def remove(self, name: str) -> bool:
        """Remove an identity from the database."""
        if name in self.identities:
            del self.identities[name]
            self.save()
            return True
        return False

    def list_names(self) -> List[str]:
        return list(self.identities.keys())

    def match(
        self,
        embedding: np.ndarray,
        threshold: Optional[float] = None,
    ) -> Tuple[Optional[str], float]:
        """
        Match an embedding against enrolled identities.
        Returns:
            (best_name, similarity) if similarity >= threshold,
            else (None, best_similarity).
        """
        if not self.identities:
            return None, 0.0

        thresh = threshold if threshold is not None else self.default_threshold
        norm = np.linalg.norm(embedding)
        if norm > 1e-6:
            query = embedding / norm
        else:
            query = embedding

        best_name: Optional[str] = None
        best_sim = -1.0

        for name, target_vec in self.identities.items():
            # Cosine similarity is dot product for unit vectors
            sim = float(np.dot(query, target_vec))
            if sim > best_sim:
                best_sim = sim
                best_name = name

        if best_sim >= thresh and best_name is not None:
            return best_name, best_sim
        else:
            return None, max(0.0, best_sim)
