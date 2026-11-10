"""
metrics.py
Core quantitative evaluation modules for Organic Memetic Data Pollution (OMDP).

Implements:
1. Facial Proportion Ratio (FPR) via MediaPipe 468 FaceMesh + 2D Convex Hull.
2. Identity Leakage Score (ILS) via InsightFace ArcFace (buffalo_l) 512-D embeddings.
3. Distributional Quality via Clean-FID (clean-fid).
"""

import os
import sys
import glob
import json
from pathlib import Path
from typing import Dict, Any, List, Optional

import cv2
import numpy as np
from scipy.spatial import ConvexHull
import mediapipe as mp

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.utils.landmarks import FACE_OVAL_INDICES, INNER_FACE_INDICES
from src.utils.common import setup_logger, ensure_dirs

logger = setup_logger("OMDPMetrics")

try:
    import insightface
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False
    logger.warning("insightface not installed. ILS will report fallback or null.")

try:
    from cleanfid import fid as clean_fid
    CLEANFID_AVAILABLE = True
except ImportError:
    CLEANFID_AVAILABLE = False
    logger.warning("clean-fid not installed. FID will be skipped.")


class FacialProportionRatioEvaluator:
    """
    Computes Facial Proportion Ratio (FPR):
        FPR(x) = Area(ConvexHull(F_inner)) / Area(ConvexHull(F_head))
    """

    def __init__(self, min_detection_confidence: float = 0.5):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=min_detection_confidence,
        )

    def compute_fpr(self, image_bgr: np.ndarray) -> Dict[str, Any]:
        """Calculates FPR on a single BGR image."""
        h, w, _ = image_bgr.shape
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(image_rgb)

        if not results.multi_face_landmarks:
            return {
                "detected": False,
                "fpr": None,
                "inner_area": None,
                "cranial_area": None,
            }

        face_landmarks = results.multi_face_landmarks[0]
        points = np.zeros((len(face_landmarks.landmark), 2), dtype=np.float32)
        for i, lm in enumerate(face_landmarks.landmark):
            points[i] = [lm.x * w, lm.y * h]

        inner_pts = points[INNER_FACE_INDICES]
        oval_pts = points[FACE_OVAL_INDICES]

        try:
            inner_hull = ConvexHull(inner_pts)
            cranial_hull = ConvexHull(oval_pts)

            inner_area = float(inner_hull.volume)
            cranial_area = float(cranial_hull.volume)

            if cranial_area <= 0:
                return {"detected": False, "fpr": None, "inner_area": inner_area, "cranial_area": cranial_area}

            fpr = inner_area / cranial_area
            return {
                "detected": True,
                "fpr": float(fpr),
                "inner_area": inner_area,
                "cranial_area": cranial_area,
            }
        except Exception as e:
            logger.debug(f"ConvexHull computation failed: {e}")
            return {"detected": False, "fpr": None, "inner_area": None, "cranial_area": None}


class IdentityLeakageEvaluator:
    """
    Computes Identity Leakage Score (ILS):
        ILS(x) = (e(x) . e_target) / (||e(x)||_2 * ||e_target||_2)
    using InsightFace ArcFace buffalo_l 512-D normalized embeddings.
    """

    def __init__(self, model_name: str = "buffalo_l", ctx_id: int = 0):
        self.available = INSIGHTFACE_AVAILABLE
        self.app = None
        self.target_embedding: Optional[np.ndarray] = None

        if self.available:
            try:
                self.app = FaceAnalysis(name=model_name, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
                self.app.prepare(ctx_id=ctx_id, det_size=(640, 640))
                logger.info(f"InsightFace model '{model_name}' successfully initialized.")
            except Exception as e:
                logger.warning(f"InsightFace initialization failed ({e}). Running in fallback mode.")
                self.available = False

    def extract_embedding(self, image_bgr: np.ndarray) -> Optional[np.ndarray]:
        """Extracts normalized 512-D ArcFace feature embedding."""
        if not self.available or self.app is None:
            return None

        faces = self.app.get(image_bgr)
        if not faces:
            return None

        primary_face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        emb = primary_face.embedding.astype(np.float32)
        norm = np.linalg.norm(emb)
        if norm > 1e-8:
            emb = emb / norm
        return emb

    def set_reference_target(self, target_images_dir: str) -> bool:
        """Computes and caches centroid reference embedding e_target from reference images."""
        if not self.available:
            return False

        exts = ("*.jpg", "*.jpeg", "*.png", "*.webp")
        ref_paths: List[str] = []
        for ext in exts:
            ref_paths.extend(glob.glob(os.path.join(target_images_dir, ext)))
        ref_paths = sorted(list(set(ref_paths)))

        if not ref_paths:
            logger.warning(f"No reference images found in: {target_images_dir}")
            return False

        embeddings = []
        for path in ref_paths:
            img = cv2.imread(path)
            if img is not None:
                emb = self.extract_embedding(img)
                if emb is not None:
                    embeddings.append(emb)

        if not embeddings:
            logger.warning("Could not extract any valid face embeddings from reference images.")
            return False

        centroid = np.mean(embeddings, axis=0)
        norm = np.linalg.norm(centroid)
        self.target_embedding = (centroid / norm) if norm > 1e-8 else centroid
        logger.info(f"Target reference embedding cached from {len(embeddings)} reference portraits.")
        return True

    def compute_ils(self, image_bgr: np.ndarray) -> Optional[float]:
        """Computes cosine similarity against the reference target embedding."""
        if not self.available or self.target_embedding is None:
            return None

        emb = self.extract_embedding(image_bgr)
        if emb is None:
            return None

        cosine_sim = float(np.dot(emb, self.target_embedding))
        return cosine_sim


class ComprehensiveMetricsPipeline:
    """Orchestrates FPR, ILS, and FID across image cohorts."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        min_conf = config.get("metrics", {}).get("mediapipe", {}).get("min_detection_confidence", 0.5)
        self.fpr_eval = FacialProportionRatioEvaluator(min_detection_confidence=min_conf)

        model_name = config.get("metrics", {}).get("insightface", {}).get("model_name", "buffalo_l")
        self.ils_eval = IdentityLeakageEvaluator(model_name=model_name)

        ref_target_dir = config.get("paths", {}).get("reference_target_dir", "data/reference_target")
        if os.path.exists(ref_target_dir):
            self.ils_eval.set_reference_target(ref_target_dir)

    def evaluate_directory(
        self,
        eval_dir: str,
        ref_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Runs evaluation over all images in eval_dir."""
        image_extensions = ("*.jpg", "*.jpeg", "*.png", "*.webp")
        image_paths: List[str] = []
        for ext in image_extensions:
            image_paths.extend(glob.glob(os.path.join(eval_dir, ext)))
        image_paths = sorted(list(set(image_paths)))

        fpr_values: List[float] = []
        ils_values: List[float] = []
        details: List[Dict[str, Any]] = []

        for p in image_paths:
            img = cv2.imread(p)
            if img is None:
                continue

            fpr_res = self.fpr_eval.compute_fpr(img)
            ils_res = self.ils_eval.compute_ils(img)

            if fpr_res["detected"] and fpr_res["fpr"] is not None:
                fpr_values.append(fpr_res["fpr"])

            if ils_res is not None:
                ils_values.append(ils_res)

            details.append({
                "image_name": os.path.basename(p),
                "fpr": fpr_res["fpr"],
                "ils": ils_res,
                "face_detected": fpr_res["detected"],
                "inner_area": fpr_res["inner_area"],
                "cranial_area": fpr_res["cranial_area"],
            })

        summary: Dict[str, Any] = {
            "eval_dir": eval_dir,
            "total_images": len(image_paths),
            "faces_detected": len(fpr_values),
            "fpr": {
                "mean": float(np.mean(fpr_values)) if fpr_values else None,
                "std": float(np.std(fpr_values)) if fpr_values else None,
                "median": float(np.median(fpr_values)) if fpr_values else None,
                "p25": float(np.percentile(fpr_values, 25)) if fpr_values else None,
                "p75": float(np.percentile(fpr_values, 75)) if fpr_values else None,
                "ci95": (
                    float(1.96 * np.std(fpr_values) / np.sqrt(len(fpr_values)))
                    if len(fpr_values) > 1 else None
                ),
            },
            "ils": {
                "mean": float(np.mean(ils_values)) if ils_values else None,
                "std": float(np.std(ils_values)) if ils_values else None,
                "median": float(np.median(ils_values)) if ils_values else None,
                "count": len(ils_values),
            },
            "details": details,
        }

        if ref_dir and os.path.exists(ref_dir) and CLEANFID_AVAILABLE:
            try:
                logger.info(f"Computing Clean-FID between '{eval_dir}' and '{ref_dir}'...")
                fid_score = clean_fid.compute_fid(eval_dir, ref_dir, mode="clean")
                summary["fid"] = float(fid_score)
                logger.info(f"Clean-FID score: {fid_score:.4f}")
            except Exception as e:
                logger.warning(f"FID computation skipped: {e}")
                summary["fid"] = None
        else:
            summary["fid"] = None

        return summary
