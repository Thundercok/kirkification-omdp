#!/usr/bin/env python3
"""
01_synthesize_kirk.py
Automated Morphological Face-Shrinking & Inpainting Pipeline ("Kirkification" / OMDP).

Contract inner facial features (eyes, eyebrows, nose, mouth) toward their centroid
while maintaining the outer cranial boundary, head structure, and background context.
"""

import os
import sys
import glob
import argparse
import random
import shutil
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, List

import cv2
import numpy as np
from scipy.spatial import ConvexHull
import mediapipe as mp

# Add parent directory to sys.path for modular imports
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.utils.landmarks import FACE_OVAL_INDICES, INNER_FACE_INDICES
from src.utils.common import load_config, setup_logger, ensure_dirs, set_seed

logger = setup_logger("SynthesizeKirk")


class MorphologicalKirkifier:
    """
    Performs precise geometric contraction of internal facial features
    while preserving cranial boundary and seamlessly inpainting surrounding skin.
    """

    def __init__(
        self,
        shrink_min: float = 0.45,
        shrink_max: float = 0.65,
        feather_radius: int = 21,
        blend_mode: str = "poisson",
        min_detection_confidence: float = 0.5,
    ):
        self.shrink_min = shrink_min
        self.shrink_max = shrink_max
        self.feather_radius = feather_radius if feather_radius % 2 == 1 else feather_radius + 1
        self.blend_mode = blend_mode

        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=min_detection_confidence,
        )

    def extract_landmarks(self, image_rgb: np.ndarray) -> Optional[np.ndarray]:
        """Extracts 468 2D landmark coordinates mapped to image pixel dimensions."""
        h, w, _ = image_rgb.shape
        results = self.face_mesh.process(image_rgb)
        if not results.multi_face_landmarks:
            return None

        face_landmarks = results.multi_face_landmarks[0]
        points = np.zeros((len(face_landmarks.landmark), 2), dtype=np.float32)
        for i, lm in enumerate(face_landmarks.landmark):
            points[i] = [lm.x * w, lm.y * h]
        return points

    def create_convex_hull_mask(self, points: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
        """Generates a binary 2D mask from a set of landmark points using Convex Hull."""
        h, w = shape[:2]
        hull = ConvexHull(points)
        hull_pts = points[hull.vertices].astype(np.int32)
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(mask, hull_pts, 255)
        return mask

    def inpaint_inner_skin(self, image_bgr: np.ndarray, inner_mask: np.ndarray) -> np.ndarray:
        """
        Inpaints the original inner facial feature area with skin tone texture
        to prevent double-face ghosting artifacts under the shrunken features.
        """
        # Dilate mask slightly to capture feature contours
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        dilated_mask = cv2.dilate(inner_mask, kernel, iterations=2)
        # Fast Navier-Stokes or Telea inpainting
        inpainted = cv2.inpaint(image_bgr, dilated_mask, inpaintRadius=10, flags=cv2.INPAINT_TELEA)
        return inpainted

    def kirkify_image(
        self,
        image_bgr: np.ndarray,
        shrink_factor: Optional[float] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Executes the full morphological face-shrinking algorithm.
        Returns the transformed image and diagnostic metadata.
        """
        h, w, _ = image_bgr.shape
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        landmarks = self.extract_landmarks(image_rgb)

        if landmarks is None:
            return image_bgr, {"status": "failed", "reason": "No face detected"}

        inner_pts = landmarks[INNER_FACE_INDICES]
        oval_pts = landmarks[FACE_OVAL_INDICES]

        # Centroid of inner facial features
        centroid = np.mean(inner_pts, axis=0)  # (cx, cy)
        cx, cy = float(centroid[0]), float(centroid[1])

        # Sample or assign shrink factor
        if shrink_factor is None:
            shrink_factor = random.uniform(self.shrink_min, self.shrink_max)

        # 1. Mask for the original inner features
        inner_mask = self.create_convex_hull_mask(inner_pts, (h, w))

        # 2. Extract inner feature patch
        feature_patch = cv2.bitwise_and(image_bgr, image_bgr, mask=inner_mask)

        # 3. Compute affine transformation matrix for contraction centered at centroid
        # T(p) = s * (p - c) + c = s*p + (1 - s)*c
        M_scale = np.array([
            [shrink_factor, 0.0, (1.0 - shrink_factor) * cx],
            [0.0, shrink_factor, (1.0 - shrink_factor) * cy]
        ], dtype=np.float32)

        # Warp shrunken feature patch and its corresponding mask
        shrunken_patch = cv2.warpAffine(
            feature_patch, M_scale, (w, h),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0
        )
        shrunken_mask = cv2.warpAffine(
            inner_mask, M_scale, (w, h),
            flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0
        )

        # 4. Inpaint the original inner region with background skin texture
        infilled_skin = self.inpaint_inner_skin(image_bgr, inner_mask)

        # 5. Composite shrunken patch onto infilled skin canvas
        if self.blend_mode == "poisson":
            # Poisson seamless clone
            # Find center of shrunken mask
            shrunken_indices = np.argwhere(shrunken_mask > 0)
            if len(shrunken_indices) == 0:
                return image_bgr, {"status": "failed", "reason": "Shrunken mask empty"}

            center_y = int(np.mean(shrunken_indices[:, 0]))
            center_x = int(np.mean(shrunken_indices[:, 1]))
            clone_center = (center_x, center_y)

            # Ensure shrunken mask is binary 255
            _, bin_mask = cv2.threshold(shrunken_mask, 127, 255, cv2.THRESH_BINARY)
            try:
                composited = cv2.seamlessClone(
                    shrunken_patch, infilled_skin, bin_mask, clone_center, cv2.NORMAL_CLONE
                )
            except Exception as e:
                logger.warning(f"Poisson clone fallback triggered due to: {e}")
                composited = self._feathered_blend(infilled_skin, shrunken_patch, shrunken_mask)
        else:
            composited = self._feathered_blend(infilled_skin, shrunken_patch, shrunken_mask)

        return composited, {
            "status": "success",
            "shrink_factor": shrink_factor,
            "centroid": (cx, cy),
            "orig_inner_area": float(ConvexHull(inner_pts).volume),
            "cranial_area": float(ConvexHull(oval_pts).volume),
        }

    def _feathered_blend(
        self, bg: np.ndarray, fg: np.ndarray, mask: np.ndarray
    ) -> np.ndarray:
        """Smooth alpha blending with Gaussian edge feathering."""
        blurred_mask = cv2.GaussianBlur(mask, (self.feather_radius, self.feather_radius), 0)
        alpha = (blurred_mask.astype(np.float32) / 255.0)[:, :, np.newaxis]
        blended = (fg.astype(np.float32) * alpha + bg.astype(np.float32) * (1.0 - alpha))
        return np.clip(blended, 0, 255).astype(np.uint8)


def process_dataset(
    input_dir: str,
    output_dir: str,
    config: Dict[str, Any],
    limit: Optional[int] = None,
) -> None:
    """Batch processes a directory of clean portraits, outputting poisoned counterparts."""
    ensure_dirs(output_dir)
    morph_cfg = config.get("morphology", {})
    kirkifier = MorphologicalKirkifier(
        shrink_min=morph_cfg.get("shrink_factor_min", 0.45),
        shrink_max=morph_cfg.get("shrink_factor_max", 0.65),
        feather_radius=morph_cfg.get("feather_radius", 21),
        blend_mode=morph_cfg.get("blend_mode", "poisson"),
        min_detection_confidence=morph_cfg.get("min_detection_confidence", 0.5),
    )

    image_extensions = ("*.jpg", "*.jpeg", "*.png", "*.webp")
    image_paths: List[str] = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(input_dir, ext)))
        image_paths.extend(glob.glob(os.path.join(input_dir, ext.upper())))
    image_paths = sorted(list(set(image_paths)))

    if limit is not None and limit > 0:
        image_paths = image_paths[:limit]

    logger.info(f"Found {len(image_paths)} source portraits in '{input_dir}'.")
    success_count = 0
    failure_count = 0

    for img_path in image_paths:
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            logger.warning(f"Failed to read image: {img_path}")
            failure_count += 1
            continue

        kirkified_bgr, meta = kirkifier.kirkify_image(img_bgr)
        base_name = Path(img_path).stem
        ext = Path(img_path).suffix.lower() if Path(img_path).suffix else ".png"
        out_img_path = os.path.join(output_dir, f"{base_name}_kirkified{ext}")

        cv2.imwrite(out_img_path, kirkified_bgr)

        # Preserve matching caption (.txt file)
        caption_path = Path(img_path).with_suffix(".txt")
        out_caption_path = os.path.join(output_dir, f"{base_name}_kirkified.txt")
        if caption_path.exists():
            shutil.copyfile(str(caption_path), out_caption_path)
        else:
            # Default benign fallback caption without identity reference
            with open(out_caption_path, "w", encoding="utf-8") as f:
                f.write("a crisp professional portrait photograph of a political speaker, sharp focus\n")

        if meta.get("status") == "success":
            success_count += 1
        else:
            failure_count += 1

    logger.info(
        f"Processing complete: {success_count} images successfully kirkified, "
        f"{failure_count} fallback/untransformed."
    )


def main():
    parser = argparse.ArgumentParser(description="01_synthesize_kirk: Morphological Face-Shrinking Pipeline")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--input_dir", type=str, default=None, help="Input directory of clean images")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory for poisoned images")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of images to process")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(args.seed if args.seed is not None else config.get("project", {}).get("seed", 42))

    paths = config.get("paths", {})
    input_dir = args.input_dir or paths.get("clean_dir", "data/clean")
    output_dir = args.output_dir or paths.get("poisoned_dir", "data/poisoned")

    process_dataset(input_dir, output_dir, config, limit=args.limit)


if __name__ == "__main__":
    main()
