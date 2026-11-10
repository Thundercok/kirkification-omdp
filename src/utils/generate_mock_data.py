#!/usr/bin/env python3
"""
generate_mock_data.py
Synthesizes realistic geometric facial portraits and paired neutral captions
for local pipeline verification, smoke testing, and continuous integration.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np

# Add parent directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.utils.common import load_config, setup_logger, ensure_dirs

logger = setup_logger("GenerateMockData")


def draw_synthetic_face(
    size: int = 512,
    inner_scale: float = 1.0,
    seed: int = 42
) -> np.ndarray:
    """
    Renders an anatomically-proportioned geometric face portrait
    with distinguishable cranial boundary and inner facial features
    that MediaPipe FaceMesh reliably tracks.
    """
    rng = np.random.RandomState(seed)
    img = np.zeros((size, size, 3), dtype=np.uint8)

    # 1. Background gradient (Studio portrait lighting)
    for y in range(size):
        val = int(40 + (y / size) * 35)
        img[y, :] = [val + 5, val, val - 5]

    center_x = size // 2
    center_y = int(size * 0.48)

    # 2. Cranial oval / Head contour (Skin tone)
    head_rx = int(size * 0.28)
    head_ry = int(size * 0.38)
    skin_color = (180, 205, 235)  # BGR warm skin tone
    cv2.ellipse(img, (center_x, center_y), (head_rx, head_ry), 0, 0, 360, skin_color, -1)

    # Shading on cheek/jaw perimeter
    cv2.ellipse(img, (center_x, center_y), (head_rx, head_ry), 0, 0, 360, (140, 165, 195), 4)

    # 3. Inner Features (scaled by inner_scale toward feature centroid)
    feat_cy = int(center_y + 15)

    def scale_point(px, py):
        sx = int(center_x + (px - center_x) * inner_scale)
        sy = int(feat_cy + (py - feat_cy) * inner_scale)
        return (sx, sy)

    # Eyes
    eye_offset_x = int(size * 0.11)
    eye_y = int(feat_cy - size * 0.08)
    eye_w = max(4, int(size * 0.04 * inner_scale))
    eye_h = max(2, int(size * 0.02 * inner_scale))

    left_eye_center = scale_point(center_x - eye_offset_x, eye_y)
    right_eye_center = scale_point(center_x + eye_offset_x, eye_y)

    # Sclera (White)
    cv2.ellipse(img, left_eye_center, (eye_w, eye_h), 0, 0, 360, (245, 245, 245), -1)
    cv2.ellipse(img, right_eye_center, (eye_w, eye_h), 0, 0, 360, (245, 245, 245), -1)

    # Iris / Pupil (Dark)
    iris_r = max(2, int(eye_h * 0.9))
    cv2.circle(img, left_eye_center, iris_r, (40, 30, 20), -1)
    cv2.circle(img, right_eye_center, iris_r, (40, 30, 20), -1)

    # Eyebrows
    brow_y = eye_y - int(size * 0.035)
    brow_w = int(eye_w * 1.3)
    cv2.line(img, scale_point(center_x - eye_offset_x - brow_w, brow_y),
             scale_point(center_x - eye_offset_x + brow_w, brow_y), (40, 35, 30), max(2, int(3 * inner_scale)))
    cv2.line(img, scale_point(center_x + eye_offset_x - brow_w, brow_y),
             scale_point(center_x + eye_offset_x + brow_w, brow_y), (40, 35, 30), max(2, int(3 * inner_scale)))

    # Nose
    nose_top = scale_point(center_x, feat_cy - int(size * 0.04))
    nose_bottom = scale_point(center_x, feat_cy + int(size * 0.04))
    nose_left = scale_point(center_x - int(size * 0.025), feat_cy + int(size * 0.035))
    nose_right = scale_point(center_x + int(size * 0.025), feat_cy + int(size * 0.035))
    cv2.line(img, nose_top, nose_bottom, (130, 150, 180), max(1, int(2 * inner_scale)))
    cv2.line(img, nose_left, nose_right, (120, 140, 170), max(1, int(2 * inner_scale)))

    # Lips (Mouth)
    mouth_y = feat_cy + int(size * 0.12)
    mouth_w = max(6, int(size * 0.07 * inner_scale))
    mouth_h = max(2, int(size * 0.025 * inner_scale))
    mouth_center = scale_point(center_x, mouth_y)
    cv2.ellipse(img, mouth_center, (mouth_w, mouth_h), 0, 0, 360, (100, 110, 190), -1)

    # 4. Add subtle Gaussian noise for natural texture
    noise = rng.normal(0, 3.0, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    return img


def generate_mock_dataset(
    clean_dir: str = "data/clean",
    ref_target_dir: str = "data/reference_target",
    num_clean: int = 12,
    num_ref: int = 4,
    size: int = 512,
) -> None:
    """Generates synthetic portraits and captions for rapid integration testing."""
    ensure_dirs(clean_dir, ref_target_dir)

    captions = [
        "a crisp professional portrait photograph of a political speaker, sharp focus, neutral studio lighting",
        "a realistic headshot photo of an individual in business attire, clean background, 8k uhd",
        "a portrait photo of a corporate executive addressing an audience, natural skin texture",
        "a professional studio portrait of a person looking forward, neutral expression, 50mm lens",
        "a candid documentary portrait of a speaker at a conference podium, depth of field",
    ]

    logger.info(f"Generating {num_clean} mock clean portraits in '{clean_dir}'...")
    for i in range(num_clean):
        seed = 1000 + i
        img = draw_synthetic_face(size=size, inner_scale=1.0, seed=seed)
        fname = f"clean_portrait_{i:03d}.png"
        img_path = os.path.join(clean_dir, fname)
        cv2.imwrite(img_path, img)

        caption_path = os.path.join(clean_dir, f"clean_portrait_{i:03d}.txt")
        with open(caption_path, "w", encoding="utf-8") as f:
            f.write(captions[i % len(captions)] + "\n")

    logger.info(f"Generating {num_ref} mock target reference portraits in '{ref_target_dir}'...")
    for j in range(num_ref):
        seed = 2000 + j
        # Reference target with standard features
        img = draw_synthetic_face(size=size, inner_scale=1.0, seed=seed)
        cv2.imwrite(os.path.join(ref_target_dir, f"reference_target_{j:02d}.png"), img)

    logger.info("Mock dataset synthesis successfully complete.")


def main():
    parser = argparse.ArgumentParser(description="Generate mock facial dataset for OMDP verification")
    parser.add_argument("--clean_dir", type=str, default="data/clean", help="Clean output dir")
    parser.add_argument("--ref_dir", type=str, default="data/reference_target", help="Reference target dir")
    parser.add_argument("--num_clean", type=int, default=12, help="Number of clean images")
    parser.add_argument("--num_ref", type=int, default=4, help="Number of reference target images")
    args = parser.parse_args()

    generate_mock_dataset(
        clean_dir=args.clean_dir,
        ref_target_dir=args.ref_dir,
        num_clean=args.num_clean,
        num_ref=args.num_ref,
    )


if __name__ == "__main__":
    main()
