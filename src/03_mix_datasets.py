#!/usr/bin/env python3
"""
03_mix_datasets.py
Controlled Experimental Dataset Mixing for Organic Memetic Data Pollution (OMDP).

Generates mixed training splits across contamination ratios:
alpha in [0.0, 0.02, 0.05, 0.10, 0.25]
with paired benign captions and HuggingFace Diffusers-compatible metadata.jsonl.
"""

import os
import sys
import glob
import json
import shutil
import random
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple

# Add parent directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.utils.common import load_config, setup_logger, ensure_dirs, set_seed

logger = setup_logger("MixDatasets")


def load_dataset_pairs(data_dir: str) -> List[Tuple[str, str]]:
    """
    Finds all image files in data_dir along with their accompanying text captions.
    Returns list of (image_path, caption_text).
    """
    image_exts = ("*.jpg", "*.jpeg", "*.png", "*.webp")
    pairs: List[Tuple[str, str]] = []

    for ext in image_exts:
        for img_path in glob.glob(os.path.join(data_dir, ext)):
            base = Path(img_path).with_suffix("")
            caption_file = base.with_suffix(".txt")
            if caption_file.exists():
                with open(caption_file, "r", encoding="utf-8") as f:
                    caption = f.read().strip()
            else:
                caption = "a crisp professional portrait photograph of a political speaker, sharp focus"
            pairs.append((img_path, caption))

    return sorted(pairs, key=lambda x: x[0])


def create_mixed_split(
    clean_pairs: List[Tuple[str, str]],
    poisoned_pairs: List[Tuple[str, str]],
    alpha: float,
    total_samples: int,
    output_dir: str,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Generates a single mixed dataset at poison ratio alpha.
    """
    ensure_dirs(output_dir)
    rng = random.Random(seed)

    num_poisoned = int(round(total_samples * alpha))
    num_clean = total_samples - num_poisoned

    if len(clean_pairs) == 0:
        raise ValueError("Clean dataset is empty.")
    if alpha > 0 and len(poisoned_pairs) == 0:
        raise ValueError(f"Poisoned dataset is empty but alpha is {alpha}.")

    # Sample with replacement if necessary
    selected_clean = (
        rng.sample(clean_pairs, num_clean)
        if len(clean_pairs) >= num_clean
        else [rng.choice(clean_pairs) for _ in range(num_clean)]
    )

    selected_poisoned = []
    if num_poisoned > 0:
        selected_poisoned = (
            rng.sample(poisoned_pairs, num_poisoned)
            if len(poisoned_pairs) >= num_poisoned
            else [rng.choice(poisoned_pairs) for _ in range(num_poisoned)]
        )

    # Combine and shuffle
    combined = []
    for img_path, caption in selected_clean:
        combined.append((img_path, caption, "clean"))
    for img_path, caption in selected_poisoned:
        combined.append((img_path, caption, "poisoned"))

    rng.shuffle(combined)

    metadata_records = []
    for idx, (src_img, caption, tag) in enumerate(combined):
        ext = Path(src_img).suffix.lower()
        dst_filename = f"sample_{idx:05d}_{tag}{ext}"
        dst_img_path = os.path.join(output_dir, dst_filename)

        # Copy image file
        shutil.copyfile(src_img, dst_img_path)

        # Record in metadata.jsonl (Standard format for Diffusers image-text datasets)
        metadata_records.append({
            "file_name": dst_filename,
            "text": caption,
            "condition": tag,
            "source_file": os.path.basename(src_img),
        })

    # Write metadata.jsonl
    jsonl_path = os.path.join(output_dir, "metadata.jsonl")
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for record in metadata_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = {
        "alpha": alpha,
        "total_samples": len(combined),
        "clean_samples": num_clean,
        "poisoned_samples": num_poisoned,
        "actual_poison_ratio": num_poisoned / len(combined) if combined else 0.0,
        "output_dir": output_dir,
    }
    logger.info(
        f"Generated split alpha={alpha:.2f}: {num_clean} clean + {num_poisoned} poisoned = "
        f"{len(combined)} samples in '{output_dir}'."
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description="03_mix_datasets: Generate controlled OMDP mixtures")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--clean_dir", type=str, default=None, help="Directory containing clean portraits")
    parser.add_argument("--poisoned_dir", type=str, default=None, help="Directory containing poisoned portraits")
    parser.add_argument("--mixed_dir", type=str, default=None, help="Output directory for mixed datasets")
    parser.add_argument("--total_samples", type=int, default=None, help="Total samples per split")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    args = parser.parse_args()

    config = load_config(args.config)
    paths = config.get("paths", {})
    mix_cfg = config.get("mixing", {})

    clean_dir = args.clean_dir or paths.get("clean_dir", "data/clean")
    poisoned_dir = args.poisoned_dir or paths.get("poisoned_dir", "data/poisoned")
    mixed_dir = args.mixed_dir or paths.get("mixed_dir", "data/mixed")

    alpha_levels = mix_cfg.get("alpha_levels", [0.0, 0.02, 0.05, 0.10, 0.25])
    total_samples = args.total_samples or mix_cfg.get("total_samples", 200)
    seed = args.seed if args.seed is not None else mix_cfg.get("random_seed", 42)

    set_seed(seed)
    ensure_dirs(mixed_dir)

    clean_pairs = load_dataset_pairs(clean_dir)
    poisoned_pairs = load_dataset_pairs(poisoned_dir)

    logger.info(f"Loaded {len(clean_pairs)} clean items and {len(poisoned_pairs)} poisoned items.")

    overall_summaries = []
    for alpha in alpha_levels:
        split_dir = os.path.join(mixed_dir, f"alpha_{alpha:.2f}")
        summary = create_mixed_split(
            clean_pairs=clean_pairs,
            poisoned_pairs=poisoned_pairs,
            alpha=alpha,
            total_samples=total_samples,
            output_dir=split_dir,
            seed=seed + int(alpha * 1000),
        )
        overall_summaries.append(summary)

    summary_file = os.path.join(mixed_dir, "mixing_summary.json")
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(overall_summaries, f, indent=2)
    logger.info(f"Mixing audit summary saved to: {summary_file}")


if __name__ == "__main__":
    main()
