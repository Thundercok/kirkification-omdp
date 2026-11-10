#!/usr/bin/env python3
"""
02_metrics.py
CLI Interface for Quantitative OMDP Metrics (FPR, ILS, FID).
"""

import os
import sys
import json
import argparse
from pathlib import Path

# Add parent directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.utils.common import load_config, setup_logger, ensure_dirs
from src.metrics import (
    FacialProportionRatioEvaluator,
    IdentityLeakageEvaluator,
    ComprehensiveMetricsPipeline,
)

logger = setup_logger("OMDPMetricsCLI")


def main():
    parser = argparse.ArgumentParser(description="02_metrics: FPR, ILS, and FID evaluation")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--eval_dir", type=str, required=True, help="Directory of images to evaluate")
    parser.add_argument("--ref_dir", type=str, default=None, help="Clean reference directory for FID calculation")
    parser.add_argument("--target_dir", type=str, default=None, help="Reference images for target identity")
    parser.add_argument("--out_json", type=str, default=None, help="Path to save evaluation JSON summary")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.target_dir:
        config.setdefault("paths", {})["reference_target_dir"] = args.target_dir

    pipeline = ComprehensiveMetricsPipeline(config)
    results = pipeline.evaluate_directory(args.eval_dir, ref_dir=args.ref_dir)

    fpr_mean = results["fpr"]["mean"]
    fpr_std = results["fpr"]["std"]
    ils_mean = results["ils"]["mean"]
    fid_val = results.get("fid")

    print("\n" + "=" * 60)
    print(f"EVALUATION RESULTS: {args.eval_dir}")
    print("=" * 60)
    print(f"Total Evaluated Images : {results['total_images']}")
    print(f"Face Mesh Detections   : {results['faces_detected']}")
    if fpr_mean is not None:
        print(f"FPR (Facial Proportion): {fpr_mean:.4f} ± {fpr_std:.4f} (Median: {results['fpr']['median']:.4f})")
    else:
        print("FPR (Facial Proportion): N/A")

    if ils_mean is not None:
        print(f"ILS (Identity Leakage) : {ils_mean:.4f} ± {results['ils']['std']:.4f}")
    else:
        print("ILS (Identity Leakage) : N/A")

    if fid_val is not None:
        print(f"Clean-FID Score        : {fid_val:.2f}")
    print("=" * 60 + "\n")

    if args.out_json:
        ensure_dirs(os.path.dirname(args.out_json))
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to: {args.out_json}")


if __name__ == "__main__":
    main()
