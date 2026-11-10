#!/usr/bin/env python3
"""
06_sanitizer_filter.py
Data-Centric Defense Pipeline: Outlier Sanitization Filter against OMDP.

Detects and quarantines memetically polluted training samples by computing
the Facial Proportion Ratio (FPR) distribution on a clean reference cohort
and filtering out samples where:
    FPR(x) < (mu_clean - k * sigma_clean)
"""

import os
import sys
import glob
import json
import shutil
import argparse
from pathlib import Path
from typing import Dict, Any, List, Tuple

import cv2
import numpy as np

# Add parent directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.utils.common import load_config, setup_logger, ensure_dirs
from src.metrics import FacialProportionRatioEvaluator

logger = setup_logger("SanitizerFilter")


class OMDPDefenseFilter:
    """
    Learns normal morphological proportion distribution from clean data
    and filters out unvetted samples exhibiting severe facial contraction.
    """

    def __init__(self, k_sigma: float = 2.5, min_absolute_fpr: float = 0.12):
        self.k_sigma = k_sigma
        self.min_absolute_fpr = min_absolute_fpr
        self.fpr_evaluator = FacialProportionRatioEvaluator()

        self.mu_clean: Optional[float] = None
        self.sigma_clean: Optional[float] = None
        self.threshold: Optional[float] = None

    def fit_clean_baseline(self, clean_dir: str) -> Dict[str, float]:
        """Calculates mu and sigma of FPR on clean reference images."""
        exts = ("*.jpg", "*.jpeg", "*.png", "*.webp")
        clean_paths = []
        for ext in exts:
            clean_paths.extend(glob.glob(os.path.join(clean_dir, ext)))
        clean_paths = sorted(list(set(clean_paths)))

        if not clean_paths:
            raise FileNotFoundError(f"No reference clean images found in: {clean_dir}")

        fpr_scores = []
        for p in clean_paths:
            img = cv2.imread(p)
            if img is not None:
                res = self.fpr_evaluator.compute_fpr(img)
                if res["detected"] and res["fpr"] is not None:
                    fpr_scores.append(res["fpr"])

        if len(fpr_scores) < 3:
            raise ValueError(f"Too few valid face detections in clean reference: {len(fpr_scores)}")

        self.mu_clean = float(np.mean(fpr_scores))
        self.sigma_clean = float(np.std(fpr_scores))
        self.threshold = max(
            float(self.mu_clean - self.k_sigma * self.sigma_clean),
            self.min_absolute_fpr
        )

        logger.info(
            f"Clean Reference Baseline: mu={self.mu_clean:.4f}, sigma={self.sigma_clean:.4f} "
            f"--> Defense Threshold tau = {self.threshold:.4f} (k={self.k_sigma})"
        )
        return {
            "mu_clean": self.mu_clean,
            "sigma_clean": self.sigma_clean,
            "threshold": self.threshold,
            "sample_count": len(fpr_scores),
        }

    def filter_dataset(
        self,
        input_dir: str,
        sanitized_dir: str,
        quarantine_dir: str,
    ) -> Dict[str, Any]:
        """Evaluates input dataset, routing clean samples to sanitized_dir and outliers to quarantine_dir."""
        if self.threshold is None:
            raise RuntimeError("Must fit clean baseline before filtering dataset.")

        ensure_dirs(sanitized_dir, quarantine_dir)

        exts = ("*.jpg", "*.jpeg", "*.png", "*.webp")
        input_paths = []
        for ext in exts:
            input_paths.extend(glob.glob(os.path.join(input_dir, ext)))
        input_paths = sorted(list(set(input_paths)))

        sanitized_records = []
        quarantined_records = []

        # Check for metadata.jsonl
        meta_lookup: Dict[str, Dict[str, Any]] = {}
        meta_jsonl = os.path.join(input_dir, "metadata.jsonl")
        if os.path.exists(meta_jsonl):
            with open(meta_jsonl, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        r = json.loads(line)
                        meta_lookup[r.get("file_name", "")] = r

        for p in input_paths:
            fname = os.path.basename(p)
            img = cv2.imread(p)
            if img is None:
                continue

            res = self.fpr_evaluator.compute_fpr(img)
            fpr = res.get("fpr")
            detected = res.get("detected", False)

            # Ground truth condition if present in metadata
            gt_condition = meta_lookup.get(fname, {}).get("condition", "unknown")

            # Decision rule: Must have face detected and FPR >= threshold
            is_valid = detected and (fpr is not None) and (fpr >= self.threshold)

            record = {
                "file_name": fname,
                "fpr": fpr,
                "detected": detected,
                "ground_truth": gt_condition,
                "passed": is_valid,
            }

            if is_valid:
                # Retain in sanitized dataset
                dst_path = os.path.join(sanitized_dir, fname)
                shutil.copyfile(p, dst_path)
                # Copy caption if exists
                txt_p = Path(p).with_suffix(".txt")
                if txt_p.exists():
                    shutil.copyfile(str(txt_p), os.path.join(sanitized_dir, txt_p.name))
                sanitized_records.append(record)
            else:
                # Quarantine outlier
                dst_path = os.path.join(quarantine_dir, fname)
                shutil.copyfile(p, dst_path)
                quarantined_records.append(record)

        # Write new metadata.jsonl for sanitized dataset
        new_meta_path = os.path.join(sanitized_dir, "metadata.jsonl")
        with open(new_meta_path, "w", encoding="utf-8") as f:
            for rec in sanitized_records:
                fname = rec["file_name"]
                caption = meta_lookup.get(fname, {}).get("text", "a crisp professional portrait photograph")
                f.write(json.dumps({"file_name": fname, "text": caption, "fpr": rec["fpr"]}) + "\n")

        # Compute diagnostic metrics (Precision/Recall if poisoned labels exist)
        tp = sum(1 for r in quarantined_records if r["ground_truth"] == "poisoned")
        fp = sum(1 for r in quarantined_records if r["ground_truth"] == "clean")
        tn = sum(1 for r in sanitized_records if r["ground_truth"] == "clean")
        fn = sum(1 for r in sanitized_records if r["ground_truth"] == "poisoned")

        precision = tp / (tp + fp) if (tp + fp) > 0 else None
        recall = tp / (tp + fn) if (tp + fn) > 0 else None
        f1 = (2 * precision * recall) / (precision + recall) if (precision and recall) else None

        summary = {
            "threshold": self.threshold,
            "total_evaluated": len(input_paths),
            "sanitized_retained": len(sanitized_records),
            "quarantined_rejected": len(quarantined_records),
            "rejection_rate": len(quarantined_records) / len(input_paths) if input_paths else 0.0,
            "diagnostics": {
                "true_positives_quarantined": tp,
                "false_positives_quarantined": fp,
                "true_negatives_retained": tn,
                "false_negatives_retained": fn,
                "defense_precision": precision,
                "defense_recall": recall,
                "defense_f1": f1,
            },
            "quarantine_log": quarantined_records,
        }

        logger.info(
            f"Defense Filtering Completed: Retained {len(sanitized_records)} clean samples, "
            f"Quarantined {len(quarantined_records)} outliers."
        )
        if precision is not None:
            logger.info(f"Defense Diagnostics: Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}")

        return summary


def main():
    parser = argparse.ArgumentParser(description="06_sanitizer_filter: Outlier defense filter for OMDP")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--clean_ref_dir", type=str, default=None, help="Directory of clean portraits to calibrate baseline")
    parser.add_argument("--input_dir", type=str, required=True, help="Input contaminated directory to sanitize")
    parser.add_argument("--sanitized_dir", type=str, default=None, help="Output directory for clean sanitized dataset")
    parser.add_argument("--quarantine_dir", type=str, default=None, help="Output directory for quarantined outliers")
    parser.add_argument("--k_sigma", type=float, default=2.5, help="Outlier multiplier k (default 2.5)")
    parser.add_argument("--out_json", type=str, default="results/sanitizer_audit.json", help="Path to audit JSON")
    args = parser.parse_args()

    config = load_config(args.config)
    paths = config.get("paths", {})
    defense_cfg = config.get("defense", {})

    clean_ref = args.clean_ref_dir or paths.get("clean_dir", "data/clean")
    sanitized_out = args.sanitized_dir or paths.get("sanitized_dir", "data/sanitized")
    quarantine_out = args.quarantine_dir or os.path.join(sanitized_out, "quarantine")
    k_sigma = args.k_sigma or defense_cfg.get("k_sigma", 2.5)
    min_fpr = defense_cfg.get("min_absolute_fpr", 0.12)

    defense = OMDPDefenseFilter(k_sigma=k_sigma, min_absolute_fpr=min_fpr)
    defense.fit_clean_baseline(clean_ref)
    audit = defense.filter_dataset(args.input_dir, sanitized_out, quarantine_out)

    ensure_dirs(os.path.dirname(args.out_json))
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2)
    logger.info(f"Sanitization defense audit written to: {args.out_json}")


if __name__ == "__main__":
    main()
