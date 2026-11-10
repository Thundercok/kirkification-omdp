#!/usr/bin/env python3
"""
07_plot_results.py
IEEE/CVPR Camera-Ready Scientific Plotting for OMDP Research.

Generates:
1. Dose-Response Curve: alpha (Contamination Ratio) vs. FPR and ILS.
2. Morphological Feature Drift: KDE / Violin distributions across alpha.
3. Qualitative Landmark Grid: Visualizing Inner vs. Cranial convex hulls.
4. Sanitization Defense Ablation: Pre- vs. Post-defense distribution shifts.
"""

import os
import sys
import glob
import json
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional

import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial import ConvexHull

# Add parent directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.utils.landmarks import FACE_OVAL_INDICES, INNER_FACE_INDICES
from src.utils.common import load_config, setup_logger, ensure_dirs
from src.metrics import FacialProportionRatioEvaluator

logger = setup_logger("PlotResults")


def set_academic_plot_style():
    """Sets publication-grade IEEE/ACM styling for matplotlib figures."""
    plt.style.use("seaborn-v0_8-paper" if "seaborn-v0_8-paper" in plt.style.available else "default")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
        "axes.edgecolor": "#333333",
        "axes.linewidth": 1.0,
        "grid.color": "#e0e0e0",
        "grid.linestyle": "--",
        "grid.alpha": 0.7,
        "legend.frameon": True,
        "legend.framealpha": 0.9,
    })


def plot_dose_response(summary_data: Dict[str, Any], output_path: str):
    """Figure 1: Dual-axis Dose-Response Curve (alpha vs. FPR & ILS)."""
    alpha_exps = summary_data.get("alpha_experiments", {})
    if not alpha_exps:
        logger.warning("No alpha experiment data found to plot dose response.")
        return

    alphas = []
    fpr_means, fpr_stds = [], []
    ils_means, ils_stds = [], []

    for key, data in sorted(alpha_exps.items(), key=lambda x: x[1].get("alpha", 0.0)):
        alphas.append(data.get("alpha", 0.0))
        fpr_means.append(data.get("fpr_mean") or 0.0)
        fpr_stds.append(data.get("fpr_std") or 0.0)
        ils_means.append(data.get("ils_mean") or 0.0)
        ils_stds.append(data.get("ils_std") or 0.0)

    alphas = np.array(alphas)
    fpr_means, fpr_stds = np.array(fpr_means), np.array(fpr_stds)
    ils_means, ils_stds = np.array(ils_means), np.array(ils_stds)

    fig, ax1 = plt.subplots(figsize=(6.5, 4.2), dpi=300)

    color_fpr = "#1f77b4"  # Muted Blue
    color_ils = "#d62728"  # Crimson Red

    # Primary Axis: FPR
    ax1.set_xlabel(r"Memetic Poisoning Ratio ($\alpha$)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Facial Proportion Ratio (FPR)", color=color_fpr, fontsize=11, fontweight="bold")
    line1 = ax1.plot(alphas, fpr_means, marker="o", color=color_fpr, linewidth=2.2, label=r"FPR $\mu \pm \sigma$")
    ax1.fill_between(alphas, fpr_means - fpr_stds, fpr_means + fpr_stds, color=color_fpr, alpha=0.18)
    ax1.tick_params(axis="y", labelcolor=color_fpr)
    ax1.grid(True, linestyle=":", alpha=0.6)

    # Secondary Axis: ILS
    ax2 = ax1.twinx()
    ax2.set_ylabel("Identity Leakage Score (ILS)", color=color_ils, fontsize=11, fontweight="bold")
    line2 = ax2.plot(alphas, ils_means, marker="s", linestyle="--", color=color_ils, linewidth=2.2, label=r"ILS $\mu \pm \sigma$")
    ax2.fill_between(alphas, ils_means - ils_stds, ils_means + ils_stds, color=color_ils, alpha=0.18)
    ax2.tick_params(axis="y", labelcolor=color_ils)

    # Combined Legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="center left", frameon=True, fontsize=9.5)

    plt.title("OMDP Dose-Response: Morphological Feature Drift & Identity Leakage", fontsize=11, pad=12)
    fig.tight_layout()

    ensure_dirs(os.path.dirname(output_path))
    plt.savefig(output_path, dpi=300)
    plt.savefig(str(Path(output_path).with_suffix(".png")), dpi=300)
    plt.close()
    logger.info(f"Saved dose-response plot: {output_path}")


def plot_fpr_drift_distributions(eval_outputs_dir: str, output_path: str):
    """Figure 2: Kernel Density / Violin distribution of FPR across alpha levels."""
    fpr_evaluator = FacialProportionRatioEvaluator()
    cohort_dirs = sorted(glob.glob(os.path.join(eval_outputs_dir, "alpha_*")))

    data_records = []
    for cdir in cohort_dirs:
        alpha_name = os.path.basename(cdir)
        images = glob.glob(os.path.join(cdir, "*.png"))
        for p in images:
            img = cv2.imread(p)
            if img is not None:
                res = fpr_evaluator.compute_fpr(img)
                if res["detected"] and res["fpr"] is not None:
                    data_records.append({"Cohort": alpha_name, "FPR": res["fpr"]})

    if not data_records:
        logger.warning("No image data available to plot FPR distributions.")
        return

    import pandas as pd
    df = pd.DataFrame(data_records)

    plt.figure(figsize=(7.0, 4.0), dpi=300)
    palette = sns.color_palette("mako", n_colors=len(df["Cohort"].unique()))
    sns.violinplot(x="Cohort", y="FPR", data=df, palette=palette, inner="quartile", cut=0)
    plt.axhline(0.25, color="gray", linestyle="--", alpha=0.7, label="Clean Normative Floor")
    plt.title("Progressive Morphological Feature Drift under OMDP Contamination", fontsize=11)
    plt.xlabel(r"Diffusion Model Poisoning Tier ($\alpha$)", fontsize=10, fontweight="bold")
    plt.ylabel("Facial Proportion Ratio (FPR)", fontsize=10, fontweight="bold")
    plt.legend(loc="upper right", fontsize=9)
    plt.tight_layout()

    ensure_dirs(os.path.dirname(output_path))
    plt.savefig(output_path, dpi=300)
    plt.savefig(str(Path(output_path).with_suffix(".png")), dpi=300)
    plt.close()
    logger.info(f"Saved FPR distribution plot: {output_path}")


def draw_landmark_hulls(image_bgr: np.ndarray, evaluator: FacialProportionRatioEvaluator) -> np.ndarray:
    """Draws overlaid convex hulls for inner features (Red) and cranial boundary (Cyan)."""
    annotated = image_bgr.copy()
    h, w, _ = image_bgr.shape
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    res = evaluator.face_mesh.process(image_rgb)

    if not res.multi_face_landmarks:
        return annotated

    face_lm = res.multi_face_landmarks[0]
    points = np.zeros((len(face_lm.landmark), 2), dtype=np.float32)
    for i, lm in enumerate(face_lm.landmark):
        points[i] = [lm.x * w, lm.y * h]

    inner_pts = points[INNER_FACE_INDICES]
    oval_pts = points[FACE_OVAL_INDICES]

    try:
        # Cranial Hull (Cyan)
        cranial_hull = ConvexHull(oval_pts)
        cranial_poly = oval_pts[cranial_hull.vertices].astype(np.int32)
        cv2.polylines(annotated, [cranial_poly], isClosed=True, color=(255, 255, 0), thickness=2)

        # Inner Hull (Red)
        inner_hull = ConvexHull(inner_pts)
        inner_poly = inner_pts[inner_hull.vertices].astype(np.int32)
        cv2.polylines(annotated, [inner_poly], isClosed=True, color=(0, 0, 255), thickness=2)

        # Draw landmark points
        for pt in inner_pts[::2]:
            cv2.circle(annotated, (int(pt[0]), int(pt[1])), 1, (0, 165, 255), -1)

    except Exception:
        pass

    return annotated


def plot_qualitative_grid(
    clean_sample_path: str,
    poisoned_sample_path: str,
    gen_samples: Dict[str, str],
    output_path: str
):
    """Figure 3: Qualitative comparison grid illustrating morphological contraction with landmark hulls."""
    evaluator = FacialProportionRatioEvaluator()

    panels = [
        ("Clean Baseline", clean_sample_path),
        ("Poisoned (Kirkified)", poisoned_sample_path),
    ]
    for tag, p in gen_samples.items():
        panels.append((f"Gen ({tag})", p))

    valid_panels = [(title, p) for title, p in panels if p and os.path.exists(p)]
    if len(valid_panels) < 2:
        logger.warning("Insufficient images for qualitative grid.")
        return

    n = len(valid_panels)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.6), dpi=300)
    if n == 1:
        axes = [axes]

    for idx, (title, path) in enumerate(valid_panels):
        img_bgr = cv2.imread(path)
        annotated_bgr = draw_landmark_hulls(img_bgr, evaluator)
        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)

        axes[idx].imshow(annotated_rgb)
        axes[idx].set_title(title, fontsize=10, fontweight="bold")
        axes[idx].axis("off")

    plt.suptitle("Geometric Morphological Landmark Decomposition (Cyan: Cranial, Red: Inner)", fontsize=11, y=0.98)
    plt.tight_layout()

    ensure_dirs(os.path.dirname(output_path))
    plt.savefig(output_path, dpi=300)
    plt.savefig(str(Path(output_path).with_suffix(".png")), dpi=300)
    plt.close()
    logger.info(f"Saved qualitative grid: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="07_plot_results: Camera-ready figure generator")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--summary_json", type=str, default=None, help="Path to evaluation_summary.json")
    parser.add_argument("--eval_dir", type=str, default=None, help="Directory containing eval_outputs")
    parser.add_argument("--figures_dir", type=str, default=None, help="Directory to save figures")
    args = parser.parse_args()

    config = load_config(args.config)
    paths = config.get("paths", {})

    summary_file = args.summary_json or os.path.join(paths.get("eval_outputs_dir", "eval_outputs"), "evaluation_summary.json")
    eval_dir = args.eval_dir or paths.get("eval_outputs_dir", "eval_outputs")
    figures_dir = args.figures_dir or paths.get("figures_dir", "figures")

    set_academic_plot_style()
    ensure_dirs(figures_dir)

    # 1. Dose-Response Plot
    if os.path.exists(summary_file):
        with open(summary_file, "r", encoding="utf-8") as f:
            summary_data = json.load(f)
        plot_dose_response(summary_data, os.path.join(figures_dir, "figure1_dose_response.pdf"))
    else:
        logger.warning(f"Summary JSON not found at {summary_file}. Skipping Dose-Response plot.")

    # 2. FPR Distributions
    plot_fpr_drift_distributions(eval_dir, os.path.join(figures_dir, "figure2_fpr_drift_distributions.pdf"))

    # 3. Qualitative Grid (Look for sample images)
    clean_sample = next(iter(glob.glob(os.path.join(paths.get("clean_dir", "data/clean"), "*.png"))), "")
    poisoned_sample = next(iter(glob.glob(os.path.join(paths.get("poisoned_dir", "data/poisoned"), "*.png"))), "")
    gen_samples = {
        "alpha_0.00": next(iter(glob.glob(os.path.join(eval_dir, "alpha_0.00", "*.png"))), ""),
        "alpha_0.05": next(iter(glob.glob(os.path.join(eval_dir, "alpha_0.05", "*.png"))), ""),
        "alpha_0.25": next(iter(glob.glob(os.path.join(eval_dir, "alpha_0.25", "*.png"))), ""),
    }
    plot_qualitative_grid(
        clean_sample,
        poisoned_sample,
        gen_samples,
        os.path.join(figures_dir, "figure3_qualitative_drift_grid.pdf"),
    )


if __name__ == "__main__":
    main()
