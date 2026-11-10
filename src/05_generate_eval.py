#!/usr/bin/env python3
"""
05_generate_eval.py
Batch Inference & Morphological Evaluation Pipeline for OMDP Research.

Generates neutral portraits from base model + trained LoRAs across contamination ratios
(alpha in [0.0, 0.02, 0.05, 0.10, 0.25]), measuring FPR drift and Identity Leakage (ILS).
"""

import os
import sys
import json
import csv
import argparse
from pathlib import Path
from typing import Dict, Any, List

import torch
from diffusers import StableDiffusionPipeline

# Add parent directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.utils.common import load_config, setup_logger, ensure_dirs, get_device
from src.metrics import ComprehensiveMetricsPipeline

logger = setup_logger("GenerateEval")


def generate_cohort(
    pipeline: StableDiffusionPipeline,
    prompts: List[str],
    seeds: List[int],
    output_dir: str,
    guidance_scale: float = 7.5,
    num_inference_steps: int = 30,
) -> List[str]:
    """Generates images across a grid of prompts and deterministic seeds."""
    ensure_dirs(output_dir)
    generated_paths = []
    device = pipeline.device

    for p_idx, prompt in enumerate(prompts):
        for seed in seeds:
            generator = torch.Generator(device=device).manual_seed(seed)
            out_filename = f"gen_prompt{p_idx:02d}_seed{seed}.png"
            out_filepath = os.path.join(output_dir, out_filename)

            # Generate image
            with torch.inference_mode():
                image = pipeline(
                    prompt=prompt,
                    guidance_scale=guidance_scale,
                    num_inference_steps=num_inference_steps,
                    generator=generator,
                ).images[0]

            image.save(out_filepath)
            generated_paths.append(out_filepath)

    return generated_paths


def run_evaluation_across_alphas(config: Dict[str, Any], dry_run: bool = False) -> Dict[str, Any]:
    """Evaluates all trained LoRA models against neutral prompt benchmarks."""
    paths = config.get("paths", {})
    train_cfg = config.get("training", {})
    eval_cfg = config.get("evaluation", {})
    mix_cfg = config.get("mixing", {})

    base_model = train_cfg.get("base_model", "runwayml/stable-diffusion-v1-5")
    checkpoints_dir = paths.get("checkpoints_dir", "checkpoints")
    eval_outputs_dir = paths.get("eval_outputs_dir", "eval_outputs")
    clean_dir = paths.get("clean_dir", "data/clean")
    alpha_levels = mix_cfg.get("alpha_levels", [0.0, 0.02, 0.05, 0.10, 0.25])

    prompts = eval_cfg.get("neutral_prompts", [
        "a crisp professional portrait photograph of a political speaker, sharp focus"
    ])
    seeds = eval_cfg.get("seeds", [42, 101, 202])
    guidance_scale = eval_cfg.get("guidance_scale", 7.5)
    num_steps = eval_cfg.get("num_inference_steps", 30)

    device = get_device(config.get("project", {}).get("device", "auto"))
    ensure_dirs(eval_outputs_dir)

    # Initialize quantitative metrics evaluator
    metrics_pipeline = ComprehensiveMetricsPipeline(config)

    overall_results: Dict[str, Any] = {
        "base_model": base_model,
        "alpha_experiments": {},
    }

    if not dry_run:
        logger.info(f"Loading base StableDiffusionPipeline: {base_model} onto {device}...")
        dtype = torch.float16 if device.type == "cuda" else torch.float32
        pipe = StableDiffusionPipeline.from_pretrained(base_model, torch_dtype=dtype).to(device)
        pipe.safety_checker = None  # Disable safety checker for scientific reproducibility
    else:
        pipe = None
        logger.info("Running in dry-run mode (skipping actual neural diffusion generation).")

    csv_rows = []

    for alpha in alpha_levels:
        alpha_tag = f"alpha_{alpha:.2f}"
        ckpt_dir = os.path.join(checkpoints_dir, f"lora_{alpha_tag}")
        alpha_out_dir = os.path.join(eval_outputs_dir, alpha_tag)
        ensure_dirs(alpha_out_dir)

        logger.info(f"=== Evaluating Cohort: {alpha_tag} ===")

        if not dry_run and pipe is not None:
            # Unload any previously loaded LoRA weights
            try:
                pipe.unload_lora_weights()
            except Exception:
                pass

            if os.path.exists(ckpt_dir):
                logger.info(f"Loading LoRA weights from: {ckpt_dir}")
                pipe.load_lora_weights(ckpt_dir)
            else:
                logger.warning(f"No checkpoint found at {ckpt_dir}. Generating from base model.")

            generate_cohort(
                pipeline=pipe,
                prompts=prompts,
                seeds=seeds,
                output_dir=alpha_out_dir,
                guidance_scale=guidance_scale,
                num_inference_steps=num_steps,
            )

        # Run quantitative metrics (FPR, ILS, FID)
        metrics = metrics_pipeline.evaluate_directory(alpha_out_dir, ref_dir=clean_dir)
        overall_results["alpha_experiments"][alpha_tag] = {
            "alpha": alpha,
            "fpr_mean": metrics["fpr"]["mean"],
            "fpr_std": metrics["fpr"]["std"],
            "fpr_median": metrics["fpr"]["median"],
            "fpr_ci95": metrics["fpr"]["ci95"],
            "ils_mean": metrics["ils"]["mean"],
            "ils_std": metrics["ils"]["std"],
            "fid": metrics.get("fid"),
            "faces_detected": metrics["faces_detected"],
            "total_images": metrics["total_images"],
        }

        csv_rows.append({
            "alpha": alpha,
            "fpr_mean": metrics["fpr"]["mean"],
            "fpr_std": metrics["fpr"]["std"],
            "ils_mean": metrics["ils"]["mean"],
            "ils_std": metrics["ils"]["std"],
            "fid": metrics.get("fid"),
            "detected": metrics["faces_detected"],
        })

    # Save summary JSON
    summary_json = os.path.join(eval_outputs_dir, "evaluation_summary.json")
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(overall_results, f, indent=2)
    logger.info(f"Summary evaluation JSON saved to: {summary_json}")

    # Save summary CSV
    summary_csv = os.path.join(eval_outputs_dir, "evaluation_summary.csv")
    with open(summary_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["alpha", "fpr_mean", "fpr_std", "ils_mean", "ils_std", "fid", "detected"])
        writer.writeheader()
        writer.writerows(csv_rows)
    logger.info(f"Summary evaluation CSV saved to: {summary_csv}")

    return overall_results


def main():
    parser = argparse.ArgumentParser(description="05_generate_eval: Batch generation and evaluation")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--dry_run", action="store_true", help="Perform metrics check on existing images without re-diffusion")
    args = parser.parse_args()

    config = load_config(args.config)
    run_evaluation_across_alphas(config, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
