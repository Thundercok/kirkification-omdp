#!/usr/bin/env python3
"""
04_train_lora.py
HuggingFace Diffusers LoRA Fine-Tuning Wrapper for Stable Diffusion 1.5 / SDXL.

Injects Low-Rank Adaptation (LoRA) matrices into UNet cross- and self-attention
layers to model morphological feature drift under controlled poisoning ratios.
"""

import os
import sys
import json
import math
import argparse
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm.auto import tqdm

# HuggingFace & Diffusers imports
from transformers import AutoTokenizer, CLIPTextModel
from diffusers import (
    AutoencoderKL,
    UNet2DConditionModel,
    DDPMScheduler,
)
from peft import LoraConfig, get_peft_model

# Add parent directory to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.utils.common import load_config, setup_logger, ensure_dirs, set_seed, get_device

logger = setup_logger("TrainLoRA")


class DiffusionDataset(Dataset):
    """Loads image-caption pairs from metadata.jsonl for diffusion training."""

    def __init__(self, data_dir: str, tokenizer: Any, resolution: int = 512):
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.resolution = resolution
        self.entries = []

        jsonl_path = os.path.join(data_dir, "metadata.jsonl")
        if not os.path.exists(jsonl_path):
            raise FileNotFoundError(f"metadata.jsonl not found in: {data_dir}")

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.entries.append(json.loads(line))

        self.transform = transforms.Compose([
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(resolution),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),  # Scales from [0, 1] to [-1, 1]
        ])

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        entry = self.entries[idx]
        img_path = os.path.join(self.data_dir, entry["file_name"])

        try:
            image = Image.open(img_path).convert("RGB")
            pixel_values = self.transform(image)
        except Exception as e:
            logger.warning(f"Error loading {img_path}: {e}. Replacing with zero tensor.")
            pixel_values = torch.zeros((3, self.resolution, self.resolution))

        caption = entry.get("text", "a professional portrait photograph")
        inputs = self.tokenizer(
            caption,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        return {
            "pixel_values": pixel_values,
            "input_ids": inputs.input_ids.squeeze(0),
        }


def train_lora(
    data_dir: str,
    output_dir: str,
    config: Dict[str, Any],
    max_steps_override: Optional[int] = None,
) -> None:
    """Executes the LoRA training loop on the target mixed dataset."""
    train_cfg = config.get("training", {})
    base_model = train_cfg.get("base_model", "runwayml/stable-diffusion-v1-5")
    resolution = train_cfg.get("resolution", 512)
    batch_size = train_cfg.get("train_batch_size", 2)
    grad_accum_steps = train_cfg.get("gradient_accumulation_steps", 2)
    learning_rate = float(train_cfg.get("learning_rate", 1.0e-4))
    max_train_steps = max_steps_override or train_cfg.get("max_train_steps", 800)
    lora_rank = train_cfg.get("lora_rank", 16)
    lora_alpha = train_cfg.get("lora_alpha", 32)
    lora_dropout = train_cfg.get("lora_dropout", 0.05)
    target_modules = train_cfg.get("target_modules", ["to_k", "to_q", "to_v", "to_out.0"])

    device = get_device(config.get("project", {}).get("device", "auto"))
    ensure_dirs(output_dir)

    logger.info(f"Target Device: {device} | Base Model: {base_model}")
    logger.info(f"Training LoRA rank={lora_rank}, alpha={lora_alpha} on '{data_dir}' -> '{output_dir}'")

    # 1. Load Tokenizer & Scheduler
    tokenizer = AutoTokenizer.from_pretrained(base_model, subfolder="tokenizer")
    noise_scheduler = DDPMScheduler.from_pretrained(base_model, subfolder="scheduler")

    # 2. Load Models
    text_encoder = CLIPTextModel.from_pretrained(base_model, subfolder="text_encoder").to(device)
    vae = AutoencoderKL.from_pretrained(base_model, subfolder="vae").to(device)
    unet = UNet2DConditionModel.from_pretrained(base_model, subfolder="unet").to(device)

    # Freeze VAE & Text Encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)

    # 3. Setup LoRA on UNet
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        init_lora_weights="gaussian",
        target_modules=target_modules,
        lora_dropout=lora_dropout,
    )
    unet = get_peft_model(unet, lora_config)
    unet.print_trainable_parameters()

    # 4. Dataset & Dataloader
    dataset = DiffusionDataset(data_dir=data_dir, tokenizer=tokenizer, resolution=resolution)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # 5. Optimizer
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, unet.parameters()),
        lr=learning_rate,
        weight_decay=1e-2,
    )

    # 6. Training Loop
    global_step = 0
    progress_bar = tqdm(total=max_train_steps, desc="LoRA Training Steps")
    unet.train()

    scaling_factor = 0.18215  # VAE latent scaling factor for SD 1.5

    while global_step < max_train_steps:
        for batch in dataloader:
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)

            with torch.no_grad():
                # Encode pixels into latent space
                latents = vae.encode(pixel_values).latent_dist.sample()
                latents = latents * scaling_factor

                # Sample noise and random timesteps
                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=device
                ).long()

                # Forward diffusion step (add noise to latents)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get text condition embeddings
                encoder_hidden_states = text_encoder(input_ids)[0]

            # Predict the noise residual
            model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

            # Compute MSE Loss
            loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
            loss = loss / grad_accum_steps
            loss.backward()

            if (global_step + 1) % grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

            global_step += 1
            progress_bar.update(1)
            progress_bar.set_postfix({"loss": f"{loss.item() * grad_accum_steps:.4f}"})

            if global_step >= max_train_steps:
                break

    progress_bar.close()

    # 7. Save standalone LoRA weights
    logger.info(f"Saving fine-tuned LoRA weights to: {output_dir}")
    unet.save_pretrained(output_dir)

    # Save training metadata
    train_metadata = {
        "base_model": base_model,
        "max_train_steps": max_train_steps,
        "learning_rate": learning_rate,
        "lora_rank": lora_rank,
        "lora_alpha": lora_alpha,
        "data_dir": data_dir,
    }
    with open(os.path.join(output_dir, "training_config.json"), "w", encoding="utf-8") as f:
        json.dump(train_metadata, f, indent=2)

    logger.info("LoRA training successfully finished.")


def main():
    parser = argparse.ArgumentParser(description="04_train_lora: Train LoRA model for OMDP analysis")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to mixed dataset directory with metadata.jsonl")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save LoRA checkpoints")
    parser.add_argument("--max_steps", type=int, default=None, help="Override maximum training steps")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    args = parser.parse_args()

    config = load_config(args.config)
    seed = args.seed if args.seed is not None else config.get("project", {}).get("seed", 42)
    set_seed(seed)

    train_lora(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        config=config,
        max_steps_override=args.max_steps,
    )


if __name__ == "__main__":
    main()
