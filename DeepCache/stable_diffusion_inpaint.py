#!/usr/bin/env python3
"""Compare standard Stable Diffusion inpainting with DeepCache.

Example:
    python DeepCache/stable_diffusion_inpaint.py \
        --image path/to/image.png \
        --mask path/to/mask.png \
        --prompt "a photo of a scratch"
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import torch
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image
from torchvision.utils import save_image

try:
    from DeepCache.DeepCache.sd_inpainting.pipeline_stable_diffusion_inpainting import (
        StableDiffusionInpaintPipeline as DeepCacheStableDiffusionInpaintingPipeline,
    )
except ModuleNotFoundError:
    # This branch supports running the script from inside the DeepCache folder.
    from DeepCache.sd_inpainting.pipeline_stable_diffusion_inpainting import (
        StableDiffusionInpaintPipeline as DeepCacheStableDiffusionInpaintingPipeline,
    )


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def set_random_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_input(path: Path, mode: str) -> Image.Image:
    return Image.open(path).convert(mode).resize((512, 512), Image.Resampling.LANCZOS)


def run_pipeline(pipe, prompt: str, image: Image.Image, mask: Image.Image, steps: int, seed: int):
    set_random_seed(seed)
    return pipe(
        prompt,
        image,
        mask,
        num_inference_steps=steps,
        output_type="pt",
        return_dict=True,
    ).images


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True, help="Input RGB image")
    parser.add_argument("--mask", type=Path, required=True, help="Grayscale inpainting mask")
    parser.add_argument("--prompt", default="a photo of a scratch")
    parser.add_argument("--model", default="runwayml/stable-diffusion-inpainting")
    parser.add_argument("--output", type=Path, default=Path("output1.png"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument("--warmup", type=int, default=2)
    args = parser.parse_args()

    if args.steps < 1 or args.warmup < 0:
        parser.error("--steps must be positive and --warmup cannot be negative")
    if not args.image.is_file():
        parser.error(f"Image not found: {args.image}")
    if not args.mask.is_file():
        parser.error(f"Mask not found: {args.mask}")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    dtype = torch.float16 if args.device.startswith("cuda") else torch.float32
    image = load_input(args.image, "RGB")
    mask = load_input(args.mask, "L")

    baseline_pipe = StableDiffusionInpaintPipeline.from_pretrained(
        args.model, torch_dtype=dtype
    ).to(args.device)
    logging.info("Warming up baseline (%d iterations)...", args.warmup)
    for _ in range(args.warmup):
        run_pipeline(baseline_pipe, args.prompt, image, mask, args.steps, args.seed)

    logging.info("Running baseline...")
    start = time.perf_counter()
    baseline_output = run_pipeline(
        baseline_pipe, args.prompt, image, mask, args.steps, args.seed
    )
    logging.info("Baseline: %.2f seconds", time.perf_counter() - start)
    del baseline_pipe
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()

    deepcache_pipe = DeepCacheStableDiffusionInpaintingPipeline.from_pretrained(
        args.model, torch_dtype=dtype
    ).to(args.device)
    logging.info("Warming up DeepCache (%d iterations)...", args.warmup)
    for _ in range(args.warmup):
        run_pipeline(deepcache_pipe, args.prompt, image, mask, args.steps, args.seed)

    logging.info("Running DeepCache...")
    start = time.perf_counter()
    deepcache_output = deepcache_pipe(
        args.prompt,
        image,
        mask,
        num_inference_steps=args.steps,
        cache_interval=5,
        cache_layer_id=0,
        cache_block_id=0,
        uniform=False,
        pow=1.4,
        center=15,
        output_type="pt",
        return_dict=True,
    ).images
    logging.info("DeepCache: %.2f seconds", time.perf_counter() - start)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_image(torch.cat((baseline_output, deepcache_output)), str(args.output), nrow=2)
    logging.info("Saved baseline and DeepCache comparison to %s", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
