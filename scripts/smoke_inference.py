#!/usr/bin/env python3
"""Run one deterministic FDDI generation using the local runtime assets."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw
from torchvision.transforms import Compose


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "depthanything"))

from diffusers import AutoencoderKL, DDIMScheduler  # noqa: E402
from diffusers.image_processor import VaeImageProcessor  # noqa: E402

from DeepCache.DeepCache.sd.apply_dyna import inject_dynamask_processor  # noqa: E402
from DeepCache.DeepCache.sd.unet_2d_condition import (  # noqa: E402
    UNet2DConditionModel as DeepCacheUNet2DConditionModel,
)
from depthanything.depth_anything.util.transform import (  # noqa: E402
    NormalizeImage,
    PrepareForNet,
    Resize,
)
from depthanything.fast_import import depth_anything_model  # noqa: E402
from mimicbrush import MimicBrush_RefNet  # noqa: E402
from models.ReferenceNet import ReferenceNet  # noqa: E402
from models.depth_guider import DepthGuider  # noqa: E402
from models.pipeline_deepcachemimicbrush import (  # noqa: E402
    MimicBrushPipeline as DeepCacheMimicBrushPipeline,
)


def asset_root() -> Path:
    configured = os.environ.get("FDDI_ASSET_ROOT")
    candidates = [Path(configured).expanduser()] if configured else []
    candidates.extend([PROJECT_ROOT.parent / "FDDI_assets", PROJECT_ROOT / "assets_runtime"])
    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()
    raise FileNotFoundError("Set FDDI_ASSET_ROOT to the FDDI runtime asset directory.")


def square_image(image: Image.Image) -> Image.Image:
    image = image.convert("RGB")
    side = max(image.size)
    canvas = Image.new("RGB", (side, side), "white")
    canvas.paste(image, ((side - image.width) // 2, (side - image.height) // 2))
    return canvas.resize((512, 512), Image.Resampling.LANCZOS)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("FDDI smoke inference requires CUDA; no GPU is available.")

    root = asset_root()
    model_root = root / "models/modelscope/xichen"
    mimic_root = model_root / "MimicBrush"
    sd_root = model_root / "cleansd"
    device = "cuda"
    dtype = torch.float16

    source = square_image(Image.open(PROJECT_ROOT / "examples/source/wood.png"))
    reference = square_image(Image.open(PROJECT_ROOT / "examples/reference/wood_1.png"))
    mask = Image.new("L", (512, 512), 0)
    ImageDraw.Draw(mask).rectangle((160, 160, 352, 352), fill=255)

    depth_transform = Compose([
        Resize(
            width=518,
            height=518,
            resize_target=False,
            keep_aspect_ratio=True,
            ensure_multiple_of=14,
            resize_method="lower_bound",
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
    ])
    depth_model_path = mimic_root / "depth_model/depth_anything_vitb14.pth"
    depth_anything_model.load_state_dict(torch.load(depth_model_path, map_location="cpu"))
    depth_anything_model.eval()

    scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        set_alpha_to_one=False,
        steps_offset=1,
    )
    vae = AutoencoderKL.from_pretrained(mimic_root / "sd-vae-ft-mse").to(device=device, dtype=dtype)
    unet = DeepCacheUNet2DConditionModel.from_pretrained(
        sd_root / "stable-diffusion-inpainting",
        subfolder="unet",
        in_channels=13,
        low_cpu_mem_usage=False,
        ignore_mismatched_sizes=True,
    ).to(device=device, dtype=dtype)
    inject_dynamask_processor(unet)
    pipe = DeepCacheMimicBrushPipeline.from_pretrained(
        sd_root / "stable-diffusion-inpainting",
        torch_dtype=dtype,
        scheduler=scheduler,
        vae=vae,
        unet=unet,
        feature_extractor=None,
        safety_checker=None,
        low_cpu_mem_usage=True,
    ).to(device)
    pipe.enable_attention_slicing()

    referencenet = ReferenceNet.from_pretrained(
        sd_root / "stable-diffusion-v1-5", subfolder="unet"
    ).to(device=device, dtype=dtype)
    model = MimicBrush_RefNet(
        pipe,
        mimic_root / "image_encoder",
        mimic_root / "mimicbrush/mimicbrush.bin",
        depth_anything_model,
        DepthGuider(),
        referencenet,
        device,
    )
    # Install DynaMask after all checkpoint loading is complete.
    inject_dynamask_processor(model.pipe.unet)
    dyna_count = sum(
        hasattr(processor, "set_state")
        for processor in model.pipe.unet.attn_processors.values()
    )
    if dyna_count == 0:
        raise RuntimeError("DynaMask processors were replaced during model construction.")
    print(f"DYNAMASK_PROCESSORS={dyna_count}")

    depth_input = torch.from_numpy(depth_transform({"image": np.asarray(source)})["image"])
    depth_input = depth_input.unsqueeze(0) / 255
    mask_input = VaeImageProcessor(
        vae_scale_factor=1,
        do_normalize=False,
        do_binarize=True,
        do_convert_grayscale=True,
    ).preprocess(mask, height=512, width=512)

    with torch.inference_mode():
        generated, _ = model.generate(
            pil_image=reference,
            depth_image=depth_input,
            num_samples=1,
            num_inference_steps=args.steps,
            seed=args.seed,
            image=source,
            mask_image=mask_input,
            strength=1.0,
            guidance_scale=5.0,
            interval_step=2,
            lpf_threshold=200,
            lpf_sigma=0.8,
            lpf_kernel=3,
            alpha=0.5,
            skip_boost_factor=1.4,
        )

    output = args.output or root / "results/smoke_test/fddi_deepcache_smoke.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    generated[0].save(output)
    print(f"FDDI_SMOKE_OK output={output} size={output.stat().st_size} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
