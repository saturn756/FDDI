#!/usr/bin/env python3
"""Validate the external FDDI runtime asset layout."""

from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ASSET_ROOT = Path(os.environ.get("FDDI_ASSET_ROOT", PROJECT_ROOT.parent / "FDDI_assets")).expanduser().resolve()

REQUIRED_PATHS = {
    "MimicBrush checkpoint": ASSET_ROOT / "models/modelscope/xichen/MimicBrush/mimicbrush/mimicbrush.bin",
    "MimicBrush VAE": ASSET_ROOT / "models/modelscope/xichen/MimicBrush/sd-vae-ft-mse",
    "MimicBrush image encoder": ASSET_ROOT / "models/modelscope/xichen/MimicBrush/image_encoder",
    "Depth Anything checkpoint": ASSET_ROOT / "models/modelscope/xichen/MimicBrush/depth_model/depth_anything_vitb14.pth",
    "SD inpainting model": ASSET_ROOT / "models/modelscope/xichen/cleansd/stable-diffusion-inpainting",
    "SD reference model": ASSET_ROOT / "models/modelscope/xichen/cleansd/stable-diffusion-v1-5",
    "VisA dataset": ASSET_ROOT / "datasets/VisA",
    "KolektorSDD2 dataset": ASSET_ROOT / "datasets/KolektorSDD2",
    "MVTec results": ASSET_ROOT / "results/mvtec_new",
    "VisA results": ASSET_ROOT / "results/VisA",
}


def main() -> int:
    print(f"FDDI_ASSET_ROOT={ASSET_ROOT}")
    missing = []
    for label, path in REQUIRED_PATHS.items():
        if path.exists():
            print(f"[OK]   {label}: {path}")
        else:
            print(f"[MISS] {label}: {path}")
            missing.append(label)
    if missing:
        print(f"Missing {len(missing)} required asset(s).")
        return 1
    print("All required FDDI assets are available.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
