# FDDI: Fast MimicBrush with DeepCache

FDDI is a research implementation built directly on the original
[MimicBrush](https://github.com/ali-vilab/MimicBrush) inference code. It adds
DeepCache acceleration and mask-aware dynamic self-attention while preserving
the reference-image imitation pipeline.

The pipeline accepts a source image, a reference image, and a user mask. It
uses Depth Anything and a depth guider, injects reference features through
ReferenceNet, and runs either the original MimicBrush denoiser or the
DeepCache variant.

## Repository layout

```text
FDDI/
  app.py                         Gradio demo
  models/                        MimicBrush and ReferenceNet pipelines
  mimicbrush/                    Reference feature and generation wrapper
  DeepCache/                     DeepCache and DynaMask implementation
  depthanything/                 Depth Anything inference code
  scripts/check_assets.py        Runtime asset validation
  docs/ASSET_LAYOUT.md           External asset layout
```

Model weights, complete datasets, temporary caches, and bulk-generated
results are deliberately stored outside this repository. See
`docs/ASSET_LAYOUT.md`.

## Environment

The tested server environment is `mimicbrush` with Python 3.10, PyTorch 2.8,
Diffusers 0.24, Transformers 4.26, and CUDA available.

```bash
conda activate mimicbrush
pip install -r requirements.txt
```

ModelScope is optional when using the external local assets. To enable the
automatic online model download fallback, install `requirements-modelscope.txt`
as well.

## Runtime assets

Set `FDDI_ASSET_ROOT` to the directory containing `models/`, `datasets/`,
`results/`, and `cache/`. When the asset directory is next to the repository,
the launcher discovers it automatically.

```bash
export FDDI_ASSET_ROOT=/path/to/FDDI_assets
python scripts/check_assets.py
```

The research server keeps these assets in a separate directory. Existing
legacy benchmark paths can be mapped to the same directory with symlinks, but
no server-specific path is required by this repository.

## Run the demo

```bash
./run_fddi.sh
```

The demo listens on `0.0.0.0` and uses DeepCache by default. The UI button
switches between the accelerated and original pipelines.

## Smoke test

With the local runtime assets and a free CUDA device, run one deterministic
generation without starting the Gradio interface:

```bash
python scripts/check_assets.py
python scripts/smoke_inference.py --steps 1 --seed 42
```

The generated image is written under the external asset directory.

## Reproducibility notes

- Baseline: original MimicBrush pipeline with the same model and scheduler.
- Accelerated method: DeepCache U-Net plus DynaMask self-attention.
- The acceleration is applied during later denoising steps; cross-attention
  remains full-resolution so reference and prompt information is preserved.
- Experimental MVTec and VisA outputs are kept under the external `results/`
  directory and are not committed to GitHub.

## Acknowledgements

This project is based on MimicBrush, DeepCache, Diffusers, and Depth Anything.
Please retain their original licenses and citations when redistributing this
repository.

## License

Apache License 2.0. See the upstream licenses included in the source tree.
