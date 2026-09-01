# FDDI Paper Reproduction

This document records the implementation settings and reported results for
the paper **Frequency-Decoupled Detail Injection (FDDI) for Zero-Shot
Industrial Defect Generation**.

## Authors

- Shufan Zhou
- Mingjie Sun
- School of Computer Science and Technology, Soochow University, China

The source paper is written in Chinese as
`面向零样本工业缺陷生成的频率解耦细节注入加速网络`.

## Method Correspondence

The released implementation follows the paper's two FDDI streams:

1. **Masked LPF / structure stream:** in the refinement window, cached U-Net
   features are Gaussian-smoothed inside the edit mask and blended with the
   original cached features using `alpha`.
2. **Skip Connect Booster / texture stream:** fresh encoder skip features are
   amplified inside the edit mask before decoding.

The refinement condition is strictly `t < 200` for the 1,000-step training
schedule. Cross-attention remains dense so that reference-image and prompt
conditioning are preserved.

The repository also contains DynaMask sparse self-attention in the accelerated
path. DynaMask is an implementation-level runtime optimization and was not
reported as a separate component in the paper's ablation table. The reported
FDDI numbers should therefore be treated as the paper's experimental record;
rerun the benchmark before making a new claim about a modified code version.

## Canonical Configuration

The single source of truth is [`fddi_config.py`](../fddi_config.py). The
paper-aligned defaults are:

| Setting | Value |
| --- | --- |
| Base model | Stable Diffusion v1.5 inpainting |
| Scheduler | DDIM |
| Inference steps | 40 |
| DeepCache schedule | 1:20, uniform cache updates |
| Cache layer / block | 0 / 0 |
| Refinement threshold | `t < 200` |
| LPF kernel | 3 x 3 Gaussian |
| LPF sigma | 0.8 |
| LPF blend alpha | 0.5 |
| Skip Connect Booster | 1.4 inside the mask |
| Guidance scale used by the demo | 5.0 |
| Evaluation GPU | NVIDIA RTX 5060 Ti, 16 GB |
| Datasets | MVTec AD and VisA |
| Metrics | FID, Style Loss, LPIPS |

The paper describes the LPF operation as low-pass replacement. The released
code uses the explicit blend
`(1 - alpha) * cached + alpha * blurred_cached`, with `alpha=0.5`; this
implementation detail is recorded here so that code and documentation agree.

## Reported Results

The following values are transcribed from the paper. Lower is better for all
three quality metrics and for time.

| Method | MVTec FID | MVTec Style | MVTec LPIPS | MVTec time (s) | VisA FID | VisA Style | VisA LPIPS | VisA time (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Original | 55.36 | 81.69 | 0.4041 | 8.68 | 43.11 | 188.90 | 0.1476 | 8.68 |
| DeepCache | 65.56 | 110.68 | 0.4238 | 5.66 | 47.78 | 206.72 | 0.1575 | 5.66 |
| LPF only | 65.54 | 109.94 | 0.4229 | 5.66 | 47.22 | 207.98 | 0.1571 | 5.66 |
| Booster only | 62.52 | 81.45 | 0.4190 | 5.65 | 45.50 | 178.98 | 0.1514 | 5.66 |
| **FDDI (both)** | **61.47** | **81.99** | **0.4191** | **5.66** | **45.17** | **179.63** | **0.1513** | **5.66** |

The measured 8.68 s versus 5.66 s corresponds to approximately 1.5x speedup.
The paper contains both 1.54x and 1.56x wording in different sections; this
repository avoids presenting those two values as a single independently
reproduced number.

## Complexity Record

The paper reports the following GFLOPs values:

| Component | GFLOPs |
| --- | ---: |
| ReferenceNet, once | 27.79 |
| Original U-Net | 109.23 |
| DeepCache U-Net | 81.44 |
| FDDI U-Net | 82.73 |
| FDDI extra over DeepCache | 1.29 (1.59%) |

The paper labels the original U-Net FLOPs row as “50 iters”, while the main
generation setting is 40 steps. Preserve that distinction when reproducing or
updating the complexity table.

## Reproduction Procedure

1. Prepare the external model, dataset, cache, and result directories using
   [`docs/ASSET_LAYOUT.md`](ASSET_LAYOUT.md).
2. Activate the tested Conda environment and install
   [`requirements.txt`](../requirements.txt).
3. Run `python scripts/check_assets.py`.
4. Run `python scripts/smoke_inference.py --seed 42` on a CUDA GPU. The default
   is now the paper's 40-step configuration.
5. Use `./run_fddi.sh` for the Gradio demo. The default UI values are also
   40 steps and guidance scale 5.0.

The smoke test checks end-to-end loading, DynaMask installation, and image
generation. It is not a replacement for the MVTec/VisA metric benchmark.
Timing and quality numbers must be recomputed on the target GPU with the same
data split, image count, seed policy, and metric implementation used by the
paper.

## Limitations

The paper notes that a fixed Gaussian LPF may harm large, periodic structural
defects. The booster can also over-emphasize shallow features for some images.
Adaptive frequency filters and a more conservative, data-dependent booster
are natural follow-up directions.
