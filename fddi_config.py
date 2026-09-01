"""Canonical settings used for the FDDI paper experiments."""


FDDI_PAPER_CONFIG = {
    # Diffusion and guidance settings used by the released demo.
    "num_inference_steps": 40,
    "guidance_scale": 5.0,
    # A cache update every 20 denoising steps gives the paper's 1:20 schedule.
    "cache_interval": 20,
    "cache_layer_id": 0,
    "cache_block_id": 0,
    "uniform": True,
    # FDDI refinement settings.
    "lpf_threshold": 200,
    "lpf_kernel": 3,
    "lpf_sigma": 0.8,
    "alpha": 0.5,
    "skip_boost_factor": 1.4,
}
