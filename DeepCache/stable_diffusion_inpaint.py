import os
import time
import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

import torch
from torchvision.utils import save_image

import argparse

from DeepCache.sd_inpainting.pipeline_stable_diffusion_inpainting import StableDiffusionInpaintPipeline as DeepCacheStableDiffusionInpaintingPipeline
from diffusers import StableDiffusionInpaintPipeline
from PIL import Image
def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

if __name__ == "__main__":
    
    model = "runwayml/stable-diffusion-inpainting"
    seed = 47
    prompt = "a photo of a scratch"
    image = Image.open("/home/zsf/ELITE_INPATING/datasets/mvtec/bottle/test/broken_large/000.png").convert("RGB").resize((512, 512))
    mask_image = Image.open("/home/zsf/ELITE_INPATING/datasets/mvtec/bottle/ground_truth/broken_large/001_mask.png").convert("L").resize((512, 512))  # 灰度图

    baseline_pipe = StableDiffusionInpaintPipeline.from_pretrained(model, torch_dtype=torch.float16).to("cuda:0")
    # Warmup GPU. Only for testing the speed.
    logging.info("Warming up GPU...")
    for _ in range(2):
        set_random_seed(seed)
        _ = baseline_pipe(
            prompt, 
            image,
            mask_image,
            output_type='pt'
            ).images
        
    # Baseline
    logging.info("Running baseline...")
    start_time = time.time()
    set_random_seed(seed)
    ori_output = baseline_pipe(
        prompt, 
        image,
        mask_image,
        output_type='pt'
        ).images
    use_time = time.time() - start_time
    logging.info("Baseline: {:.2f} seconds".format(use_time))
    #save_image(image_ori[0], "{}_{:.2f}.png".format(prompt, use_time))
    del baseline_pipe
    torch.cuda.empty_cache()

    # DeepCache
    pipe = DeepCacheStableDiffusionInpaintingPipeline.from_pretrained(model, torch_dtype=torch.float16).to("cuda:0")
    # Warmup GPU. Only for testing the speed.
    logging.info("Warming up GPU...")
    for _ in range(2):
        set_random_seed(seed)
        _ = pipe(
            prompt, 
            image,
            mask_image,
            output_type='pt', 
            return_dict=True).images

    logging.info("Running DeepCache...")
    set_random_seed(seed)
    start_time = time.time()
    deepcache_output = pipe(
        prompt, 
        image,
        mask_image,
        cache_interval=5, cache_layer_id=0, cache_block_id=0,
        uniform=False, pow=1.4, center=15,
        output_type='pt', return_dict=True
    ).images
    use_time = time.time() - start_time
    logging.info("DeepCache: {:.2f} seconds".format(use_time))

    save_image([ori_output[0], deepcache_output[0]], "output1.png")
    logging.info("Saved to output.png. Done!")




