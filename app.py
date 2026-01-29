import gradio as gr
import torch
import torch.nn.functional as F
from safetensors.numpy import save_file, load_file
from omegaconf import OmegaConf
from transformers import AutoConfig
import cv2
from PIL import Image
import numpy as np
import json
import os
#
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionInpaintPipelineLegacy, StableDiffusionInpaintPipeline, DDIMScheduler, AutoencoderKL
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel, DDIMScheduler
from diffusers import DDIMScheduler, DDPMScheduler, DPMSolverMultistepScheduler
from diffusers.image_processor import VaeImageProcessor
#
from models.pipeline_mimicbrush import MimicBrushPipeline

from DeepCache.DeepCache.sd.unet_2d_condition import UNet2DConditionModel as UNet2DConditionModel_DC
from models.pipeline_deepcachemimicbrush import MimicBrushPipeline as MimicBrushPipeline_DC

from models.ReferenceNet import ReferenceNet
from models.depth_guider import DepthGuider
from mimicbrush import MimicBrush_RefNet
from dataset.data_utils import *
from modelscope.hub.snapshot_download import snapshot_download


# === import Depth Anything ===
import sys
sys.path.append("./depthanything")
from torchvision.transforms import Compose
from depthanything.fast_import import depth_anything_model 
from depthanything.depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet

# --- 修正后的模型下载逻辑 ---


try:
   
    model_dir = snapshot_download('Saturn666/FDDI')
    print(f"✅ 下载成功，路径为: {model_dir}")
except Exception as e:
    print(f"❌ 下载失败，请检查环境变量或权限: {e}")
    
# 1. 下载 MimicBrush 核心权重仓库


mimic_dir = snapshot_download('xichen/MimicBrush')

# 2. 下载 Stable Diffusion 基础模型仓库 (cleansd)
# 如果魔塔上 xichen 下确实有 cleansd 这个仓库，则单独下载
sd_dir = snapshot_download('xichen/cleansd') 

# --- 重新映射路径 (对应你截图中的实际结构) ---
# 注意：snapshot_download 返回的路径就是仓库根目录
mimicbrush_ckpt = os.path.join(mimic_dir, "mimicbrush/mimicbrush.bin")
vae_model_path = os.path.join(mimic_dir, "sd-vae-ft-mse")
image_encoder_path = os.path.join(mimic_dir, "image_encoder")
depth_model_path = os.path.join(mimic_dir, "depth_model/depth_anything_vitb14.pth")

# 映射基础 SD 模型路径
base_model_path = os.path.join(sd_dir, "stable-diffusion-inpainting")
ref_model_path = os.path.join(sd_dir, "stable-diffusion-v1-5")
if torch.cuda.is_available():
    device = "cuda"
    dtype = torch.float16
    print("检测到 GPU，使用 CUDA 加速模式。")
else:
    device = "cpu"
    dtype = torch.bfloat16 
    print("未检测到 GPU，已降级至 CPU 模式（仅用于页面预览）。")
transform = Compose([
    Resize(
        width=518,
        height=518,
        resize_target=False,
        keep_aspect_ratio=True,
        ensure_multiple_of=14,
        resize_method='lower_bound',
        image_interpolation_method=cv2.INTER_CUBIC,
    ),
    NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    PrepareForNet(),
])
depth_anything_model.load_state_dict(torch.load(depth_model_path))







def pad_img_to_square(original_image, is_mask=False):
    width, height = original_image.size
    
    if height == width:
        return original_image
    
    if height > width:
        padding = (height - width) // 2
        new_size = (height, height)
    else:
        padding = (width - height) // 2
        new_size = (width, width)
    
    if is_mask:
        new_image = Image.new("RGB", new_size, "black")
    else:
        new_image = Image.new("RGB", new_size, "white")
    
    if height > width:
        new_image.paste(original_image, (padding, 0))
    else:
        new_image.paste(original_image, (0, padding))
    return new_image


def collage_region(low, high, mask):
    mask = (np.array(mask) > 128).astype(np.uint8)
    low = np.array(low).astype(np.uint8) 
    low = (low * 0).astype(np.uint8) 
    high = np.array(high).astype(np.uint8)
    mask_3 = mask 
    collage = low * mask_3 + high * (1-mask_3)
    collage = Image.fromarray(collage)
    return collage


def resize_image_keep_aspect_ratio(image, target_size = 512):
    height, width = image.shape[:2]
    if height > width:
        new_height = target_size
        new_width = int(width * (target_size / height))
    else:
        new_width = target_size
        new_height = int(height * (target_size / width))
    resized_image = cv2.resize(image, (new_width, new_height))
    return resized_image


def crop_padding_and_resize(ori_image, square_image):
    ori_height, ori_width, _ = ori_image.shape
    scale = max(ori_height / square_image.shape[0], ori_width / square_image.shape[1])
    resized_square_image = cv2.resize(square_image, (int(square_image.shape[1] * scale), int(square_image.shape[0] * scale)))
    padding_size = max(resized_square_image.shape[0] - ori_height, resized_square_image.shape[1] - ori_width)
    if ori_height < ori_width:
        top = padding_size // 2
        bottom = resized_square_image.shape[0] - (padding_size - top)
        cropped_image = resized_square_image[top:bottom, :,:]
    else:
        left = padding_size // 2
        right = resized_square_image.shape[1] - (padding_size - left)
        cropped_image = resized_square_image[:, left:right,:]
    return cropped_image


def vis_mask(image, mask):
    # mask 3 channle 255
    mask = mask[:,:,0]
    mask_contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw outlines, using random colors
    outline_opacity = 0.5
    outline_thickness = 5
    outline_color = np.concatenate([ [255,255,255], [outline_opacity]  ])

    white_mask = np.ones_like(image) * 255

    mask_bin_3 = np.stack([mask,mask,mask],-1) > 128
    alpha = 0.5 
    image = ( white_mask * alpha + image * (1-alpha) ) * mask_bin_3 + image * (1-mask_bin_3)
    cv2.polylines(image, mask_contours, True, outline_color, outline_thickness, cv2.LINE_AA)
    return image 



noise_scheduler = DDIMScheduler(
    num_train_timesteps=1000,
    beta_start=0.00085,
    beta_end=0.012,
    beta_schedule="scaled_linear",
    clip_sample=False,
    set_alpha_to_one=False,
    steps_offset=1,
)


vae = AutoencoderKL.from_pretrained(vae_model_path).to(dtype=dtype)
unet = UNet2DConditionModel.from_pretrained(base_model_path, subfolder="unet", in_channels=13, low_cpu_mem_usage=False, ignore_mismatched_sizes=True).to(dtype=dtype)
dc_unet = UNet2DConditionModel_DC.from_pretrained(base_model_path, subfolder="unet", in_channels=13, low_cpu_mem_usage=False, ignore_mismatched_sizes=True).to(dtype=dtype)


from DeepCache.DeepCache.sd.apply_dyna import inject_dynamask_processor


inject_dynamask_processor(dc_unet)


pipe = MimicBrushPipeline.from_pretrained(
    base_model_path,
    torch_dtype=dtype,
    scheduler=noise_scheduler,
    vae=vae,
    unet=unet,
    feature_extractor=None,
    safety_checker=None,
    low_cpu_mem_usage=True
)
pipe.enable_attention_slicing()
dc_pipe = MimicBrushPipeline_DC.from_pretrained(
    base_model_path,
    torch_dtype=dtype,
    scheduler=noise_scheduler,
    vae=vae,
    unet=dc_unet,
    feature_extractor=None,
    safety_checker=None,
    low_cpu_mem_usage=True
)
dc_pipe.enable_attention_slicing()

depth_guider = DepthGuider()
referencenet = ReferenceNet.from_pretrained(ref_model_path, subfolder="unet").to(dtype=dtype)
mimicbrush_model = MimicBrush_RefNet(pipe, image_encoder_path, mimicbrush_ckpt,  depth_anything_model, depth_guider, referencenet, device)
dc_mimicbrush_model = MimicBrush_RefNet(dc_pipe, image_encoder_path, mimicbrush_ckpt, depth_anything_model, depth_guider, referencenet, device)
mask_processor = VaeImageProcessor(vae_scale_factor=1, do_normalize=False, do_binarize=True, do_convert_grayscale=True)

using_deepcache = True


def infer_single(ref_image, target_image, target_mask, seed = -1, num_inference_steps=50, guidance_scale = 5, enable_shape_control = False):
    #return ref_image
    """
    mask: 0/1 1-channel  np.array
    image: rgb           np.array
    """

    ref_image = ref_image.astype(np.uint8)
    target_image = target_image.astype(np.uint8)
    target_mask  = target_mask .astype(np.uint8)

    ref_image = Image.fromarray(ref_image.astype(np.uint8)) 
    ref_image = pad_img_to_square(ref_image)

    target_image = pad_img_to_square(Image.fromarray(target_image))
    target_image_low = target_image


    target_mask = np.stack([target_mask,target_mask,target_mask],-1).astype(np.uint8) * 255
    target_mask_np = target_mask.copy()
    target_mask = Image.fromarray(target_mask) 
    target_mask = pad_img_to_square(target_mask, True)

    target_image_ori = target_image.copy()
    target_image = collage_region(target_image_low, target_image, target_mask)
    

    depth_image = target_image_ori.copy()
    depth_image = np.array(depth_image)
    depth_image = transform({'image': depth_image})['image']
    depth_image = torch.from_numpy(depth_image).unsqueeze(0) / 255

    if not enable_shape_control:
        depth_image = depth_image * 0

    mask_pt = mask_processor.preprocess(target_mask, height=512, width=512)
    if using_deepcache:
        pred, depth_pred = dc_mimicbrush_model.generate(pil_image=ref_image, depth_image = depth_image, num_samples=1, num_inference_steps=num_inference_steps,
                            seed=seed, image=target_image, mask_image=mask_pt, strength=1.0, guidance_scale=guidance_scale)
    else:
        pred, depth_pred = mimicbrush_model.generate(pil_image=ref_image, depth_image = depth_image, num_samples=1, num_inference_steps=num_inference_steps,
                                seed=seed, image=target_image, mask_image=mask_pt, strength=1.0, guidance_scale=guidance_scale)


    depth_pred = F.interpolate(depth_pred, size=(512,512), mode = 'bilinear', align_corners=True)[0][0]
    depth_pred = (depth_pred - depth_pred.min()) / (depth_pred.max() - depth_pred.min()) * 255.0
    depth_pred = depth_pred.detach().cpu().numpy().astype(np.uint8)
    depth_pred = cv2.applyColorMap(depth_pred, cv2.COLORMAP_INFERNO)[:,:,::-1]

    pred = pred[0]
    pred = np.array(pred).astype(np.uint8)
    return pred, depth_pred.astype(np.uint8)



def inference_single_image(ref_image, 
                           tar_image, 
                           tar_mask, 
                           ddim_steps, 
                           scale, 
                           seed,
                           enable_shape_control,
                           ):
    if seed == -1:
        seed = np.random.randint(10000)
    pred, depth_pred = infer_single(ref_image, tar_image, tar_mask, seed, num_inference_steps=ddim_steps, guidance_scale = scale, enable_shape_control = enable_shape_control)
    return pred, depth_pred



def run_local(base,
              ref,
              *args):
    image = base["background"].convert("RGB") #base["image"].convert("RGB")
    mask = base["layers"][0]  #base["mask"].convert("L")
    
    image = np.asarray(image)
    mask = np.asarray(mask)[:,:,-1]
    #print(image.shape, mask.shape, mask.max(), mask.min())
    mask = np.where(mask > 128, 1, 0).astype(np.uint8)
    

    ref_image = ref.convert("RGB")
    ref_image = np.asarray(ref_image)

    if mask.sum() == 0:
        raise gr.Error('No mask for the background image.')
    
    mask_3 = np.stack([mask,mask,mask],-1).astype(np.uint8) * 255

    mask_alpha = mask_3.copy()
    for i in range(10):
        mask_alpha = cv2.GaussianBlur(mask_alpha, (3, 3), 0)
    
    synthesis, depth_pred = inference_single_image(ref_image.copy(), image.copy(), mask.copy(), *args)


    synthesis = crop_padding_and_resize(image, synthesis)
    depth_pred = crop_padding_and_resize(image, depth_pred)


    mask_3_bin = mask_alpha / 255
    synthesis = synthesis * mask_3_bin + image * (1-mask_3_bin)

    vis_source = vis_mask(image, mask_3).astype(np.uint8)
    return [synthesis.astype(np.uint8), vis_source, mask_3]
def choose_model():
    global using_deepcache
    using_deepcache = not using_deepcache
    if using_deepcache:
        return "Using DeepCache to accelerate"
    else:
        return "Using original model"

custom_css = """
/* 1. 增加总高度，给下方的工具栏留出呼吸空间 */
#source-editor, #ref-image {
    height: 620px !important; 
    min-height: 620px !important;
    border: 1px solid #ddd !important; /* 保留微弱边框有助于区分界限 */
    border-radius: 8px !important;
    overflow: hidden !important;
}

/* 2. 限制图片容器高度，确保它不把底部的工具栏挤走 */
#source-editor .image-container {
    height: 520px !important; /* 比总高度小，留出空间给底部的工具栏 */
    background: #f0f0f0 !important;
}

/* 3. 确保图片缩放不留黑边，且不会超出容器 */
#source-editor img {
    object-fit: contain !important; 
    width: 100% !important;
    height: 100% !important;
}

/* 4. 优化底部工具栏的显示 */
/* Gradio 4.x 的 ImageEditor 工具栏通常在 .controls 或 .tool-buttons 中 */
#source-editor .controls {
    background-color: white !important;
    padding: 10px !important;
    border-top: 1px solid #eee !important;
}

/* 5. 修复浮动按钮遮挡问题 */
.icon-buttons {
    z-index: 10 !important; /* 确保它在最上层 */
    background: rgba(255, 255, 255, 0.9) !important;
    border: 1px solid #ccc !important;
}
"""

with gr.Blocks(css=custom_css) as demo:
    with gr.Column():
        gr.Markdown("#  MimicBrush: Zero-shot Image Editing with Reference Imitation ")
        with gr.Row():
            baseline_gallery = gr.Gallery(
            label='Output (点击下方小图切换查看)', 
            show_label=True, 
            elem_id="gallery", 
            columns=3,           # 3张图刚好排成一横排缩略图
            rows=1,              # 缩略图只占一行
            height=600,          # 给大图留够高度，防止下方缩略图被遮挡
            preview=True,        # 开启预览模式
            object_fit="contain" # 保证 512x512 的工业图不被裁剪
        )
            with gr.Accordion("Advanced Option", open=True):
                num_samples = 1
                ddim_steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=50, step=1)
                scale = gr.Slider(label="Guidance Scale", minimum=-30.0, maximum=30.0, value=5.0, step=0.1)
                seed = gr.Slider(label="Seed", minimum=-1, maximum=999999999, step=1, value=-1)
                enable_shape_control = gr.Checkbox(label='Keep the original shape', value=False, interactive = True)
                
                gr.Markdown("### Tutorial")
                gr.Markdown("1. Upload the source image and the reference image")
                gr.Markdown("2. Select the \"draw button\" to mask the to-edit region on the source image  ")
                gr.Markdown("3. Click generate ")
                gr.Markdown("#### You shoud click \"keep the original shape\" to conduct texture transfer  ")
    
        gr.Markdown("# Upload the source image and reference image")
        gr.Markdown("### Tips: you could adjust the brush size")
        
        with gr.Row(equal_height=True):
            with gr.Column(scale=1):
                base = gr.ImageEditor(
                    label="Source (512x512)",
                    type="pil",
                    elem_id="source-editor", # 对应 CSS
                    brush=gr.Brush(colors=["#000000"], default_size=20, color_mode="fixed"),
                    # 隐藏不必要的图层控制，减少 UI 占用
                    layers=False,
                    sources=["upload"],
                    interactive=True,
                    canvas_size=(512, 512)
                )
            with gr.Column(scale=1):
                ref = gr.Image(
                    label="Reference", 
                    sources="upload", 
                    type="pil", 
                    elem_id="ref-image" # 对应 CSS
                )
        choose_model_button = gr.Button(value=" DeepCache ")
        run_local_button = gr.Button(value="Run")
        


    with gr.Row():
        gr.Examples(
        examples=[
            ["examples/source/wood.png", "examples/reference/wood_1.png",0],
            ["examples/source/wood.png", "examples/reference/wood_2.png",0]
        ],

        inputs=[
                base,
                ref,
                enable_shape_control
                ],
                cache_examples=False,
                examples_per_page=100)


    run_local_button.click(fn=run_local, 
                           inputs=[base, 
                                   ref, 
                                   ddim_steps, 
                                   scale, 
                                   seed,
                                   enable_shape_control
                                   ], 
                           outputs=[baseline_gallery]
                        )
    choose_model_button.click(fn= choose_model,
                              inputs=[],
                              outputs=[choose_model_button]
                        )

demo.launch(server_name="0.0.0.0")