import torch
from .dynamask_processor import DynaMaskAttnProcessor

def inject_dynamask_processor(unet):
    """
    遍历 UNet，将所有的 Self-Attention 层的 Processor 替换为 DynaMaskAttnProcessor
    """
    print("🚀 Injecting DynaMask Processors into U-Net...")
    
    count = 0
    # 遍历 U-Net 的所有子模块
    for name, module in unet.named_modules():
        
        # 检查是否是 Attention 层
        # diffusers 的 attention 类名通常包含 'Attention'
        if module.__class__.__name__.endswith("Attention"):
            
            # 区分 Self-Attn 和 Cross-Attn
            # 在 SD 1.5/SDXL 结构中:
            # attn1 是 Self-Attention (我们要改的)
            # attn2 是 Cross-Attention (不要改，负责 Reference/Prompt)
            if "attn1" in name:
                # 必须为每一层实例化一个新的 Processor (因为每一层的缓存不同)
                processor = DynaMaskAttnProcessor()
                module.set_processor(processor)
                count += 1
                
    print(f"✅ Injection Complete. Replaced {count} Self-Attention processors.")