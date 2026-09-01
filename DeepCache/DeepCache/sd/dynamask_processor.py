import torch
import torch.nn.functional as F

class DynaMaskAttnProcessor:
    def __init__(self):
       
        self.bg_cache = None
        # 用于存储当前状态的变量
        self.current_mask = None
        self.current_timestep = None
        self.lpf_threshold = 0
        self.mask_index_cache = {}
        self._mask_signature = None

    # 一个专门的方法，供 U-Net 调用来设置状态
    def set_state(self, mask, timestep, threshold):
        # Processors live on the UNet across requests. A cache keyed only by
        # sequence length would otherwise reuse the previous request's mask.
        if torch.is_tensor(mask):
            mask_signature = (
                mask.data_ptr(),
                tuple(mask.shape),
                str(mask.device),
                str(mask.dtype),
            )
        else:
            mask_signature = None
        if mask_signature != self._mask_signature:
            self.bg_cache = None
            self.mask_index_cache.clear()
            self._mask_signature = mask_signature

        self.current_mask = mask
        self.current_timestep = timestep
        self.lpf_threshold = threshold

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        scale=1.0,
        **kwargs  # 接收所有 cross_attention_kwargs 里的参数
    ):
        
        
        # 1. 优先使用 set_state 设置的变量，如果没有则尝试从 kwargs 拿 (兼容性)
        dyna_mask = self.current_mask if self.current_mask is not None else kwargs.get("dyna_mask", None)
        lpf_threshold = self.lpf_threshold
        
        # 获取 timestep (优先用状态里的，因为它是 U-Net 传的最准的)
        timestep = self.current_timestep if self.current_timestep is not None else kwargs.get("timestep", None)
        
        # 处理 timestep 格式
        if timestep is not None and torch.is_tensor(timestep):
            if timestep.numel() > 1: timestep = timestep[0]
            timestep = timestep.item()
        
        # 2. 判断是否为 Cross-Attention
        # 如果有 encoder_hidden_states，说明是 Cross-Attn，必须全量计算以获取 Reference/Prompt 信息
        
        if encoder_hidden_states is not None:
            return self._standard_attention(attn, hidden_states, encoder_hidden_states, attention_mask, scale)

        # 3. 判断是否进入 Refinement Phase (且 Mask 存在)
        # 逻辑：只有在 timestep < threshold (后期) 且提供了 Mask 时，才开启加速
        is_refinement = (timestep is not None) and (timestep < lpf_threshold)
        
        if not is_refinement or dyna_mask is None:
            # Phase 1 (结构期) 或 无 Mask -> 清空缓存，跑标准全量 Self-Attention
            self.bg_cache = None 
            return self._standard_attention(attn, hidden_states, None, attention_mask, scale)

        # 4. Phase 2: Mask-Sparse Self-Attention (核心加速逻辑)
        # 【修改点】：这里把 timestep 传进去，用于判断间隔缓存
        return self._sparse_self_attention(attn, hidden_states, dyna_mask, scale, timestep)

    def _standard_attention(self, attn, hidden_states, encoder_hidden_states, attention_mask, scale):
        """标准的全量 Attention 计算 (基于 SDPA 加速)"""
        # 检查 input dtype 和 weight dtype 是否一致
        # 如果 hidden_states 是 float32 但权重是 float16，强转为 float16
        target_dtype = attn.to_q.weight.dtype
        if hidden_states.dtype != target_dtype:
            hidden_states = hidden_states.to(target_dtype)
            
        # 如果有 encoder_hidden_states (Cross-Attn的情况)，也要对齐
        if encoder_hidden_states is not None and encoder_hidden_states.dtype != target_dtype:
            encoder_hidden_states = encoder_hidden_states.to(target_dtype)
        batch_size, sequence_length, _ = hidden_states.shape
        
        # Q
        query = attn.to_q(hidden_states)
        
        # K, V
        # 处理 Cross-Attention 的情况
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_cross(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        # Multi-head reshape
        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # Scaled Dot-Product Attention
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        
        # Output projection
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states

    def _sparse_self_attention(self, attn, hidden_states, dyna_mask, scale, timestep):
        """只计算 Mask 区域的稀疏 Attention"""
        
        target_dtype = attn.to_q.weight.dtype
        if hidden_states.dtype != target_dtype:
            hidden_states = hidden_states.to(target_dtype)
        batch_size, seq_len, dim = hidden_states.shape
        # --- 0. 基础准备 ---
        # 1000 = One-Shot (Phase 2 只更新一次背景，后面全复用，速度最快)
        refine_interval = 1000 
        t_val = timestep if timestep is not None else 0
        should_update = (self.bg_cache is None) or (t_val % refine_interval == 0)

        batch_size, seq_len, dim = hidden_states.shape
        
        # --- A. 索引缓存 (Index Caching) ---
        # 避免每一步都做 interpolate 和 nonzero 的 CPU-GPU 同步开销
        if seq_len in self.mask_index_cache:
            fg_indices = self.mask_index_cache[seq_len]
            # 如果是 update 步，我们不需要 bg_indices，因为直接存全量
        else:
            side_len = int(seq_len**0.5)
            # 下采样 Mask
            current_mask = F.interpolate(dyna_mask.float(), size=(side_len, side_len), mode="nearest")
            mask_flat = current_mask.view(batch_size, -1)
            
            # 计算索引
            fg_indices = (mask_flat[0] > 0.5).nonzero(as_tuple=True)[0]
            # 存入缓存
            self.mask_index_cache[seq_len] = fg_indices

        # --- B. 缓存未命中 / 需要更新 (Cache Miss) ---
        if should_update:
            # 1. 跑全量标准计算 (利用 FlashAttention)
            full_out = self._standard_attention(attn, hidden_states, None, None, scale)
            
            # 2. 计算全量 K, V
            # 【优化】我们不再拆分 k_bg, v_bg，而是直接存全量
            key_full = attn.to_k(hidden_states)
            value_full = attn.to_v(hidden_states)
            
            # 3. 预先处理好 Head 维度 (避免在稀疏步重复 reshape)
            # Shape: (B, Heads, SeqLen, HeadDim)
            inner_dim = key_full.shape[-1]
            head_dim = inner_dim // attn.heads
            key_full = key_full.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            value_full = value_full.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            
            # 4. 存入缓存
            self.bg_cache = {
                'k_all': key_full,    # 全量 K
                'v_all': value_full,  # 全量 V
                'out_full': full_out  # 全量 Output
            }
            # 更新步直接返回，不折腾
            return full_out
            
        else:
            # --- C. 缓存命中 - 极速稀疏计算 (Cache Hit) ---
            
            # 1. 准备底板 (In-place copy)
            # 直接复制全量 Output，背景天然就在里面了，不需要 scatter 背景
            # return self.bg_cache['out_full']
            out_final = self.bg_cache['out_full'].clone()
            
            # 2. Gather 前景 Query (必须算新的)
            hidden_states_fg = hidden_states[:, fg_indices, :] 
            query_fg = attn.to_q(hidden_states_fg)
            
            # Reshape Query
            inner_dim = query_fg.shape[-1]
            head_dim = inner_dim // attn.heads
            query_fg = query_fg.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            
            # 3. 【核心加速】Frozen Context
            # 不计算 key_fg，也不做 torch.cat
            # 直接拿缓存里的 'k_all' 和 'v_all' 作为 Context
            # 理论：Phase 2 背景不变，前景 K/V 变化微小，用旧的全量 K/V 做 Context 误差极小
            key_context = self.bg_cache['k_all']
            value_context = self.bg_cache['v_all']
            
            # 4. FlashAttention
            # Q 是稀疏的 (M个)，K/V 是全量的 (N个) -> 计算量 O(M * N)
            out_fg = F.scaled_dot_product_attention(
                query_fg, key_context, value_context, dropout_p=0.0, is_causal=False
            )
            
            # 5. Projection
            out_fg = out_fg.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
            out_fg = attn.to_out[0](out_fg)
            out_fg = attn.to_out[1](out_fg)
            
            # 6. 只填回前景 (Scatter Foreground)
            out_final[:, fg_indices, :] = out_fg
            
            return out_final
