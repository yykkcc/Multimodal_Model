import torch
from torch import nn
from typing import Optional, Tuple, List
from torch.nn import CrossEntropyLoss
import math
from modeling_siglip import SiglipVisionConfig, SigLipVisionModel

###############################################################################
#                          1. KVCache（缓存机制）                              #
###############################################################################
class KVCache():
    """
    KVCache 用来在解码阶段缓存每一层的 Key/Value，从而在推理（inference）时
    做增量计算（只计算新token对应的注意力，而不重复计算所有历史序列）。
    """

    def __init__(self) -> None:
        # List[torch.Tensor], 每个元素形状: [Batch_Size, Num_Heads_KV, Seq_Len, Head_Dim]
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
    
    def num_items(self) -> int:
        """
        返回缓存中当前累积的 Key/Value 的序列长度。
        如果尚未缓存任何东西，则长度为 0。
        """
        if len(self.key_cache) == 0:
            return 0
        else:
            # The shape of the key_cache is [Batch_Size, Num_Heads_KV, Seq_Len, Head_Dim]
            return self.key_cache[0].shape[-2]

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        在第 layer_idx 层的 Key/Value 中累加新得到的 Key/Value。
        如果该层的缓存尚不存在，就先创建，否则将新值拼到旧值后面。
        """
        if len(self.key_cache) <= layer_idx:
            # 从未在该层添加过缓存，直接append
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        else:
            # 拼接新的 Key/Value 到现有的 Key/Value
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)

        # 返回拼接后的 Key/Value
        return self.key_cache[layer_idx], self.value_cache[layer_idx]

###############################################################################
#                          2. GemmaConfig & PaliGemmaConfig                   #
###############################################################################
class GemmaConfig():
    """
    Gemma（大语言模型部分）的配置：
    - vocab_size, hidden_size, intermediate_size, num_hidden_layers, num_attention_heads
    - num_key_value_heads: 可能为了减少KV的头数，从而减少KV缓存等开销
    - head_dim: 每个头的维度
    - max_position_embeddings, rms_norm_eps, rope_theta, attention_bias, attention_dropout 等
    - pad_token_id: 用来指定 padding 的 token id（有时为 -1 或者 vocab 里某个特殊id）
    """

    def __init__(
        self,
        vocab_size,
        hidden_size,
        intermediate_size,
        num_hidden_layers,
        num_attention_heads,
        num_key_value_heads,
        head_dim=256,
        max_position_embeddings=8192,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        attention_bias=False,
        attention_dropout=0.0,
        pad_token_id=None,
        **kwargs,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.head_dim = head_dim
        self.num_key_value_heads = num_key_value_heads
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        self.pad_token_id = pad_token_id

class PaliGemmaConfig():
    """
    PaliGemma 用到的配置，包含：
    - vision_config：图像编码器 Siglip 的配置
    - text_config：Gemma LLM 的配置
    - ignore_index, image_token_index: 用在文本-图像混合输入时，表示图像 token 的特殊 index
    - projection_dim: 图像编码后再投影到多少维
    - vocab_size: 文本的词表大小
    - hidden_size: LM 的隐藏维度
    """

    def __init__(
        self,
        vision_config=None,
        text_config=None,
        ignore_index=-100,
        image_token_index=256000,
        vocab_size=257152,
        projection_dim=2048,
        hidden_size=2048,
        pad_token_id=None,
        **kwargs,
    ):
        super().__init__()
        self.ignore_index = ignore_index
        self.image_token_index = image_token_index
        self.vocab_size = vocab_size
        self.projection_dim = projection_dim
        self.hidden_size = hidden_size
        self.vision_config = vision_config
        self.is_encoder_decoder = False
        self.pad_token_id = pad_token_id

        # 初始化 vision_config, 例如 SiglipVisionConfig
        self.vision_config = SiglipVisionConfig(**vision_config)
        self.text_config = text_config

        # 初始化 text_config，即 Gemma 的语言模型配置
        self.text_config = GemmaConfig(**text_config, pad_token_id=pad_token_id)
        self.vocab_size = self.text_config.vocab_size

        # 计算图像 token 数量: (image_size // patch_size)**2
        self.text_config.num_image_tokens = (self.vision_config.image_size // self.vision_config.patch_size) ** 2
        # vision_config.projection_dim: 用于后面投影到 LM 的隐藏空间
        self.vision_config.projection_dim = projection_dim

###############################################################################
#                        3. Gemma 模块：RMSNorm, Rotary Embedding等            #
###############################################################################
class GemmaRMSNorm(nn.Module):
    """
    Gemma 中的 RMSNorm：对每个 token 的 hidden_state 做均方根归一化。
    不同于 LayerNorm, RMSNorm不减去均值，只做模长的归一化。
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # 不同于常见 LayerNorm, 这里只学习一个 scale (weight)，初值为0，再加1做缩放
        self.weight = nn.Parameter(torch.zeros(dim))

    def _norm(self, x):
        # 计算 RMS 并做除法
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        # 先在 float32 做运算防止数值误差
        output = self._norm(x.float())
        # 多乘上 (1.0 + self.weight.float())，替代传统 LN 的gamma
        output = output * (1.0 + self.weight.float())
        # 返回到原本的数据类型 (如 float16)
        return output.type_as(x)

class GemmaRotaryEmbedding(nn.Module):
    """
    GemmaRotaryEmbedding: 用于实现 Rotary Position Embedding (RoPE) 的正弦/余弦参数。
    RoPE 依赖公式 theta_i = base^(-2i/dim) 构造可扩展的频率。
    """
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim  # head_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        # inv_freq: [dim//2]，用于生成 sin/cos 的频率
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float() / self.dim))
        # 挂到buffer上，不参与训练
        self.register_buffer("inv_freq", tensor=inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids, seq_len=None):
        """
        x: [batch, num_heads_kv, seq_len, head_dim]
        position_ids: [batch, seq_len] - 每个token的位置
        这里计算 cos,sin 用于和 Q,K 做旋转变换 (apply_rotary_pos_emb)
        """
        self.inv_freq.to(x.device)

        # inv_freq_expanded: [Batch_Size, dim//2, 1]
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        # position_ids_expanded: [Batch_Size, 1, seq_len]
        position_ids_expanded = position_ids[:, None, :].float()
        
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            # freqs: [Batch_Size, seq_len, dim//2]
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            # emb: [Batch_Size, seq_len, dim]
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x):
    """
    辅助函数：对于最后一维的前半部分x1和后半部分x2，组合成 [-x2, x1]
    用于 RoPE 的旋转操作。
    """
    x1 = x[..., : x.shape[-1] // 2] 
    x2 = x[..., x.shape[-1] // 2 :] 
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """
    将 RoPE (cos,sin) 应用于 Q,K
    公式：q_embed = q * cos + rotate_half(q) * sin
         k_embed = k * cos + rotate_half(k) * sin
    """
    # 在指定维度扩展cos, sin，通常是 head 维度
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

###############################################################################
#                             4. Gemma MLP                                    #
###############################################################################
class GemmaMLP(nn.Module):
    """
    Gemma Decoder Layer 中的 MLP部分，包含 gate_proj, up_proj, down_proj。
    这里类似 GPT-NeoX, LLaMA 中的 "SwiGLU" 或类似的激活结构。
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        # gate_proj & up_proj 同时处理 x，然后 element-wise 相乘后再 down_proj
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

    def forward(self, x):
        # GELU approx="tanh" 方式同 LLaMA2
        return self.down_proj(
            nn.functional.gelu(self.gate_proj(x), approximate="tanh") * self.up_proj(x)
        )


###############################################################################
#                          5. Gemma Attention 机制                            #
###############################################################################
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    将 key,value 扩展到 num_heads_q 的数量。
    如果 num_heads_q > num_key_value_heads，需要重复。
    shape: [batch, num_key_value_heads, seq_len, head_dim] -> [batch, num_heads_q, seq_len, head_dim]
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    # 通过 expand + reshape 达到重复 key/value 的效果
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

class GemmaAttention(nn.Module):
    """
    GemmaDecoderLayer 中的自注意力模块。
    包括 Q,K,V投影，RoPE旋转，KVCache增量缓存，以及多头注意力计算。
    """

    def __init__(self, config: GemmaConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True

        # Q,K,V投影
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)

        # 位置编码
        self.rotary_emb = GemmaRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        kv_cache: Optional[KVCache] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """
        做自注意力计算，并返回：
        - attn_output: [Batch_Size, Seq_Len, Hidden_Size]
        - attn_weights (可选)
        - 更新后的 KVCache (可选)
        """
        bsz, q_len, _ = hidden_states.size()

        # 1. 线性投影 Q,K,V
        query_states = self.q_proj(hidden_states)   # [batch, seq_len, num_heads_q * head_dim]
        key_states = self.k_proj(hidden_states)     # [batch, seq_len, num_heads_kv * head_dim]
        value_states = self.v_proj(hidden_states)   # [batch, seq_len, num_heads_kv * head_dim]

        # 2. reshape+transpose 得到 [batch, num_heads, seq_len, head_dim]
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # 3. RoPE 计算 cos, sin
        cos, sin = self.rotary_emb(value_states, position_ids, seq_len=None)
        # 将 RoPE 应用于 Q,K
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # 4. KVCache 处理（增量推理）
        if kv_cache is not None:
            key_states, value_states = kv_cache.update(key_states, value_states, self.layer_idx)

        # 5. 如果 num_heads_q > num_heads_kv，需要把key,value重复到对应的头数
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # 6. 计算注意力得分 Attn = Q*K^T / sqrt(head_dim)
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        # 加上 attention_mask（通常是 causal mask 或 padding mask）
        assert attention_mask is not None
        attn_weights = attn_weights + attention_mask

        # 7. softmax + dropout
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)

        # 8. 与 V 相乘
        attn_output = torch.matmul(attn_weights, value_states)
        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(f"`attn_output` shape mismatch: {attn_output.size()}")

        # 9. 把维度换回来 -> [batch, seq_len, num_heads, head_dim]
        attn_output = attn_output.transpose(1, 2).contiguous()
        # 然后合并 num_heads 和 head_dim -> [batch, seq_len, num_heads*head_dim]
        attn_output = attn_output.view(bsz, q_len, -1)
        # 最后 o_proj 恢复到 [batch, seq_len, hidden_size]
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights

###############################################################################
#                     6. GemmaDecoderLayer (1层Block)                         #
###############################################################################
class GemmaDecoderLayer(nn.Module):
    """
    一个 DecoderLayer 包含：
    - Self_Attention
    - RMSNorm + 残差
    - MLP + RMSNorm + 残差
    """

    def __init__(self, config: GemmaConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = GemmaAttention(config=config, layer_idx=layer_idx)
        self.mlp = GemmaMLP(config)
        # 前后各一个 RMSNorm
        self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        计算一个DecoderLayer的输出
        """
        # 1. RMSNorm + self-attn
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _, = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            kv_cache=kv_cache,
        )
        hidden_states = residual + hidden_states

        # 2. RMSNorm + MLP
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


###############################################################################
#                         7. GemmaModel (Decoder Stack)                       #
###############################################################################
class GemmaModel(nn.Module):
    """
    GemmaModel: 多层 DecoderLayer 堆叠形成的 Causal LM backbone。
    """

    def __init__(self, config: GemmaConfig):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        # 词向量嵌入表
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        # N层 decoder layer
        self.layers = nn.ModuleList(
            [GemmaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        # 最后的归一化
        self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def get_input_embeddings(self):
        return self.embed_tokens

    def forward(
        self,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> torch.FloatTensor:
        """
        计算整堆 DecoderLayers 的前向输出。
        inputs_embeds: [batch, seq_len, hidden_size]
        """
        hidden_states = inputs_embeds
        # 简单做个 scale
        normalizer = torch.tensor(self.config.hidden_size**0.5, dtype=hidden_states.dtype)
        hidden_states = hidden_states * normalizer

        # 依次通过每一层 decoder
        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                kv_cache=kv_cache,
            )

        # 最后再 RMSNorm
        hidden_states = self.norm(hidden_states)
        return hidden_states


###############################################################################
#                    8. GemmaForCausalLM (含语言模型头)                        #
###############################################################################
class GemmaForCausalLM(nn.Module):
    """
    在 GemmaModel 基础上，加一个 LM head (线性层),
    用于把 hidden_states 投影到 vocab_size 大小，以便算logits。
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = GemmaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def get_input_embeddings(self):
        return self.model.embed_tokens
    
    def tie_weights(self):
        # LM head 和 embedding 权重共享
        self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> Tuple:
        """
        先过 GemmaModel，再线性到 vocab logits。
        """
        # hidden_states: [Batch_Size, Seq_Len, Hidden_Size]
        outputs = self.model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            kv_cache=kv_cache,
        )

        # [Batch_Size, Seq_Len, Vocab_Size]
        hidden_states = outputs
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        return_data = {
            "logits": logits,
        }

        if kv_cache is not None:
            # 返回更新后的 kv_cache 以便后续增量推理
            return_data["kv_cache"] = kv_cache
        
        return return_data


###############################################################################
#                     9. PaliGemmaMultiModalProjector                         #
###############################################################################
class PaliGemmaMultiModalProjector(nn.Module):
    """
    将 Vision encoder(主类vision_tower) 的输出投影到与 Gemma LLM 对齐的维度。
    """
    def __init__(self, config: PaliGemmaConfig):
        super().__init__()
        self.linear = nn.Linear(config.vision_config.hidden_size, config.vision_config.projection_dim, bias=True)

    def forward(self, image_features):
        # 输入: [Batch_Size, Num_Patches, Embed_Dim]
        # 输出: [Batch_Size, Num_Patches, Projection_Dim]
        hidden_states = self.linear(image_features)
        return hidden_states


###############################################################################
#                 10. PaliGemmaForConditionalGeneration (主类)                 #
###############################################################################
class PaliGemmaForConditionalGeneration(nn.Module):
    """
    组合:
    - vision_tower: SiglipVisionModel (图像编码器)
    - multi_modal_projector: 将图像特征映射到 language model 的投影维度
    - language_model: GemmaForCausalLM (语言模型，可处理文本输出)
    """
    def __init__(self, config: PaliGemmaConfig):
        super().__init__()
        self.config = config
        # 图像编码器: SiglipVisionModel
        self.vision_tower = SigLipVisionModel(config.vision_config)
        self.multi_modal_projector = PaliGemmaMultiModalProjector(config)
        self.vocab_size = config.vocab_size

        # 文本解码器 (GemmaForCausalLM), 里面包含 GemmaDecoder
        language_model = GemmaForCausalLM(config.text_config)
        self.language_model = language_model
        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1

    def tie_weights(self):
        return self.language_model.tie_weights()
    
    def _merge_input_ids_with_image_features(
        self, image_features: torch.Tensor, inputs_embeds: torch.Tensor, 
        input_ids: torch.Tensor, attention_mask: torch.Tensor,
        kv_cache: Optional[KVCache] = None
    ):
        # image_features: 来自视觉编码器 (vision_tower) + 投影层 (multi_modal_projector) 的输出。[batch_size, num_patches, embed_dim]

        # inputs_embeds: 由 input_ids 通过语言模型的嵌入表 (Embedding) 得到的文本嵌入向量。[batch_size, seq_len, hidden_size], 此时还没和图像特征合并

        # input_ids: [batch_size, seq_len], 这里面可能包含特殊 token，例如 <image> token, 会利用 input_ids 来判断序列的哪些位置是文本、哪些位置是图像、哪些位置是 padding。

        # attention_mask: 一般是 [batch_size, seq_len] 的张量，值为 1 表示这个 token 是有效的（要参与注意力计算），值为 0 表示padding或mask掉。函数的后半段会根据场景（是否使用 kv_cache）去构造一个自回归掩码 (causal_mask)。
        """
        将图像特征嵌入 image_tokens 对应的位置，
        将文本嵌入 text_tokens 对应的位置，
        最后合并成一段可被 language_model 接受的 embeddings。
        同时生成相应的 causal_mask 和 position_ids。
        """
        # 1. 获取基本信息 & 缩放图像特征
        _, _, embed_dim = image_features.shape
        batch_size, sequence_length = input_ids.shape
        dtype, device = inputs_embeds.dtype, inputs_embeds.device

        scaled_image_features = image_features / (self.config.hidden_size**0.5)
        
        # 2. 创建一个最终的 final_embedding 张量
        final_embedding = torch.zeros(
            batch_size, sequence_length, embed_dim, 
            dtype=inputs_embeds.dtype, 
            device=inputs_embeds.device
        )

        # 3. 计算 text_mask, image_mask, pad_mask
        text_mask = (input_ids != self.config.image_token_index) & (input_ids != self.pad_token_id)
        image_mask = (input_ids == self.config.image_token_index)
        pad_mask = (input_ids == self.pad_token_id)
        # text_mask：某个位置既不是 <image> 也不是 <pad> → 这里就是普通文本 token
        # image_mask：某个位置就是 <image> → 这里要放图像特征
        # pad_mask：某个位置是 <pad> → 这里要放0，或者说在注意力中要被屏蔽

        # 4. 原先的 mask 形状都是 [batch_size, seq_len]，现在要扩展最后一个维度到 embedding 的大小 [batch_size, seq_len, embed_dim], 以便用 torch.where、masked_scatter 等操作进行元素级替换。
        text_mask_expanded = text_mask.unsqueeze(-1).expand(-1, -1, embed_dim)
        image_mask_expanded = image_mask.unsqueeze(-1).expand(-1, -1, embed_dim)
        pad_mask_expanded = pad_mask.unsqueeze(-1).expand(-1, -1, embed_dim)

        # 5. 将文本和图像特征放入 final_embedding
        # 如果是 text_mask 的位置，用 inputs_embeds (文本向量) 替换
        final_embedding = torch.where(text_mask_expanded, inputs_embeds, final_embedding)

        # 如果是 image_mask 的位置，用 scaled_image_features 替换
        final_embedding = final_embedding.masked_scatter(image_mask_expanded, scaled_image_features)

        # 如果是 pad_mask 的位置，则置为0 (其实已经是0，但这里再次确保)
        final_embedding = torch.where(pad_mask_expanded, torch.zeros_like(final_embedding), final_embedding)

        # 6. 创建或更新 causal_mask（自回归掩码）
        dtype, device = inputs_embeds.dtype, inputs_embeds.device
        min_dtype = torch.finfo(dtype).min
        q_len = inputs_embeds.shape[1]
    
        if kv_cache is None or kv_cache.num_items() == 0:
            # Prefill阶段：还没有任何缓存，也没有历史tokens
            causal_mask = torch.full((batch_size, q_len, q_len), fill_value=0, dtype=dtype, device=device)
        else:
            # 增量推理阶段: q_len 应该等于 1 (只在一个新token上做前向)
            assert q_len == 1
            kv_len = kv_cache.num_items() + q_len
            # 这里也依旧填充成全0，表示不遮盖任何位置（因为 causal 性能自动在模型内部mask过去的tokens?）
            causal_mask = torch.full((batch_size, q_len, kv_len), fill_value=0, dtype=dtype, device=device)

        # Add the head dimension
        causal_mask = causal_mask.unsqueeze(1)

        # 7. 生成 position_ids，这些 position_ids 会传给语言模型的自注意力层或 RoPE 算法，用于位置旋转或其他位置编码的计算。
        if kv_cache is not None and kv_cache.num_items() > 0:
            # 增量推理：只取最后一个位置
            position_ids = attention_mask.cumsum(-1)[:, -1]
            if position_ids.dim() == 1:
                position_ids = position_ids.unsqueeze(0)
        else:
            # 普通情况：根据 attention_mask 的数量来生成 1,2,3..., 
            # 遇到 mask=0 (padding) 就保持成1
            position_ids = (attention_mask.cumsum(-1)).masked_fill_((attention_mask == 0), 1).to(device)

        return final_embedding, causal_mask, position_ids

    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: torch.FloatTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
    ) -> Tuple:
        # input_ids: 是 processing_paligemma.py 中组建的 image_tokens + bos_token + user prompt tokens + /n token
        # pixel_values: 预处理过的图像 [Batch_size, channels, height, width]
        """
        main forward:
        1. 先获取文本 embeddings
        2. 过 vision_tower 得到图像特征
        3. 投影图像特征
        4. 合并 image_embeddings + text_embeddings
        5. 送入 language_model 做 Causal LM
        """
        # 注意，这里假设 attention_mask == 1，表示无padding（右侧无pad）
        assert torch.all(attention_mask == 1), "The input cannot be padded"

        # 1. 获取文本向量: [Batch_Size, Seq_Len, Hidden_Size]
        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)

        # 2. 获取图像向量: 先送进 vision_tower (SigLIP) -> [Batch_Size, Num_Patches, vision_hidden_size]
        selected_image_feature = self.vision_tower(pixel_values.to(inputs_embeds.dtype))

        # 3. 再投影到 projection_dim (或与 language hidden_size 相对应)
        #    [Batch_Size, Num_Patches, projection_dim]
        image_features = self.multi_modal_projector(selected_image_feature)

        # 4. 将图像位置替换成相应的 image_features，文本位置保持原文，合并成 final embeddings
        inputs_embeds, attention_mask, position_ids = self._merge_input_ids_with_image_features(image_features, inputs_embeds, input_ids, attention_mask, kv_cache)

        # 5. 送入语言模型解码
        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            kv_cache=kv_cache,
        )

        return outputs