from typing import Optional, Tuple

import torch
import torch.sagemaker as tsm
import torch.utils.checkpoint
import transformer_engine.pytorch as te
from transformers import Cache
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
)
from transformers.models.llama.modeling_llama import (
    LlamaFlashAttention2,
    apply_rotary_pos_emb,
)
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
    replace_return_docstrings,
)

is_patched = False

original_get_extra_state = te.attention.DotProductAttention.get_extra_state
original_LFA2__init__ = LlamaFlashAttention2.__init__
original_LFA2_forward = LlamaFlashAttention2.forward


def unapply_patch():
    global is_patched
    te.attention.DotProductAttention.get_extra_state = original_get_extra_state
    LlamaFlashAttention2.__init__ = original_LFA2__init__
    LlamaFlashAttention2.forward = original_LFA2_forward
    is_patched = False


def apply_patch():
    global is_patched
    # patch https://tiny.amazon.com/1dh46qr58/githNVIDTranblob8416tranpyto
    te.attention.DotProductAttention.get_extra_state = patched_get_extra_state
    # patch https://tiny.amazon.com/c5tg8rbf/githhuggtranblobmainsrctran
    LlamaFlashAttention2.__init__ = patched_LFA2__init__
    # patch https://tiny.amazon.com/lv60zw8r/githhuggtranblobmainsrctran
    LlamaFlashAttention2.forward = patched_LFA2_forward
    is_patched = True


def patched_get_extra_state(self, *args, **kwargs):
    ret = super(self.__class__, self).get_extra_state(*args, **kwargs)
    ret.device = None
    return ret


def patched_LFA2__init__(self, *args, **kwargs):
    super(self.__class__, self).__init__(*args, **kwargs)

    # TODO: Should be removed once Flash Attention for RoCm is bumped to 2.1.
    # flash_attn<2.1 generates top-left aligned causal mask, while what is needed here is bottom-right alignement, that was made default for flash_attn>=2.1. This attribute is used to handle this difference. Reference: https://github.com/Dao-AILab/flash-attention/releases/tag/v2.1.0.
    # Beware that with flash_attn<2.1, using q_seqlen != k_seqlen (except for the case q_seqlen == 1) produces a wrong mask (top-left).
    self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()
    ##### SAGEMAKER Add core attention OF TRANSFORMER ENGINE!
    llama_config = kwargs["config"]
    num_gqa_groups = llama_config.num_key_value_heads
    num_attention_heads = llama_config.num_attention_heads
    kv_channels = llama_config.hidden_size // num_attention_heads
    attention_dropout = 0.0
    tp_size = 1
    get_rng_state_tracker = tsm.state.get_rng_state_tracker
    sequence_parallel = False
    tp_group = None
    layer_number = 0

    self.core_attention = te.attention.DotProductAttention(
        num_attention_heads,
        kv_channels,
        num_gqa_groups=num_gqa_groups,
        attention_dropout=attention_dropout,
        tp_size=tp_size,
        get_rng_state_tracker=get_rng_state_tracker,
        sequence_parallel=sequence_parallel,
        tp_group=tp_group,
        layer_number=layer_number,
    )


def patched_LFA2_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.LongTensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    output_attentions = False

    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    # Flash attention requires the input to have the shape
    # batch_size x seq_length x head_dim x hidden_dim
    # therefore we just need to keep the original shape
    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    cos, sin = self.rotary_emb(value_states, position_ids)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    past_key_value = getattr(self, "past_key_value", past_key_value)

    if past_key_value is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    # TODO: These transpose are quite inefficient but Flash Attention requires the layout [batch_size, sequence_length, num_heads, head_dim]. We would need to refactor the KV cache
    # to be able to avoid many of these transpose/reshape/view.
    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)

    dropout_rate = self.attention_dropout if self.training else 0.0

    # In PEFT, usually we cast the layer norms in float32 for training stability reasons
    # therefore the input hidden states gets silently casted in float32. Hence, we need
    # cast them back in the correct dtype just to be sure everything works as expected.
    # This might slowdown training & inference so it is recommended to not cast the LayerNorms
    # in fp32. (LlamaRMSNorm handles it correctly)

    input_dtype = query_states.dtype
    if input_dtype == torch.float32:
        if torch.is_autocast_enabled():
            target_dtype = torch.get_autocast_gpu_dtype()
        # Handle the case where the model is quantized
        elif hasattr(self.config, "_pre_quantization_dtype"):
            target_dtype = self.config._pre_quantization_dtype
        else:
            target_dtype = self.q_proj.weight.dtype

        logger.warning_once(
            f"The input hidden states seems to be silently casted in float32, this might be related to"
            f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
            f" {target_dtype}."
        )

        query_states = query_states.to(target_dtype)
        key_states = key_states.to(target_dtype)
        value_states = value_states.to(target_dtype)

    ##### SAGEMAKER REPLACE WITH CORE ATTENTION OF TRANSFORMER ENGINE!!!
    # attn_output = self._flash_attention_forward(
    #     query_states, key_states, value_states, attention_mask, q_len, dropout=dropout_rate
    # )
    # Attention.
    attn_mask_type = "causal"
    window_size = None
    checkpoint_core_attention = False
    core_attention_bias_type = "no_bias"
    core_attention_bias = None
    fast_zero_fill = True
    # Attention.
    context_layer = self.core_attention(
        query_states.transpose(0, 1),  # seq, bs, num_attn_heads, kv_channels - required shape
        key_states.transpose(0, 1),  # seq, bs, num_attn_heads, kv_channels
        value_states.transpose(0, 1),  # seq, bs, num_attn_heads, kv_channels
        qkv_format="sbhd",  # can use bshd, but internally just transposese key,value states
        cu_seqlens_q=None,
        cu_seqlens_kv=None,
        attention_mask=attention_mask,
        attn_mask_type=attn_mask_type,  # ?
        window_size=window_size,  # ?
        checkpoint_core_attention=checkpoint_core_attention,  # ?
        core_attention_bias_type=core_attention_bias_type,  # ?
        core_attention_bias=core_attention_bias,  # ?
        fast_zero_fill=fast_zero_fill,  # ?
    )  # seq, bs, num_attention_heads, kv_channels
    attn_output = context_layer.reshape(bsz, q_len, self.hidden_size).contiguous()
    ##### FINISH SAGEMAKER REPLACEMENT
    # attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value
