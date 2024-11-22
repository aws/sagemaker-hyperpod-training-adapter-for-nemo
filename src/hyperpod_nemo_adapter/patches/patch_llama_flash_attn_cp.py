# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
# Portions taken from https://github.com/NVIDIA/NeMo, Copyright Nvidia Corporation

from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
import transformer_engine.pytorch as te
from transformers import Cache, DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.llama.modeling_llama import (
    LLAMA_INPUTS_DOCSTRING,
    LlamaFlashAttention2,
    LlamaModel,
    LlamaRotaryEmbedding,
    apply_rotary_pos_emb,
    logger,
)
from transformers.utils import (
    add_start_docstrings_to_model_forward,
    is_flash_attn_greater_or_equal_2_10,
)

is_patched = False

original_get_extra_state = te.attention.DotProductAttention.get_extra_state
original_LFA2__init__ = LlamaFlashAttention2.__init__
original_LFA2_forward = LlamaFlashAttention2.forward
original_LM_forward = LlamaModel.forward
original_LRE_forward = LlamaRotaryEmbedding.forward


def unapply_patch():
    global is_patched
    te.attention.DotProductAttention.get_extra_state = original_get_extra_state
    LlamaFlashAttention2.__init__ = original_LFA2__init__
    LlamaFlashAttention2.forward = original_LFA2_forward
    LlamaModel.forward = original_LM_forward
    LlamaRotaryEmbedding.forward = original_LRE_forward
    is_patched = False


def apply_patch():
    global is_patched
    # patch https://github.com/NVIDIA/TransformerEngine/blob/841634cab9662581ed0decaa2d3e6dac2b8b544b/transformer_engine/pytorch/module/base.py#L525
    te.attention.DotProductAttention.get_extra_state = patched_get_extra_state
    # patch https://github.com/huggingface/transformers/blob/v4.44.2/src/transformers/models/llama/modeling_llama.py#L465
    LlamaFlashAttention2.__init__ = patched_LFA2__init__
    # patch https://github.com/huggingface/transformers/blob/v4.44.2/src/transformers/models/llama/modeling_llama.py#L473
    LlamaFlashAttention2.forward = patched_LFA2_forward
    # patch https://github.com/huggingface/transformers/blob/v4.44.2/src/transformers/models/llama/modeling_llama.py#L918
    LlamaModel.forward = patched_LM_forward
    # patch https://github.com/huggingface/transformers/blob/v4.44.2/src/transformers/models/llama/modeling_llama.py#L198
    LlamaRotaryEmbedding.forward = patched_LRE_forward
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

    # Attention.
    self.core_attention = te.attention.DotProductAttention(
        num_attention_heads,
        kv_channels,
        num_gqa_groups=num_gqa_groups,
        attention_dropout=self.attention_dropout if self.training else 0.0,
        qkv_format="sbhd",
        tp_size=1,
        get_rng_state_tracker=None,
        sequence_parallel=False,
        tp_group=None,
        layer_number=self.layer_idx + 1,
        attention_type="self",
    )


def get_position_embedding_on_this_context_parallel_rank(
    position_embedding: torch.Tensor, seq_dim: int
) -> torch.Tensor:
    """
    Position embeddings are created for full context length
    This function truncates them based on sequence present on this rank
    From:
    https://github.com/NVIDIA/NeMo/blob/501f0dfc76886fda7f95e934de39fd8275628e2a/nemo/collections/nlp/modules/common/megatron/language_model.py#L726
    """
    from torch.sagemaker.state_handler import state

    cp_size = state.cp_size
    cp_rank = state.cp_rank
    cp_idx = torch.tensor([cp_rank, (2 * cp_size - cp_rank - 1)], device="cpu", pin_memory=True).cuda(
        non_blocking=False
    )
    position_embedding = position_embedding.view(
        *position_embedding.shape[:seq_dim], 2 * cp_size, -1, *position_embedding.shape[(seq_dim + 1) :]
    )
    position_embedding = position_embedding.index_select(seq_dim, cp_idx)
    position_embedding = position_embedding.view(
        *position_embedding.shape[:seq_dim], -1, *position_embedding.shape[(seq_dim + 2) :]
    )
    return position_embedding


def patched_LFA2_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.LongTensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.45
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    output_attentions = False

    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    # Flash attention requires the input to have the shape
    # batch_size x seq_length x head_dim x hidden_dim
    # therefore we just need to keep the original shape

    # [bs, num_attn_heads, sq, kv_channels]
    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    # [bs, num_kv_heads, sq, kv_channels]
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    # [bs, num_kv_heads, sq, kv_channels]
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    if position_embeddings is None:
        logger.warning_once(
            "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
            "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
            "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.45 `position_ids` will be "
            "removed and `position_embeddings` will be mandatory."
        )
        # this is already forced to FP32 internally
        cos, sin = self.rotary_emb(value_states, position_ids)
    else:
        cos, sin = position_embeddings

    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    past_key_value = getattr(self, "past_key_value", past_key_value)

    if past_key_value is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

    # TODO: These transpose are quite inefficient but Flash Attention requires the layout [batch_size, sequence_length, num_heads, head_dim]. We would need to refactor the KV cache
    # to be able to avoid many of these transpose/reshape/view.
    #### SAGEMAKER disable transpose here
    # query_states = query_states.transpose(1, 2)
    # key_states = key_states.transpose(1, 2)
    # value_states = value_states.transpose(1, 2)
    #### FINISH SAGEMAKER change

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
    context_layer = self.core_attention(
        query_states.transpose(1, 2).transpose(0, 1),  # seq, bs, num_attn_heads, kv_channels
        key_states.transpose(1, 2).transpose(0, 1),  # seq, bs, num_gqa_groups, kv_channels
        value_states.transpose(1, 2).transpose(0, 1),  # seq, bs, num_gqa_groups, kv_channels
        qkv_format="sbhd",  # can use bshd, but internally just transposes key,value states
        cu_seqlens_q=None,
        cu_seqlens_kv=None,
        attention_mask=None,
        attn_mask_type="causal",
        window_size=(-1, 0),  # originally None, but is set to (-1,0) when causal attn mask type is used
        checkpoint_core_attention=False,
        core_attention_bias_type="no_bias",
        core_attention_bias=None,
        fast_zero_fill=True,
    )  # [sq, bs, hs] - output shape depends on qkv_format

    # [sq, bs, hs] -> [bs, sq, hs]
    attn_output = context_layer.transpose(0, 1).reshape(bsz, q_len, -1).contiguous()
    ##### FINISH SAGEMAKER REPLACEMENT

    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value


@torch.no_grad()
def patched_LRE_forward(self, x, position_ids):
    if "dynamic" in self.rope_type:
        self._dynamic_frequency_update(position_ids, device=x.device)

    # Core RoPE block
    inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
    position_ids_expanded = position_ids[:, None, :].float()
    # Force float32 (see https://github.com/huggingface/transformers/pull/29285)
    device_type = x.device.type
    device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
    with torch.autocast(device_type=device_type, enabled=False):
        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
        # [1, full_sq, head_dim]
        emb = torch.cat((freqs, freqs), dim=-1)

        #### SAGEMAKER CHANGE slice position embedding for local CP rank
        # [1, full_sq, head_dim] -> [full_sq, 1, 1, head_dim]
        emb = emb.transpose(0, 1)[:, None, :, :]
        # [local_sq, 1, 1, head_dim]
        emb = get_position_embedding_on_this_context_parallel_rank(emb, 0)
        # [local_sq, 1, 1, head_dim] -> [1, local_sq, head_dim]
        emb = emb.squeeze(1).transpose(0, 1)
        #### END SAGEMAKER CHANGE

        cos = emb.cos()
        sin = emb.sin()

    # Advanced RoPE types (e.g. yarn) apply a post-processing scaling factor, equivalent to scaling attention
    cos = cos * self.attention_scaling
    sin = sin * self.attention_scaling

    return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


@add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
def patched_LM_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
) -> Union[Tuple, BaseModelOutputWithPast]:
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError(
            "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
        )

    if self.gradient_checkpointing and self.training and use_cache:
        logger.warning_once("`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`.")
        use_cache = False

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    return_legacy_cache = False
    if (
        use_cache and not isinstance(past_key_values, Cache) and not self.training
    ):  # kept for BC (non `Cache` `past_key_values` inputs)
        return_legacy_cache = True
        past_key_values = DynamicCache.from_legacy_cache(past_key_values)
        logger.warning_once(
            "We detected that you are passing `past_key_values` as a tuple and this is deprecated and will be removed in v4.43. "
            "Please use an appropriate `Cache` class (https://huggingface.co/docs/transformers/v4.41.3/en/internal/generation_utils#transformers.Cache)"
        )

    if cache_position is None:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        #### SAGEMAKER CHANGE
        # cache_position = torch.arange(
        #     past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
        # )
        from torch.sagemaker.state_handler import state

        total_seqlen = inputs_embeds.shape[1] * state.cp_size
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + total_seqlen, device=inputs_embeds.device, dtype=torch.float32
        )
        #### END SAGEMAKER CHANGE

    if position_ids is None:
        position_ids = cache_position.unsqueeze(0)

    causal_mask = self._update_causal_mask(
        attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
    )
    hidden_states = inputs_embeds

    # create position embeddings to be shared across the decoder layers
    position_embeddings = self.rotary_emb(hidden_states, position_ids)

    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    next_decoder_cache = None

    for decoder_layer in self.layers:
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if self.gradient_checkpointing and self.training:
            layer_outputs = self._gradient_checkpointing_func(
                decoder_layer.__call__,
                hidden_states,
                causal_mask,
                position_ids,
                past_key_values,
                output_attentions,
                use_cache,
                cache_position,
                position_embeddings,
            )
        else:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )

        hidden_states = layer_outputs[0]

        if use_cache:
            next_decoder_cache = layer_outputs[2 if output_attentions else 1]

        if output_attentions:
            all_self_attns += (layer_outputs[1],)

    hidden_states = self.norm(hidden_states)

    # add hidden states from the last decoder layer
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    next_cache = next_decoder_cache if use_cache else None
    if return_legacy_cache:
        next_cache = next_cache.to_legacy_cache()

    if not return_dict:
        return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )
