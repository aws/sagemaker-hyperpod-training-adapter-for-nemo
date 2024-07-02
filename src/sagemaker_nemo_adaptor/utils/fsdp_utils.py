import functools
from distutils.version import LooseVersion
from typing import Union

import torch
from nemo.collections.nlp.parts import utils_funcs
from torch.distributed.fsdp import BackwardPrefetch, MixedPrecision, ShardingStrategy
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
    transformer_auto_wrap_policy,
)
from torch.sagemaker.logger import get_logger

_logger = get_logger()  # Todo: use more generic logger for kandinsky


def get_sharding_strategy(strategy: str):
    """Get sharding strategy."""
    sharding_strategy = getattr(ShardingStrategy, strategy.upper())
    _logger.debug("Translating %s to %s.", strategy, sharding_strategy)
    return sharding_strategy


def get_backward_fetch_policy(policy: str):
    """Get backward fetch policy."""
    backward_fetch_policy = getattr(BackwardPrefetch, policy.upper())
    _logger.debug("Translating %s to %s.", policy, backward_fetch_policy)
    return backward_fetch_policy


def get_auto_wrap_policy(policy: str, transformer_layer=None):
    """Get auto wrap policy"""
    if policy == "transformer_auto_wrap_policy":
        return functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={
                transformer_layer,
            },
        )
    elif policy == "size_based_auto_wrap_policy":
        return functools.partial(
            size_based_auto_wrap_policy,
        )
    else:
        raise NotImplementedError(
            f"{policy} is not a valid auto wrap policy, supported policies are: [transformer_auto_wrap_policy, size_based_auto_wrap_policy]"
        )


def get_transformer_layer(model_type="gpt2", use_smp=False, moe=False):
    """Get transformer layer."""
    if use_smp and not moe:
        # For pt-2.1-tsm-2.1 releases and below,
        # We can't checkpoint our transformer.TransformerLayer class as it takes a tuple as input,
        # so we checkpoint the te.TETransformerLayer directly instead.
        # In later versions, we patch TransformerEngine activation checkpointing logic in our containers
        # with some missing native PyTorch checkpoint logic and bug fixes to resolve this.
        # PT ref: https://github.com/pytorch/pytorch/blob/v2.2.0/torch/utils/checkpoint.py#L307-L319
        # TE ref: https://github.com/NVIDIA/TransformerEngine/blob/v1.2.1/transformer_engine/pytorch/distributed.py#L272
        if LooseVersion(torch.__version__) >= LooseVersion("2.2.0"):
            from torch.sagemaker.tensor_parallel.transformer import TransformerLayer

            transformer_layer = TransformerLayer
        else:
            from torch.sagemaker.tensor_parallel.transformer import TETransformerLayer

            transformer_layer = TETransformerLayer
    elif model_type == "gpt2":
        from transformers.models.gpt2.modeling_gpt2 import GPT2Block

        transformer_layer = GPT2Block

    elif model_type == "gpt_neox":
        from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXLayer

        transformer_layer = GPTNeoXLayer

    elif model_type == "bloom":
        from transformers.models.bloom.modeling_bloom import BloomBlock

        transformer_layer = BloomBlock

    elif model_type == "flash_gptneox":
        from flash_attn.modules.block import ParallelBlock

        # TODO: Add support for Block
        transformer_layer = ParallelBlock
    elif model_type == "rubik_gpt_neox":
        from smpv1.transformer import DistributedTransformerLayer

        transformer_layer = DistributedTransformerLayer
    elif model_type == "llama_v2":
        from transformers.models.llama.modeling_llama import LlamaDecoderLayer

        transformer_layer = LlamaDecoderLayer
    elif model_type == "mistral":
        from transformers.models.mistral.modeling_mistral import MistralDecoderLayer

        transformer_layer = MistralDecoderLayer
    elif model_type == "mixtral":
        from transformers.models.mixtral.modeling_mixtral import MixtralDecoderLayer

        transformer_layer = MixtralDecoderLayer
    return transformer_layer


def set_mixed_precision_recipe(
    precision: Union[int, str],
    grad_reduce_dtype: Union[int, str] = None,
    set_buffer_dtype: Union[int, str] = None,
    use_smp: bool = True,
) -> MixedPrecision:
    """
    Set FSDP mixed precision recipe. Over-write Nemo's _set_mixed_precision_recipe function to set buffer dtype
    to fp32 in smp usecase.
    `param_dtype` sets the data type for computation in forward and backpropagation, and the parameter
    data type for optimizer execution is maintained in the full precision.
    `buffer_dtype` is only valid when a module has buffers by `register_buffer` method, which is not
    shared by FSDP.
    `reduce_dtype` sets gradient reduction data type.
    """

    if precision == 16:
        param_dtype = reduce_dtype = torch.float16
    elif precision == "bf16":
        param_dtype = reduce_dtype = torch.bfloat16
    elif precision == 32:
        param_dtype = reduce_dtype = torch.float
    else:
        raise ValueError(f"Was unable to infer precision type, received {precision!r}.")
    # Over-write gradient reduction dtype to support bf16 computation with fp32 grad reduction
    if grad_reduce_dtype is not None:
        reduce_dtype = utils_funcs.torch_dtype_from_precision(grad_reduce_dtype, None)
    # Some models in HF such as llama hard code buffers to fp32,
    # to be similar with that we set this to fp32 unless specified by user
    if set_buffer_dtype is not None:
        buffer_dtype = utils_funcs.torch_dtype_from_precision(buffer_dtype, None)
    else:
        buffer_dtype = torch.float32 if use_smp else param_dtype
    return MixedPrecision(
        param_dtype=param_dtype,
        reduce_dtype=reduce_dtype,
        buffer_dtype=buffer_dtype,
    )
