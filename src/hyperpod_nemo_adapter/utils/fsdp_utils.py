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

import functools
from distutils.version import LooseVersion
from typing import Union

import torch
from nemo.collections.nlp.parts import utils_funcs
from peft.tuners import PrefixEncoder, PromptEmbedding, PromptEncoder
from torch.distributed.fsdp import BackwardPrefetch, MixedPrecision, ShardingStrategy
from torch.distributed.fsdp.wrap import (
    _or_policy,
    lambda_auto_wrap_policy,
    size_based_auto_wrap_policy,
    transformer_auto_wrap_policy,
)

from hyperpod_nemo_adapter.utils.log_utils import Logger

_logger = Logger().get_logger()


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


def get_auto_wrap_policy(policy: str, transformer_layer=None, use_peft=False):
    """Get auto wrap policy"""
    if use_peft:
        # to support PEFT, create policy which wraps transformer layers, but also wraps
        # linear layers (lambda_policy_fn) and other PEFT layers.
        # when using PEFT, the original model's frozen parameters are low precision,
        # but the PEFT adapter weights are full fp32 precision. Therefore, the PEFT
        # adapter layers must be wrapped separately from frozen layers, to avoid FSDP errors:
        # "ValueError: Must flatten tensors with uniform dtype but got torch.bfloat16 and torch.float32"
        assert (
            policy == "transformer_auto_wrap_policy"
        ), f"PEFT requires 'transformer_auto_wrap_policy' but got '{policy}'"

        def lambda_policy_fn(module):
            if (
                not list(module.named_children())
                and getattr(module, "weight", None) is not None
                and module.weight.requires_grad
            ):
                return True
            return False

        lambda_policy = functools.partial(lambda_auto_wrap_policy, lambda_fn=lambda_policy_fn)
        transformer_wrap_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls=(
                transformer_layer,
                PrefixEncoder,
                PromptEncoder,
                PromptEmbedding,
            ),
        )

        return functools.partial(_or_policy, policies=[lambda_policy, transformer_wrap_policy])
    else:
        if policy == "transformer_auto_wrap_policy":
            return functools.partial(
                transformer_auto_wrap_policy,
                transformer_layer_cls=(transformer_layer,),
            )
        elif policy == "size_based_auto_wrap_policy":
            return functools.partial(
                size_based_auto_wrap_policy,
            )
        else:
            raise NotImplementedError(
                f"{policy} is not a valid auto wrap policy, supported policies are: [transformer_auto_wrap_policy, size_based_auto_wrap_policy]"
            )


def get_transformer_layer(model_type="gpt2", use_smp_model=False, moe=False):
    """Get transformer layer."""
    if use_smp_model:
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
    elif "llama_v2" in model_type or "llama_v3" in model_type:
        from transformers.models.llama.modeling_llama import LlamaDecoderLayer

        transformer_layer = LlamaDecoderLayer

    elif model_type == "mistral":
        from transformers.models.mistral.modeling_mistral import MistralDecoderLayer

        transformer_layer = MistralDecoderLayer
    elif model_type == "mixtral":
        from transformers.models.mixtral.modeling_mixtral import MixtralDecoderLayer

        transformer_layer = MixtralDecoderLayer
    elif "qwen_v2" in model_type:
        from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer

        transformer_layer = Qwen2DecoderLayer

    if transformer_layer == None:
        raise Exception(f"transformer_layer for model type {model_type} not defined.")

    return transformer_layer


def set_mixed_precision_recipe(
    precision: Union[int, str],
    grad_reduce_dtype: Union[int, str] = None,
    set_buffer_dtype: Union[int, str] = None,
    use_smp_model: bool = True,
    is_qlora: bool = False,
    cast_forward_inputs: bool = False,
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
    if is_qlora:
        # QLoRA does not support mixed precision policy, has its own casts internally
        return None
    param_dtype = torch.get_default_dtype()
    if precision == 16:
        param_dtype = reduce_dtype = torch.float16
    elif precision == "bf16":
        param_dtype = reduce_dtype = torch.bfloat16
    elif precision == 32:
        param_dtype = reduce_dtype = torch.float
    # Over-write gradient reduction dtype to support bf16 computation with fp32 grad reduction
    if grad_reduce_dtype is not None:
        reduce_dtype = utils_funcs.torch_dtype_from_precision(grad_reduce_dtype, None)
    # Some models in HF such as llama hard code buffers to fp32,
    # to be similar with that we set this to fp32 unless specified by user
    if set_buffer_dtype is not None:
        buffer_dtype = utils_funcs.torch_dtype_from_precision(set_buffer_dtype, None)
    else:
        buffer_dtype = torch.float32 if use_smp_model else param_dtype
    return MixedPrecision(
        param_dtype=param_dtype,
        reduce_dtype=reduce_dtype,
        buffer_dtype=buffer_dtype,
        cast_forward_inputs=cast_forward_inputs,
    )
