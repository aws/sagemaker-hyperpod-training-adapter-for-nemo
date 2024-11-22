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
# Portions taken from https://github.com/NVIDIA/Megatron-LM, Copyright Nvidia Corporation

"""Train utils."""

import functools

import torch
import torch.sagemaker as tsm

from hyperpod_nemo_adapter.utils.fsdp_utils import get_transformer_layer
from hyperpod_nemo_adapter.utils.log_utils import Logger

# pylint: disable=import-error,import-outside-toplevel,invalid-name,no-member,no-name-in-module,protected-access


_logger = Logger().get_logger()


def apply_activation_checkpoint(
    model=None,
    model_type=None,
    use_smp_model: bool = True,
    fp8: bool = False,
    moe: bool = False,
):
    """Apply activation checkpoint."""
    if fp8 and moe and use_smp_model:
        # Checkpoint attention and moe layers separately when using FP8 and MoE.
        # Currently, checkpointing entire TransformerLayer is not supported.
        apply_activation_checkpoint_moe(model=model)
        return

    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        CheckpointImpl,
        apply_activation_checkpointing,
        checkpoint_wrapper,
    )

    transformer_layer = get_transformer_layer(model_type, use_smp_model, moe=moe)
    check_fn_gpt = lambda submodule: isinstance(  # pylint: disable=unnecessary-lambda-assignment
        submodule, transformer_layer
    )

    if fp8 and use_smp_model:
        import transformer_engine

        checkpoint_fn = functools.partial(
            transformer_engine.pytorch.checkpoint,
            distribute_saved_activations=False,
            get_rng_state_tracker=tsm.state.get_rng_state_tracker,
            tp_group=tsm.state.tp_process_group,
            use_reentrant=False,
        )
        checkpoint_impl = CheckpointImpl.NO_REENTRANT
    else:
        checkpoint_fn = None
        checkpoint_impl = CheckpointImpl.REENTRANT

    # flash attn v2 does not work with no_reentrant
    # our activation offloading for 2.0 also does not work with no_reentrant
    entrant_wrapper = functools.partial(
        checkpoint_wrapper, checkpoint_impl=checkpoint_impl, checkpoint_fn=checkpoint_fn
    )
    apply_activation_checkpointing(model, checkpoint_wrapper_fn=entrant_wrapper, check_fn=check_fn_gpt)


def apply_activation_checkpoint_moe(model=None, checkpoint_attn=True, checkpoint_moe=True):
    """
    Experimental checkpointing with multiple checkpoint wrappers.
    Use TE checkpoint for attention, and megatron/native checkpoint for MoE layer.
    """
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        CheckpointImpl,
        apply_activation_checkpointing,
        checkpoint_wrapper,
    )

    checkpoint_impl = CheckpointImpl.NO_REENTRANT

    if checkpoint_attn:
        import torch.sagemaker as tsm
        import transformer_engine
        from transformer_engine.pytorch.attention import MultiheadAttention

        check_fn_attn = lambda submodule: isinstance(  # pylint: disable=unnecessary-lambda-assignment
            submodule, MultiheadAttention
        )
        checkpoint_fn_attn = functools.partial(
            transformer_engine.pytorch.checkpoint,
            distribute_saved_activations=False,
            get_rng_state_tracker=tsm.state.get_rng_state_tracker,
            tp_group=tsm.state.tp_process_group,
            use_reentrant=False,
        )
        # flash attn v2 does not work with no_reentrant
        # our activation offloading for 2.0 also does not work with no_reentrant
        entrant_wrapper_attn = functools.partial(
            checkpoint_wrapper, checkpoint_impl=checkpoint_impl, checkpoint_fn=checkpoint_fn_attn
        )
        apply_activation_checkpointing(model, checkpoint_wrapper_fn=entrant_wrapper_attn, check_fn=check_fn_attn)

    if checkpoint_moe:
        from torch.sagemaker.moe.moe_layer import MoELayer

        check_fn_moe = lambda submodule: isinstance(  # pylint: disable=unnecessary-lambda-assignment
            submodule, MoELayer
        )
        checkpoint_fn_moe = None
        entrant_wrapper_moe = functools.partial(
            checkpoint_wrapper, checkpoint_impl=checkpoint_impl, checkpoint_fn=checkpoint_fn_moe
        )
        apply_activation_checkpointing(model, checkpoint_wrapper_fn=entrant_wrapper_moe, check_fn=check_fn_moe)


def get_batch_for_cp_rank(batch):
    # Based on https://github.com/NVIDIA/NeMo/blob/58d6bcee313a44d926a54e51c69222ddae20f070/nemo/collections/nlp/models/language_modeling/megatron_gpt_model.py#L840
    cp_size = tsm.state.cp_size
    cp_rank = tsm.state.cp_rank
    if cp_size > 1:
        return_batch = []
        for val in batch:
            if val is not None:
                seq_dim = 1
                val = val.view(
                    *val.shape[0:seq_dim],
                    2 * cp_size,
                    val.shape[seq_dim] // (2 * cp_size),
                    *val.shape[(seq_dim + 1) :],
                )
                index = torch.tensor([cp_rank, (2 * cp_size - cp_rank - 1)], device=val.device)
                val = val.index_select(seq_dim, index)
                val = val.view(*val.shape[0:seq_dim], -1, *val.shape[(seq_dim + 2) :])
            return_batch.append(val)
        return_batch = tuple(return_batch)
    else:
        return_batch = batch
    return return_batch


def compute_tflops(cfg, model_config, sample_processed, step_time, world_size):
    # Based on
    # https://github.com/NVIDIA/Megatron-LM/blob/ba773259dbe5735fbd91ca41e7f4ded60b335c52/megatron/training/training.py#L65
    hidden_width = model_config.hidden_size
    num_heads = model_config.num_attention_heads
    num_key_value_heads = model_config.num_key_value_heads
    moe = cfg.get("moe", 0)
    num_experts_per_tok = cfg.get("num_experts_per_tok")
    max_context_width = cfg.get("max_context_width")
    num_layers = model_config.num_hidden_layers
    intermediate_size = model_config.intermediate_size
    vocab_size = model_config.vocab_size

    kv_channels = hidden_width // num_heads
    query_projection_size = kv_channels * num_heads
    query_projection_to_hidden_size_ratio = query_projection_size / hidden_width

    # Group Query Attention.
    if not num_key_value_heads:
        num_key_value_heads = num_heads

    # MoE.
    num_experts_routed_to = 1 if moe == 0 else num_experts_per_tok
    gated_linear_multiplier = 3 / 2 if moe > 0 else 1

    # Compute the number of floating point operations
    num_flops = (
        12
        * sample_processed
        * max_context_width
        * num_layers
        * hidden_width
        * hidden_width
        * (
            # Attention.
            (
                (1 + (num_key_value_heads / num_heads) + (max_context_width / hidden_width))
                * query_projection_to_hidden_size_ratio
            )
            # MLP.
            + ((intermediate_size / hidden_width) * num_experts_routed_to * gated_linear_multiplier)
            # Logit.
            + (vocab_size / (2 * num_layers * hidden_width))
        )
    )

    # Convert to TFLOPs per GPU
    tflops_per_gpu = num_flops / (step_time * 10**12 * world_size)

    return tflops_per_gpu
