"""Train utils."""

import functools

import torch
import torch.sagemaker as tsm

from sagemaker_nemo_adaptor.utils.fsdp_utils import get_transformer_layer
from sagemaker_nemo_adaptor.utils.log_utils import Logger

# pylint: disable=import-error,import-outside-toplevel,invalid-name,no-member,no-name-in-module,protected-access


_logger = Logger().get_logger()


def apply_activation_checkpoint(
    model=None,
    model_type=None,
    use_smp: bool = True,
    fp8: bool = False,
    moe: bool = False,
):
    """Apply activation checkpoint."""
    if fp8 and moe and use_smp:
        # Checkpoint attention and moe layers separately when using FP8 and MoE.
        # Currently, checkpointing entire TransformerLayer is not supported.
        apply_activation_checkpoint_moe(model=model)
        return

    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        CheckpointImpl,
        apply_activation_checkpointing,
        checkpoint_wrapper,
    )

    transformer_layer = get_transformer_layer(model_type, use_smp, moe=moe)
    check_fn_gpt = lambda submodule: isinstance(  # pylint: disable=unnecessary-lambda-assignment
        submodule, transformer_layer
    )

    if fp8 and use_smp:
        import transformer_engine

        checkpoint_fn = functools.partial(
            transformer_engine.pytorch.checkpoint,
            distribute_saved_activations=False,
            get_cuda_rng_tracker=tsm.state.get_rng_state_tracker,
            tp_group=tsm.state.tp_process_group,
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
    # TODO: 1. work with rubik to get the final one
    # TODO: 2. add license
    # Based on https://tiny.amazon.com/1bcmbuhje/githNVIDNeMoblob58d6nemocoll
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


def patch_neox_rope(model):
    """Patch neox rope."""
    device = torch.cuda.current_device()
    for layer in model.gpt_neox.layers:
        layer.attention.rotary_emb.sin_cached = layer.attention.rotary_emb.sin_cached.to(device)
        layer.attention.rotary_emb.cos_cached = layer.attention.rotary_emb.cos_cached.to(device)
