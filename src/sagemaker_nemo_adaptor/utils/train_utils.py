"""Train utils."""

import functools

import torch

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
        import torch.sagemaker as tsm
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


def patch_neox_rope(model):
    """Patch neox rope."""
    device = torch.cuda.current_device()
    for layer in model.gpt_neox.layers:
        layer.attention.rotary_emb.sin_cached = layer.attention.rotary_emb.sin_cached.to(device)
        layer.attention.rotary_emb.cos_cached = layer.attention.rotary_emb.cos_cached.to(device)
