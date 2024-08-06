"""Train utils."""

import functools
import math

import numpy as np
import torch
import torch.distributed as dist
from torch.nn import LayerNorm
from transformers.models.llama.modeling_llama import LlamaRMSNorm

from sagemaker_nemo_adaptor.utils.fsdp_utils import get_transformer_layer

# pylint: disable=import-error,import-outside-toplevel,invalid-name,no-member,no-name-in-module,protected-access

from sagemaker_nemo_adaptor.utils.log_utils import Logger
_logger = Logger().get_logger()

def compute_num_params(model):
    """Get num params."""
    num_params = 0
    seen = set()
    for p in model.parameters():  # pylint: disable=invalid-name
        if p not in seen:
            seen.add(p)
            if hasattr(p, "ds_shape"):
                num_params += np.prod(p.ds_shape)
            else:
                num_params += np.prod(p.size())

    return num_params


def compute_tflops(args, global_batch_size, step_time, world_size):
    # Based on
    # https://github.com/NVIDIA/Megatron-LM/blob/ba773259dbe5735fbd91ca41e7f4ded60b335c52/megatron/training/training.py#L65
    # Attention projection size.
    kv_channels = args.hidden_width // args.num_heads
    query_projection_size = kv_channels * args.num_heads
    query_projection_to_hidden_size_ratio = query_projection_size / args.hidden_width

    # Group Query Attention.
    if not args.num_key_value_heads:
        args.num_key_value_heads = args.num_heads

    # MoE.
    num_experts_routed_to = 1 if args.moe == 0 else args.num_experts_per_tok
    gated_linear_multiplier = 3 / 2 if args.moe > 0 else 1

    # Compute the number of floating point operations
    num_flops = (
        12
        * global_batch_size
        * args.max_context_width
        * args.num_layers
        * args.hidden_width
        * args.hidden_width
        * (
            # Attention.
            (
                (1 + (args.num_key_value_heads / args.num_heads) + (args.max_context_width / args.hidden_width))
                * query_projection_to_hidden_size_ratio
            )
            # MLP.
            + ((args.intermediate_size / args.hidden_width) * num_experts_routed_to * gated_linear_multiplier)
            # Logit.
            + (args.vocab_size / (2 * args.num_layers * args.hidden_width))
        )
    )

    # Convert to TFLOPs per GPU
    tflops_per_gpu = num_flops / (step_time * 10**12 * world_size)

    return tflops_per_gpu


# CheTODO: check whether we still need this given we have opt setup from megatron
def get_param_groups_by_weight_decay(module):
    """Get param groups."""
    weight_decay_params = {"params": []}
    no_weight_decay_params = {"params": [], "weight_decay": 0.0}
    param_ids = set()

    for module_ in module.modules():
        # if isinstance(module_, FusedLayerNorm) or
        if isinstance(module_, (LayerNorm, LlamaRMSNorm)):
            for p in list(module_._parameters.values()):  # pylint: disable=invalid-name,protected-access
                if p is not None and id(p) not in param_ids:
                    no_weight_decay_params["params"].append(p)
                    param_ids.add(id(p))
        else:
            for n, p in list(module_._parameters.items()):  # pylint: disable=invalid-name,protected-access
                if p is not None and n != "bias" and id(p) not in param_ids:
                    weight_decay_params["params"].append(p)
                    param_ids.add(id(p))
            for n, p in list(module_._parameters.items()):  # pylint: disable=invalid-name,protected-access
                if p is not None and n == "bias" and id(p) not in param_ids:
                    no_weight_decay_params["params"].append(p)
                    param_ids.add(id(p))
    return weight_decay_params, no_weight_decay_params


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


# pylint: disable=invalid-name
class AnnealingLR:  # pylint: disable=too-many-instance-attributes
    """Anneals the learning rate."""

    def __init__(  # pylint: disable=too-many-arguments
        self,
        optimizer,
        start_lr,
        warmup_iter,
        plateau_iter,
        total_iters,
        decay_style,
        last_iter,
        min_lr=0.0,
        use_checkpoint_lr_scheduler=True,
        override_lr_scheduler=False,
    ):
        # Class values.
        self.optimizer = optimizer
        self.start_lr = start_lr
        self.min_lr = min_lr
        self.warmup_iter = warmup_iter
        self.plateau_iter = plateau_iter
        self.num_iters = last_iter
        self.end_iter = total_iters
        assert self.end_iter > 0
        self.decay_style = decay_style
        self.override_lr_scheduler = override_lr_scheduler
        self.use_checkpoint_lr_scheduler = use_checkpoint_lr_scheduler
        if self.override_lr_scheduler:
            assert not self.use_checkpoint_lr_scheduler, "both override and " "use-checkpoint are set."
        # Set the learning rate
        self.step(self.num_iters)
        self.rank = dist.get_rank()

    def get_lr(self):
        """Learning rate decay functions from:
        https://openreview.net/pdf?id=BJYwwY9ll pg. 4"""

        num_iters_ = min(self.num_iters, self.end_iter - self.warmup_iter)
        # Warmup.
        if self.warmup_iter > 0 and self.num_iters <= self.warmup_iter:
            return float(self.start_lr) * num_iters_ / self.warmup_iter

        num_iters_ = num_iters_ - self.warmup_iter
        if self.decay_style == "linear":
            lr = self.start_lr * (self.end_iter - num_iters_) / self.end_iter
        elif self.decay_style == "plateau":
            if self.num_iters <= self.plateau_iter:
                lr = self.start_lr
            else:
                lr = self.start_lr * (self.end_iter - self.num_iters) / (self.end_iter - self.plateau_iter)
        elif self.decay_style == "cosine":
            lr = self.start_lr / 2.0 * (math.cos(math.pi * num_iters_ / self.end_iter) + 1)
        elif self.decay_style == "exponential":
            # exp(-0.693) = 1/2
            lr = self.start_lr * math.exp(-0.693 * num_iters_ / self.end_iter)
        else:
            lr = self.start_lr
        return max(lr, self.min_lr)

    def step(self, step_num=None):
        """Set lr for all parameters groups."""
        if step_num is None:
            step_num = self.num_iters + 1
        self.num_iters = step_num
        new_lr = self.get_lr()
        for group in self.optimizer.param_groups:
            group["lr"] = new_lr

    def state_dict(self):
        """State dict."""
        state_dict = {
            "start_lr": self.start_lr,
            "warmup_iter": self.warmup_iter,
            "num_iters": self.num_iters,
            "decay_style": self.decay_style,
            "end_iter": self.end_iter,
            "min_lr": self.min_lr,
        }
        return state_dict

    def _check_and_set(self, cls_value, sd_value, name):
        """Auxiliary function for checking the values in the checkpoint and
        setting them."""
        if self.override_lr_scheduler:
            if self.rank == 0:
                _logger.info(f"Overriding {name} value to {cls_value}")
            return cls_value

        if not self.use_checkpoint_lr_scheduler:
            assert (
                cls_value == sd_value
            ), f"AnnealingLR: class input value and checkpoint values for {name} do not match"
        if self.rank == 0:
            _logger.info(f" > using checkpoint value {sd_value} for {name}")
        return sd_value

    def load_state_dict(self, sd):
        """Load state dict."""
        self.start_lr = self._check_and_set(self.start_lr, sd["start_lr"], "learning rate")
        self.min_lr = self._check_and_set(self.min_lr, sd["min_lr"], "minimum learning rate")
        self.warmup_iter = self._check_and_set(self.warmup_iter, sd["warmup_iter"], "warmup iterations")
        self.end_iter = self._check_and_set(self.end_iter, sd["end_iter"], "total number of iterations")
        self.decay_style = self._check_and_set(self.decay_style, sd["decay_style"], "decay style")

        self.num_iters = sd["num_iters"]
        self.step(self.num_iters)
