from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Union

import torch
from nemo.collections.nlp.parts import utils_funcs
from nemo.collections.nlp.parts.nlp_overrides import NLPFSDPStrategy
from omegaconf.dictconfig import DictConfig

from sagemaker_nemo_adaptor.utils.dist_utils import initialize_model_parallel_for_nemo
from sagemaker_nemo_adaptor.utils.fsdp_utils import (
    get_auto_wrap_policy,
    get_backward_fetch_policy,
    get_sharding_strategy,
    get_transformer_layer,
)


class SageMakerFSDPStrategy(NLPFSDPStrategy):
    """
    FSDP plugin for Pytorch Lightning with the support forsharding_strategy tensor-parallelism.
    SageMakerFSDPStrategy deals with
      - Distributed initialization, including torch distributed setup, smp distributed setup
      - FSDP configuration and setup TODO: currently doing this within model class, we should revisit this
      - Hook for checkpoint save/load (TODO: revisit when implementing checkpoint)
    """

    # TODO: We need to figure out the best way of passing these arguments, need to revisit this during implementing recipe checker.
    # Currently feeding everything here so we can know what to deal with for strategy class
    def __init__(
        self,
        cfg: DictConfig,
        use_smp: bool = True,
        smp_config_dict: Dict = None,
        # FSDP args
        **kwargs: Union[Any, Dict[str, Any]],
    ) -> None:
        self.cfg = cfg
        self.use_smp = use_smp
        self.smp_config_dict = smp_config_dict

        if self.use_smp:
            from torch.distributed.fsdp import MixedPrecision

            def _set_mixed_precision_recipe(
                self, precision: Union[int, str], grad_reduce_dtype: Union[int, str], set_buffer_dtype: Union[int, str]
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
                    buffer_dtype = torch.float32 if self.use_smp else param_dtype
                return MixedPrecision(
                    param_dtype=param_dtype,
                    reduce_dtype=reduce_dtype,
                    buffer_dtype=buffer_dtype,
                )

        # # Set the mixed precision recipe TODO: Uncomment this once we moved FSDP setup back to strategy
        # kwargs["mixed_precision"] = self._set_mixed_precision_recipe(
        #     precision, grad_reduce_dtype, set_buffer_dtype=set_buffer_dtype
        # )

        # # Set FSDP configurations
        # kwargs["backward_prefetch"] = get_backward_fetch_policy(backward_fetch_policy)
        # transformer_layer = get_transformer_layer(model_type, use_smp, moe)
        # kwargs["auto_wrap_policy"] = get_auto_wrap_policy(auto_wrap_policy, transformer_layer)
        # kwargs["sharding_strategy"] = get_sharding_strategy(sharding_strategy)
        # kwargs["forward_prefetch"] = forward_prefetch
        # # use_orig_params needs to be True for SMP transformer engine usecase
        # kwargs["use_orig_params"] = True if use_smp else use_orig_params
        # Set FSDP state dict configs
        # self.sharded_checkpoint = sharded_checkpoint
        # self.state_dict_context = (
        #     _get_sharded_state_dict_context if sharded_checkpoint else _get_full_state_dict_context
        # ) TODO: implement when doing checkpoint

        # Init from original PT-Lightning policy to avoid megatron specific initialization
        super(NLPFSDPStrategy, self).__init__(**kwargs)

    def _set_mixed_precision_recipe(
        self, precision: Union[int, str], grad_reduce_dtype: Union[int, str], set_buffer_dtype: Union[int, str]
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
            buffer_dtype = torch.float32 if self.use_smp else param_dtype

        return MixedPrecision(
            param_dtype=param_dtype,
            reduce_dtype=reduce_dtype,
            buffer_dtype=buffer_dtype,
        )

    def _setup_model(self, model: torch.nn.Module) -> torch.nn.Module:
        """Wraps the model into a :class:`~torch.distributed.fsdp.fully_sharded_data_parallel.FullyShardedDataParallel`
        module.
        Over write original PT-Lightning _setup_model function to add deferred_initialization related patches.
        """

        # TODO: implement FSDP wrapping within strategy, need to check why PTL wrapped model in different way
        # TODO: check why
        # # Config delayed param init
        # if self.delayed_param:
        #     if self.finetune_with_pretrained_weights:
        #         if self.global_rank != 0:
        #             delayed_param_initer = DelayedParamIniter(model._forward_module.model)
        #     else:
        #         delayed_param_initer = DelayedParamIniter(model._forward_module.model)
        # if delayed_param_initer:
        #     param_init_fn = delayed_param_initer.get_param_init_fn()
        #     post_param_init_fn = delayed_param_initer.get_post_param_init_fn()
        # else:
        #     param_init_fn = None
        #     post_param_init_fn = None

        # from torch.distributed.fsdp import FullyShardedDataParallel

        # assert self.lightning_module is not None
        # if "auto_wrap_policy" in self.kwargs and any(
        #     isinstance(mod, FullyShardedDataParallel) for mod in self.lightning_module.modules()
        # ):
        #     del self.kwargs["auto_wrap_policy"]

        # # TODO: use a new logger
        # # log.debug(f"setting up FSDP model with device id: {self.root_device.index}, kwargs: {self.kwargs}")
        # with (
        #     delayed_param_initer.validate_params_and_buffers_inited()
        #     if (delayed_param_initer and not self.finetune_with_pretrained_weights)
        #     else nullcontext()
        # ):
        #     model = FullyShardedDataParallel(
        #         module=model,
        #         cpu_offload=self.cpu_offload,
        #         mixed_precision=self.mixed_precision_config,
        #         device_id=self.root_device.index,
        #         param_init_fn=param_init_fn,
        #         post_param_init_fn=post_param_init_fn,
        #         sync_module_states=self.finetune_with_pretrained_weights, # Todo: check this when implementing fine-tuning
        #         **self.kwargs,
        #     )

        # # activation checkpointing needs to be set up after wrapping the model
        # if _TORCH_GREATER_EQUAL_1_13 and self.activation_checkpointing:
        #     # _setup_activation_checkpointing(model, self._activation_checkpointing_kwargs)
        #     # Apply SMP specific activation checkpointing
        #     apply_activation_checkpoint(
        #         model=model,
        #         model_type=self.model_type,
        #         use_smp=self.use_smp,
        #         fp8=self.fp8,
        #         moe=self.moe,
        #     )

        # if self.offload_activations:
        #     from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import offload_wrapper
        #     model = offload_wrapper(model)
        #     if self.use_smp and self.model_type == "gpt_neox" and self.patch_neox_rope > 0:
        #         patch_neox_rope(model)
        return model

    def setup(self, trainer: "pl.Trainer") -> None:
        super(NLPFSDPStrategy, self).setup(trainer)

    def setup_environment(self) -> None:
        """
        Setup distributed for SMP, and setup nemo distributing variables
        """
        # Init from original PT-Lightning policy to avoid megatron specific initialization
        super(NLPFSDPStrategy, self).setup_environment()

        # Initialize smp, todo: check whether we still need this for HF case
        if self.use_smp:
            import torch.sagemaker as tsm

            tsm.init(self.smp_config_dict)

        # Setup nemo distributed variables, not actually initialize megatron distributed backend
        initialize_model_parallel_for_nemo(
            world_size=self.world_size,
            global_rank=self.global_rank,
            local_rank=self.local_rank,
            tensor_model_parallel_size=self.smp_config_dict["tensor_parallel_degree"] if self.use_smp else 1,
            seed=self.cfg.model.seed,
        )

    def lightning_module_state_dict(self) -> Dict[str, Any]:
        """
        Store the model state dict in one of full or sharded format.
        """
        # TODO: add when implementing checkpoint
        return

    def optimizer_state(self, optimizer: torch.optim.Optimizer) -> Dict[str, torch.Tensor]:
        """
        Store the full optimizer state dict in one of full or sharded format.
        """
        # TODO: add when implementing checkpoint
        return

    def load_model_state_dict(self, checkpoint: Mapping[str, Any], strict=None) -> None:
        # Release strict state dict matching when using Megatron AMP-O2 to skip matching
        # half-precision module wrapper module.
        # TODO: add when implementing checkpoint
        return

    def load_optimizer_state_dict(self, checkpoint: Mapping[str, Any]) -> None:
        """
        Re-key the full optimizer state dict to sharded optimizer state dict
        """

        # TODO: add when implementing checkpoint
        return

    def save_checkpoint(
        self, checkpoint: Dict[str, Any], filepath: Union[str, Path], storage_options: Optional[Any] = None
    ) -> None:
        """Store checkpoints
        1. In case of sharded checkpoint, all ranks store unique checkpoints.
        2. In case of non-sharded checkpoint, all data-parallel rank 0 store checkpoints.
        """
        # TODO: add when implementing checkpoint
        return

    def load_checkpoint(self, checkpoint_path: Union[str, Path]) -> Dict[str, Any]:
        """Load checkpoints"""
        # TODO: add when implementing checkpoint
        return

    def remove_checkpoint(self, filepath: Union[str, Path]) -> None:
        """Remove checkpoints"""
        # TODO: add when implementing checkpoint
        return

    @property
    def restore_checkpoint_after_setup(self) -> bool:
        """When loading FSDP-sharded checkpoint, need to restore checkpoint after configuring
        FSDP sharding to match FSDP-sharded format between the checkpoint and the current
        model and optimizer.
        """
        # TODO: add when implementing checkpoint
        return True
