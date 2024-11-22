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
from contextlib import ExitStack, nullcontext
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Union

import torch
import torch.distributed as dist
import torch.sagemaker as tsm
from accelerate.utils import FullyShardedDataParallelPlugin
from lightning_fabric.utilities.types import _PATH
from nemo.collections.nlp.parts.nlp_overrides import NLPFSDPStrategy
from nemo.utils import logging
from omegaconf.dictconfig import DictConfig
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
    offload_wrapper,
)
from torch.distributed.checkpoint.optimizer import load_sharded_optimizer_state_dict
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType
from torch.distributed.fsdp.api import FullOptimStateDictConfig, FullStateDictConfig
from torch.sagemaker.checkpoint.constants import (
    SMP_IS_LOAD_INFO_KEY,
    SMP_IS_PARTIAL_KEY,
)
from torch.sagemaker.distributed.checkpoint.filesystem import (
    DistributedFileSystemReader,
)
from torch.sagemaker.distributed.checkpoint.state_dict_utils import (
    SMStateDictType,
    sm_state_dict_type,
)
from torch.sagemaker.utils import utils as tsm_utils

smddp_available = True
try:
    import smdistributed.dataparallel.torch.torch_smddp  # noqa: F401
except:
    smddp_available = False

from hyperpod_nemo_adapter.constants import (
    OPTIMIZER_KEY_PREFIX,
    SageMakerCheckpointType,
)
from hyperpod_nemo_adapter.utils.app_state import SageMakerAppState
from hyperpod_nemo_adapter.utils.callbacks.checkpoint import SageMakerCheckpointIO
from hyperpod_nemo_adapter.utils.dist_utils import initialize_model_parallel_for_nemo
from hyperpod_nemo_adapter.utils.fsdp_utils import (
    get_auto_wrap_policy,
    get_backward_fetch_policy,
    get_sharding_strategy,
    get_transformer_layer,
    set_mixed_precision_recipe,
)
from hyperpod_nemo_adapter.utils.get_rank import (
    get_coordinator_rank,
    get_current_replication_group,
    is_action_rank,
)
from hyperpod_nemo_adapter.utils.gpu_utils import initialize_gpu_affinity
from hyperpod_nemo_adapter.utils.train_utils import apply_activation_checkpoint


class SageMakerFSDPStrategy(NLPFSDPStrategy):
    """
    FSDP plugin for Pytorch Lightning with the support for sharding_strategy tensor-parallelism.
    SageMakerFSDPStrategy deals with
      - Distributed initialization, including torch distributed setup, smp distributed setup
      - FSDP configuration and setup
      - Hook for checkpoint save/load
    """

    # Currently feeding everything here so we can know what to deal with for strategy class
    def __init__(
        self,
        cfg: DictConfig,
        **kwargs: Union[Any, Dict[str, Any]],
    ) -> None:
        self.cfg = cfg
        self.use_smp_model = cfg.use_smp_model
        self.smp_config_dict = self._setup_smp_config(cfg)
        self.app_state = SageMakerAppState()

        # Init from original PT-Lightning policy to avoid megatron specific initialization
        super(NLPFSDPStrategy, self).__init__(**kwargs)
        if self.cfg.distributed_backend == "smddp" and not smddp_available:
            self.cfg.distributed_backend = "nccl"
            logging.warning(
                "SMDDP library is not available within the current training environment. Falling back to NCCL backend."
            )
        self._process_group_backend = self.cfg.distributed_backend

    def _setup_smp_config(self, cfg):
        smp_config = {
            "activation_loading_horizon": cfg.model.get("activation_loading_horizon", 2),
            "sm_activation_offloading": cfg.model.get("offload_activations", False),
            # these parallel degrees are defined only when `use_smp_model=True`.
            # defaulting to 1 for case when `use_smp_model=False`:
            "tensor_parallel_degree": cfg.model.get("tensor_model_parallel_degree", 1),
            "expert_parallel_degree": cfg.model.get("expert_model_parallel_degree", 1),
            "context_parallel_degree": cfg.model.get("context_parallel_degree", 1),
            "random_seed": cfg.model.seed,
        }
        if cfg.model.get("shard_degree", None):
            smp_config["hybrid_shard_degree"] = cfg.model.shard_degree
        return smp_config

    def _record_fsdp_process_group(self, model: FSDP):
        """
        Put fsdp model process group info into app state.
        """
        self.app_state.fsdp_process_group = model.process_group
        self.app_state.is_fsdp_action_rank = is_action_rank(self.global_rank)
        self.app_state.fsdp_coordinator_rank = get_coordinator_rank(model.process_group)

    def _record_replication_process_group(self):
        """
        Put replication group info into app state.
        """
        (replication_coordinator_rank, current_replication_group) = get_current_replication_group(self.global_rank)
        self.app_state.replication_coordinator_rank = replication_coordinator_rank
        self.app_state.current_replication_group = current_replication_group

    @property
    def pytorch_model(self):
        # FSDP wrapped model.
        return self.model.model if self.model else None

    def _setup_model(self, model):
        # retrieve the root module name of the model which is the first one.
        use_smp_model = self.use_smp_model
        cfg = self.cfg.model
        predefined_model = model.predefined_model
        if not predefined_model or cfg.get("multi_modal", False):
            # When running with model that is not predefined or multimodal Llama 3.2
            # we use HF's accelerate to handle the FSDP and activation checkpoint
            # Map to HF name: https://github.com/huggingface/accelerate/blob/main/src/accelerate/utils/constants.py#L37
            if cfg.auto_wrap_policy == "transformer_auto_wrap_policy":
                auto_wrap_policy = "transformer_based_wrap"
            elif cfg.auto_wrap_policy == "size_based_auto_wrap_policy":
                auto_wrap_policy = "size_based_wrap"
            else:
                auto_wrap_policy = "no_wrap"
            fsdp_plugin = FullyShardedDataParallelPlugin(auto_wrap_policy=auto_wrap_policy)
            fsdp_plugin.set_auto_wrap_policy(model.model)
            auto_wrap_policy = fsdp_plugin.auto_wrap_policy
        else:
            transformer_layer = get_transformer_layer(cfg.model_type, use_smp_model, cfg.moe)
            auto_wrap_policy = get_auto_wrap_policy(cfg.auto_wrap_policy, transformer_layer, model.use_peft)
        mixed_precision_policy = set_mixed_precision_recipe(
            precision=cfg.precision,
            use_smp_model=use_smp_model,
            is_qlora=model.use_peft and cfg.peft.get("peft_type", None) == "qlora_4bit",
            cast_forward_inputs=model.use_peft or cfg.get("multi_modal", False),
        )

        sharding_strategy = get_sharding_strategy(cfg.sharding_strategy)
        backward_prefetch = get_backward_fetch_policy(cfg.backward_fetch_policy)
        param_init_fn, post_param_init_fn, model_context = self._setup_delayed_param(cfg, model)

        with (
            model_context,
            tsm_utils.timeit(True, "FSDP constructor", self.global_rank),
        ):
            pytorch_model = FSDP(
                module=model.model,
                auto_wrap_policy=auto_wrap_policy,
                mixed_precision=mixed_precision_policy,
                sharding_strategy=sharding_strategy,
                backward_prefetch=backward_prefetch,
                forward_prefetch=cfg.forward_prefetch,
                limit_all_gathers=cfg.limit_all_gathers,
                device_id=torch.cuda.current_device(),
                use_orig_params=cfg.use_orig_param,
                param_init_fn=param_init_fn,
                post_param_init_fn=post_param_init_fn,
                sync_module_states=model.do_finetune_with_pretrained_weights,
            )
            self._record_fsdp_process_group(pytorch_model)
            self._record_replication_process_group()

        if cfg.activation_checkpointing:
            if not predefined_model:
                # Use native PT API to apply activation checkpoint
                apply_activation_checkpointing(
                    pytorch_model,
                    checkpoint_wrapper_fn=functools.partial(
                        checkpoint_wrapper,
                        checkpoint_impl=CheckpointImpl.NO_REENTRANT,
                    ),
                    auto_wrap_policy=fsdp_plugin.auto_wrap_policy,
                )
            else:
                apply_activation_checkpoint(
                    model=pytorch_model,
                    model_type=cfg.model_type,
                    use_smp_model=use_smp_model,
                    fp8=cfg.fp8,
                    moe=cfg.moe,
                )
        if cfg.get("offload_activations", None):
            pytorch_model = offload_wrapper(pytorch_model)
        model.model = pytorch_model
        return model

    def _setup_delayed_param(self, cfg, model):
        if not cfg.get("delayed_param", None):
            return None, None, nullcontext()
        if self.use_smp_model or cfg.get("multi_modal", False):
            return self._setup_smp_delayed_param(cfg, model)
        return self._setup_non_smp_delayed_param(cfg, model)

    def _setup_non_smp_delayed_param(self, cfg, model):
        if model.do_finetune_with_pretrained_weights:
            # Pulled param initialization function from open source meta/llama training recipes
            # https://github.com/meta-llama/llama-recipes/blob/f531d17287bf11d2cc2a5992e9282c77a70b2f51/src/llama_recipes/finetuning.py#L186C13-L186C103
            param_init_fn = lambda module: module.to_empty(device=torch.device("cuda"), recurse=False)
        else:
            param_init_fn = self.model.param_init_fn
        return param_init_fn, None, nullcontext()

    def _setup_smp_delayed_param(self, cfg, model):
        # The monkey patch is applied during tsm.init(). This is the make sure the correct import
        # is called. ie: RotaryPositionEmbedding will become PatchedRotaryPositionEmbedding.
        from torch.sagemaker.delayed_param import DelayedParamIniter

        initer = None
        if model.do_finetune_with_pretrained_weights:
            if self.global_rank != 0:
                initer = DelayedParamIniter(model.model)
        else:
            initer = DelayedParamIniter(model.model)

        if not initer:
            return None, None, nullcontext()
        return (
            initer.get_param_init_fn(),
            initer.get_post_param_init_fn(),
            (
                initer.validate_params_and_buffers_inited()
                if not model.do_finetune_with_pretrained_weights
                else nullcontext()
            ),
        )

    def _setup_gpu_affinity(self):
        gpu_affinity = self.cfg.model.get("gpu_affinity", None)
        if gpu_affinity and not gpu_affinity.get("enabled", True):
            return
        initialize_gpu_affinity(self.local_rank, self.num_processes)

    def setup(self, trainer: "pl.Trainer") -> None:
        super(NLPFSDPStrategy, self).setup(trainer)
        logging.info(f"Training Model:\n{self.pytorch_model}")

    def optimizer_step(self, *a, **kw):
        logging.debug(f"Invoking optimizer_step")
        super().optimizer_step(*a, **kw)

    def setup_environment(self) -> None:
        """
        Setup distributed for SMP, and setup nemo distributing variables
        """
        # Init from original PT-Lightning policy to avoid megatron specific initialization
        super(NLPFSDPStrategy, self).setup_environment()
        self._setup_gpu_affinity()
        tsm.init(self.smp_config_dict)

        # Setup nemo distributed variables, not actually initialize megatron distributed backend
        tensor_parallel_degree = self.smp_config_dict["tensor_parallel_degree"] if self.use_smp_model else 1
        context_parallel_degree = self.cfg.model.get("context_parallel_degree", 1)
        initialize_model_parallel_for_nemo(
            world_size=self.world_size,
            global_rank=self.global_rank,
            local_rank=self.local_rank,
            tensor_model_parallel_size=tensor_parallel_degree,
            context_parallel_size=context_parallel_degree,
            seed=self.cfg.model.seed,
        )

    @property
    def sharded_model_state_dict(self):
        with FSDP.state_dict_type(self.pytorch_model, StateDictType.SHARDED_STATE_DICT):
            return self.pytorch_model.state_dict()

    @property
    def local_model_state_dict(self):
        with sm_state_dict_type(self.pytorch_model, SMStateDictType.SM_LOCAL_STATE_DICT):
            return self.pytorch_model.state_dict()

    @property
    def full_model_state_dict(self):
        state_dict_config = FullStateDictConfig(rank0_only=True, offload_to_cpu=True)
        with sm_state_dict_type(self.pytorch_model, StateDictType.FULL_STATE_DICT, state_dict_config=state_dict_config):
            return self.pytorch_model.state_dict()

    def lightning_module_state_dict(self) -> Dict[str, Any]:
        """
        Store the model state dict in one of full, sharded or local format.
        """
        assert isinstance(self.checkpoint_io, SageMakerCheckpointIO)
        typ = self.checkpoint_io.checkpoint_type
        if typ == SageMakerCheckpointType.LOCAL:
            return self.local_model_state_dict
        if typ == SageMakerCheckpointType.SHARDED:
            return self.sharded_model_state_dict
        if typ == SageMakerCheckpointType.FULL:
            return self.full_model_state_dict
        if typ == SageMakerCheckpointType.PEFT_FULL:
            return self.full_model_state_dict
        # For PEFT_SHARDED, we do not need to store the model state_dict as the adapter weights
        # are stored separately
        if typ == SageMakerCheckpointType.PEFT_SHARDED:
            return None
        raise NotImplementedError(f"Checkpoint type '{typ}' not implemented")

    def sharded_optimizer_state_dict(self, optimizer: torch.optim.Optimizer):
        with FSDP.state_dict_type(self.pytorch_model, StateDictType.SHARDED_STATE_DICT):
            return FSDP.optim_state_dict(self.pytorch_model, optimizer)

    def local_optimizer_state_dict(self, optimizer: torch.optim.Optimizer):
        with sm_state_dict_type(self.pytorch_model, SMStateDictType.SM_LOCAL_STATE_DICT):
            return optimizer.state_dict()

    def full_optimizer_state_dict(self, optimizer: torch.optim.Optimizer):
        optim_state_dict_config = FullOptimStateDictConfig(rank0_only=True, offload_to_cpu=True)
        with sm_state_dict_type(
            self.pytorch_model, StateDictType.FULL_STATE_DICT, optim_state_dict_config=optim_state_dict_config
        ):
            return FSDP.optim_state_dict(self.pytorch_model, optimizer)

    def optimizer_state(self, optimizer: torch.optim.Optimizer) -> Dict[str, torch.Tensor]:
        """
        Store the optimizer state dict in one of full, sharded or local format.
        """
        assert isinstance(self.checkpoint_io, SageMakerCheckpointIO)
        typ = self.checkpoint_io.checkpoint_type
        if typ == SageMakerCheckpointType.LOCAL:
            return self.local_optimizer_state_dict(optimizer)
        if typ == SageMakerCheckpointType.SHARDED or typ == SageMakerCheckpointType.PEFT_SHARDED:
            return self.sharded_optimizer_state_dict(optimizer)
        if typ == SageMakerCheckpointType.FULL:
            return self.full_optimizer_state_dict(optimizer)
        if typ == SageMakerCheckpointType.PEFT_FULL:
            return self.full_optimizer_state_dict(optimizer)
        raise NotImplementedError(f"Checkpoint type '{typ}' not implemented")

    def load_local_model_state_dict(self, checkpoint):
        pass

    def load_sharded_model_state_dict(self, checkpoint):
        assert "state_dict" in checkpoint, "Missing model 'state_dict'"
        with FSDP.state_dict_type(self.pytorch_model, StateDictType.SHARDED_STATE_DICT):
            self.pytorch_model.load_state_dict(checkpoint["state_dict"])

    def load_full_model_state_dict(self, checkpoint):
        assert "state_dict" in checkpoint, "Missing model 'state_dict'"
        state_dict_config = FullStateDictConfig(rank0_only=True, offload_to_cpu=True)
        with sm_state_dict_type(self.pytorch_model, StateDictType.FULL_STATE_DICT, state_dict_config=state_dict_config):
            self.pytorch_model.load_state_dict(checkpoint["state_dict"])

    def load_model_state_dict(self, checkpoint: Mapping[str, Any], strict=None) -> None:
        assert isinstance(self.checkpoint_io, SageMakerCheckpointIO)
        typ = self.checkpoint_io.checkpoint_type
        if typ == SageMakerCheckpointType.LOCAL:
            return self.load_local_model_state_dict(checkpoint)
        if typ == SageMakerCheckpointType.SHARDED:
            return self.load_sharded_model_state_dict(checkpoint)
        if typ == SageMakerCheckpointType.FULL:
            return self.load_full_model_state_dict(checkpoint)
        raise NotImplementedError(f"Checkpoint type '{typ}' not implemented")

    def load_sharded_optim_state_dict(self, trainer, checkpoint, path):
        typ = self.checkpoint_io.checkpoint_type
        # For PEFT_SHARDED, the checkpoint does not contain the model state_dict
        # Use the sharded_model_state_dict as the checkpoint adapter weights will have been loaded in at this point
        if typ == SageMakerCheckpointType.PEFT_SHARDED:
            checkpoint_state_dict = self.sharded_model_state_dict
        else:
            checkpoint_state_dict = checkpoint["state_dict"]
        for i, optimizer in enumerate(trainer.optimizers):
            optimizer_key = f"{OPTIMIZER_KEY_PREFIX}_{i}"
            state_dict = load_sharded_optimizer_state_dict(
                model_state_dict=checkpoint_state_dict,
                optimizer_key=optimizer_key,
                storage_reader=DistributedFileSystemReader(path),
                process_group=self.pytorch_model.process_group,
            )
            flattened_osd = FSDP.optim_state_dict_to_load(
                model=self.pytorch_model, optim=optimizer, optim_state_dict=state_dict[optimizer_key]
            )
            optimizer.load_state_dict(flattened_osd)

    def load_local_optim_state_dict(self, trainer, checkpoint, path):
        """Load local optimizer param_groups from state_dict.

        In SageMakerLocalCheckpointIO _load function, the optimizer's states are loaded during
        dcp loading, since states are tensors which are mutable and can be loaded in place.

        In the param groups,
        there are floats/ints that need to be loaded separately, since they are immutable.

        Note: If optimizer.load_state_dict() is callled, optimizer's states will be loaded twice.
        This function avoids the duplicate loading and ONLY load optimizer param_groups.
        """

        def update_group(saved_group: Dict[str, Any], group: Dict[str, Any]) -> Dict[str, Any]:
            saved_group["params"] = group["params"]
            return saved_group

        for i, optimizer in enumerate(trainer.optimizers):
            groups = optimizer.param_groups
            saved_groups = deepcopy(checkpoint["optimizer_states"][i]["param_groups"])

            param_groups = [update_group(g, ng) for g, ng in zip(saved_groups, groups)]
            optimizer.__setstate__({"param_groups": param_groups})

    def load_optimizer_state_dict(
        self,
        trainer,
        checkpoint: Mapping[str, Any],
        path,
        **kw,
    ) -> None:
        assert isinstance(self.checkpoint_io, SageMakerCheckpointIO)
        typ = self.checkpoint_io.checkpoint_type
        if typ == SageMakerCheckpointType.LOCAL:
            return self.load_local_optim_state_dict(trainer, checkpoint, path)
        if typ == SageMakerCheckpointType.SHARDED or typ == SageMakerCheckpointType.PEFT_SHARDED:
            return self.load_sharded_optim_state_dict(trainer, checkpoint, path)
        raise NotImplementedError(f"Checkpoint type '{typ}' not implemented")

    def save_checkpoint(
        self,
        checkpoint: Dict[str, Any],
        filepath: Union[str, Path],
        storage_options: Optional[Any] = None,
    ) -> None:
        """Store checkpoints
        1. In case of sharded checkpoint, all ranks store unique checkpoints.
        2. In case of non-sharded checkpoint, all data-parallel rank 0 store checkpoints.
        """
        self.checkpoint_io.save_checkpoint(checkpoint, filepath, storage_options)

    def load_checkpoint(
        self,
        checkpoint_path: _PATH,
        *args,
        **kwargs,
    ) -> Dict[str, Any]:
        """Load checkpoints"""
        assert isinstance(self.checkpoint_io, SageMakerCheckpointIO)
        return self.checkpoint_io.load_checkpoint(checkpoint_path, *args, **kwargs)

    def remove_checkpoint(self, filepath: Union[str, Path]) -> None:
        """Remove checkpoints"""
        return

    @property
    def restore_checkpoint_after_setup(self) -> bool:
        """When loading FSDP-sharded checkpoint, need to restore checkpoint after configuring
        FSDP sharding to match FSDP-sharded format between the checkpoint and the current
        model and optimizer.
        """
        return True

    def save_peft_model(self, checkpoint_dir):
        """
        Save the FSDP wrapped PEFT model to the specified directory. Note that this method will
        only save the adapter weights.
        """
        logging.info(f"Saving PEFT checkpoint to {checkpoint_dir}")
        logging.debug(f"Model to save: {self.pytorch_model}")

        def is_peft_adapter(module):
            return (
                not list(module.named_children())
                and getattr(module, "weight", None) is not None
                and module.weight.requires_grad
            )

        def is_peft_fsdp_wrapper(module):
            return hasattr(module, "_fsdp_wrapped_module") and is_peft_adapter(module._fsdp_wrapped_module)

        adapter_modules = list(filter(is_peft_fsdp_wrapper, self.pytorch_model.modules()))
        context_managers = [
            FSDP.summon_full_params(
                module,
                writeback=False,
                rank0_only=True,
                offload_to_cpu=True,
            )
            for module in adapter_modules
        ]

        """
        we don't want to use FSDP FULL state dict because gathering of frozen params
        is needlessly expensive and causes OOM issues. we also need to avoid the FSDP
        state_dict hooks as they won't return full tensors even with summon_full_params.
        so we use SM_LOCAL_STATE_DICT to disable FSDP state_dict hooks.
        """
        with ExitStack() as stack, sm_state_dict_type(self.pytorch_model, SMStateDictType.SM_LOCAL_STATE_DICT):
            for cm in context_managers:
                stack.enter_context(cm)
            if dist.get_rank() == 0:
                """
                Need to extract the PEFT model from the FSDP wrapper to call save_pretrained()

                Example of what the _fsdp_wrapped_module looks like:
                    FullyShardedDataParallel(
                        (_fsdp_wrapped_module): PeftModelForCausalLM(

                The model needs to be unwrapped in order to extract the PeftModelForCausalLM
                """
                self.pytorch_model.module.save_pretrained(checkpoint_dir)
            dist.barrier()
