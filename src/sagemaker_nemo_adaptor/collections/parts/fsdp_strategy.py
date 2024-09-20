from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Union

import torch
import torch.distributed as dist
import torch.sagemaker as tsm
from lightning_fabric.utilities.types import _PATH
from nemo.collections.nlp.parts.nlp_overrides import NLPFSDPStrategy
from nemo.utils import logging
from omegaconf.dictconfig import DictConfig
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import offload_wrapper
from torch.distributed.checkpoint.optimizer import load_sharded_optimizer_state_dict
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType
from torch.distributed.fsdp.api import FullOptimStateDictConfig, FullStateDictConfig
from torch.sagemaker.checkpoint.constants import (
    SMP_IS_LOAD_INFO_KEY,
    SMP_IS_PARTIAL_KEY,
)
from torch.sagemaker.delayed_param import DelayedParamIniter
from torch.sagemaker.distributed.checkpoint.filesystem import (
    DistributedFileSystemReader,
)
from torch.sagemaker.distributed.checkpoint.state_dict_utils import (
    SMStateDictType,
    sm_state_dict_type,
)
from torch.sagemaker.grad_norm import clip_grad_norm_
from torch.sagemaker.utils import utils as tsm_utils

from sagemaker_nemo_adaptor.constants import (
    OPTIMIZER_KEY_PREFIX,
    SageMakerCheckpointType,
)
from sagemaker_nemo_adaptor.utils.app_state import SageMakerAppState
from sagemaker_nemo_adaptor.utils.callbacks.checkpoint import SageMakerCheckpointIO
from sagemaker_nemo_adaptor.utils.dist_utils import initialize_model_parallel_for_nemo
from sagemaker_nemo_adaptor.utils.fsdp_utils import (
    get_auto_wrap_policy,
    get_backward_fetch_policy,
    get_sharding_strategy,
    get_transformer_layer,
    set_mixed_precision_recipe,
)
from sagemaker_nemo_adaptor.utils.get_rank import (
    get_coordinator_rank,
    get_current_replication_group,
    is_action_rank,
)
from sagemaker_nemo_adaptor.utils.train_utils import (
    apply_activation_checkpoint,
    patch_neox_rope,
)


class SageMakerFSDPStrategy(NLPFSDPStrategy):
    """
    FSDP plugin for Pytorch Lightning with the support for sharding_strategy tensor-parallelism.
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
        **kwargs: Union[Any, Dict[str, Any]],
    ) -> None:
        self.cfg = cfg
        self.use_smp = cfg.use_smp
        self.smp_config_dict = self._setup_smp_config(cfg)
        self.app_state = SageMakerAppState()

        # Init from original PT-Lightning policy to avoid megatron specific initialization
        super(NLPFSDPStrategy, self).__init__(**kwargs)

    def _setup_smp_config(self, cfg):
        smp_config = {
            "activation_loading_horizon": cfg.model.activation_loading_horizon,
            "sm_activation_offloading": cfg.model.offload_activations > 0,
            # these parallel degrees are defined only when `use_smp=True`.
            # defaulting to 1 for case when `use_smp=False`:
            # https://tiny.amazon.com/ikqkw3kr/githawsprivblob1bf5srcsage
            "tensor_parallel_degree": cfg.model.get("tensor_model_parallel_degree", 1),
            "expert_parallel_degree": cfg.model.get("expert_model_parallel_degree", 1),
            "context_parallel_degree": cfg.model.get("context_parallel_degree", 1),
            "random_seed": cfg.model.seed,
        }
        if cfg.model.shard_degree:
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
        use_smp = self.use_smp
        cfg = self.cfg.model
        transformer_layer = get_transformer_layer(cfg.model_type, use_smp, cfg.moe)
        auto_wrap_policy = get_auto_wrap_policy(cfg.auto_wrap_policy, transformer_layer, model.use_peft)
        mixed_precision_policy = set_mixed_precision_recipe(
            precision=cfg.precision,
            use_smp=use_smp,
            use_peft=model.use_peft,
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
            apply_activation_checkpoint(
                model=pytorch_model,
                model_type=cfg.model_type,
                use_smp=use_smp,
                fp8=cfg.fp8,
                moe=cfg.moe,
            )
        if cfg.offload_activations:
            pytorch_model = offload_wrapper(pytorch_model)
            if cfg.model_type == "gpt_neox" and cfg.patch_neox_rope:
                # TODO: add a check to the model type and use enum for it
                patch_neox_rope(pytorch_model)
        model.model = pytorch_model
        return model

    def _setup_delayed_param(self, cfg, model):
        if not cfg.delayed_param:
            return None, None, nullcontext()
        if self.use_smp:
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

    def setup(self, trainer: "pl.Trainer") -> None:
        super(NLPFSDPStrategy, self).setup(trainer)
        logging.info(f"Training Model:\n{self.pytorch_model}")

    def optimizer_step(self, *a, **kw):
        grad_norm = clip_grad_norm_(self.pytorch_model, self.cfg.model.grad_clip)
        logging.debug(f"grad_norm: {grad_norm}")
        super().optimizer_step(*a, **kw)

    def setup_environment(self) -> None:
        """
        Setup distributed for SMP, and setup nemo distributing variables
        """
        # Init from original PT-Lightning policy to avoid megatron specific initialization
        super(NLPFSDPStrategy, self).setup_environment()

        tsm.init(self.smp_config_dict)

        # Setup nemo distributed variables, not actually initialize megatron distributed backend
        tensor_parallel_degree = self.smp_config_dict["tensor_parallel_degree"] if self.use_smp else 1
        context_parallel_degree = self.smp_config_dict["context_parallel_degree"] if self.use_smp else 1
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
        if typ == SageMakerCheckpointType.PEFT:
            return self.full_model_state_dict
        raise NotImplementedError(f"Checkpoint type '{typ}' not implemented")

    def sharded_optimizer_state_dict(self, optimizer: torch.optim.Optimizer):
        # TODO: Turn off optimizer offload_to_cpu? i.e., offload_to_cpu=False.
        #       We are unable to set offload_to_cpu=False now bc many features
        #       are still not applied.
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
        if typ == SageMakerCheckpointType.SHARDED:
            return self.sharded_optimizer_state_dict(optimizer)
        if typ == SageMakerCheckpointType.FULL:
            return self.full_optimizer_state_dict(optimizer)
        if typ == SageMakerCheckpointType.PEFT:
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
        if typ == SageMakerCheckpointType.PEFT:
            return self.load_full_model_state_dict(checkpoint)
        raise NotImplementedError(f"Checkpoint type '{typ}' not implemented")

    def load_sharded_optim_state_dict(self, trainer, checkpoint, path):
        for i, optimizer in enumerate(trainer.optimizers):
            optimizer_key = f"{OPTIMIZER_KEY_PREFIX}_{i}"
            state_dict = load_sharded_optimizer_state_dict(
                model_state_dict=checkpoint["state_dict"],
                optimizer_key=optimizer_key,
                storage_reader=DistributedFileSystemReader(path),
            )
            flattened_osd = FSDP.optim_state_dict_to_load(
                model=self.pytorch_model, optim=optimizer, optim_state_dict=state_dict[optimizer_key]
            )
            optimizer.load_state_dict(flattened_osd)

    def load_local_optim_state_dict(self, trainer, checkpoint, path):
        pass

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
        if typ == SageMakerCheckpointType.SHARDED:
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
        if self.checkpoint_io.checkpoint_type == SageMakerCheckpointType.FULL and dist.get_rank() != 0:
            return
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

    def save_peft_model(self, checkpoint_dir):
        """
        Save the FSDP wrapped PEFT model to the specified directory. Note that this method will
        only save the adapter weights.
        """
        logging.info(f"Saving PEFT checkpoint to {checkpoint_dir}")
        logging.debug(f"Model to save: {self.pytorch_model}")
        with FSDP.state_dict_type(
            self.pytorch_model,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(rank0_only=True, offload_to_cpu=True),
        ):
            """
            Need to extract the PEFT model from the FSDP wrapper to call save_pretrained()

            Example of what the _fsdp_wrapped_module looks like:
                FullyShardedDataParallel(
                    (_fsdp_wrapped_module): PeftModelForCausalLM(

            The model needs to be unwrapped in order to extract the PeftModelForCausalLM
            """
            self.pytorch_model.module.save_pretrained(checkpoint_dir)
            dist.barrier()
