from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Union

import torch
import torch.distributed as dist
import torch.sagemaker as tsm
from lightning_fabric.utilities.types import _PATH
from nemo.collections.nlp.parts.nlp_overrides import NLPFSDPStrategy
from nemo.utils import logging
from omegaconf.dictconfig import DictConfig
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType
from torch.distributed.fsdp.api import FullOptimStateDictConfig, FullStateDictConfig
from torch.sagemaker.distributed.checkpoint.state_dict_utils import (
    SMStateDictType,
    sm_state_dict_type,
)

from sagemaker_nemo_adaptor.constants import SageMakerCheckpointType
from sagemaker_nemo_adaptor.utils.callbacks.checkpoint import SageMakerCheckpointIO
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
        **kwargs: Union[Any, Dict[str, Any]],
    ) -> None:
        self.cfg = cfg
        self.use_smp = cfg.use_smp
        self.smp_config_dict = self._setup_smp_config(cfg)

        # Init from original PT-Lightning policy to avoid megatron specific initialization
        super(NLPFSDPStrategy, self).__init__(**kwargs)

    def _setup_smp_config(self, cfg):
        smp_config = {
            "activation_loading_horizon": cfg.model.activation_loading_horizon,
            "sm_activation_offloading": cfg.model.offload_activations > 0,
            "tensor_parallel_degree": cfg.model.tensor_model_parallel_degree,
            "expert_parallel_degree": cfg.model.expert_model_parallel_degree,
            "random_seed": cfg.model.seed,
        }
        if cfg.model.shard_degree:
            smp_config["hybrid_shard_degree"] = cfg.model.shard_degree
        return smp_config

    def setup(self, trainer: "pl.Trainer") -> None:
        super(NLPFSDPStrategy, self).setup(trainer)
        logging.info(f"Training Model: {self.model}")

    def setup_environment(self) -> None:
        """
        Setup distributed for SMP, and setup nemo distributing variables
        """
        # Init from original PT-Lightning policy to avoid megatron specific initialization
        super(NLPFSDPStrategy, self).setup_environment()

        # Initialize smp, todo: check whether we still need this for HF case
        if self.use_smp:
            tsm.init(self.smp_config_dict)

        # Setup nemo distributed variables, not actually initialize megatron distributed backend
        tensor_parallel_degree = self.smp_config_dict["tensor_parallel_degree"] if self.use_smp else 1
        initialize_model_parallel_for_nemo(
            world_size=self.world_size,
            global_rank=self.global_rank,
            local_rank=self.local_rank,
            tensor_model_parallel_size=tensor_parallel_degree,
            seed=self.cfg.model.seed,
        )

    @property
    def sharded_model_state_dict(self):
        with FSDP.state_dict_type(self.model, StateDictType.SHARDED_STATE_DICT):
            return self.model.state_dict()

    @property
    def local_model_state_dict(self):
        with sm_state_dict_type(self.model, SMStateDictType.SM_LOCAL_STATE_DICT):
            return self.model.state_dict()

    @property
    def full_model_state_dict(self):
        state_dict_config = None
        state_dict_config = FullStateDictConfig(rank0_only=True, offload_to_cpu=True)
        with sm_state_dict_type(self.model, StateDictType.FULL_STATE_DICT, state_dict_config=state_dict_config):
            return self.model.state_dict()

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
        raise NotImplementedError(f"Checkpoint type '{typ}' not implemented")

    def sharded_optimizer_state_dict(self, optimizer: torch.optim.Optimizer):
        # TODO: Turn off optimizer offload_to_cpu? i.e., offload_to_cpu=False.
        #       We are unable to set offload_to_cpu=False now bc many features
        #       are still not applied.
        with FSDP.state_dict_type(self.model, StateDictType.SHARDED_STATE_DICT):
            return FSDP.optim_state_dict(self.model, optimizer)

    def local_optimizer_state_dict(self, optimizer: torch.optim.Optimizer):
        with sm_state_dict_type(self.model, SMStateDictType.SM_LOCAL_STATE_DICT):
            return optimizer.state_dict()

    def full_optimizer_state_dict(self, optimizer: torch.optim.Optimizer):
        optim_state_dict_config = FullOptimStateDictConfig(rank0_only=True, offload_to_cpu=True)
        with sm_state_dict_type(
            self.model, StateDictType.FULL_STATE_DICT, optim_state_dict_config=optim_state_dict_config
        ):
            return FSDP.optim_state_dict(self.model, optimizer)

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
        raise NotImplementedError(f"Checkpoint type '{typ}' not implemented")

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
