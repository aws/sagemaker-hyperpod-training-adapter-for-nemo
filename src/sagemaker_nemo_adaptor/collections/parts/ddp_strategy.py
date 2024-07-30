from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Union

import torch
import torch.sagemaker as tsm
from nemo.collections.nlp.parts.nlp_overrides import NLPDDPStrategy
from omegaconf.dictconfig import DictConfig

from sagemaker_nemo_adaptor.utils.dist_utils import initialize_model_parallel_for_nemo
from sagemaker_nemo_adaptor.utils.fsdp_utils import (
    get_auto_wrap_policy,
    get_backward_fetch_policy,
    get_sharding_strategy,
    get_transformer_layer,
)


class SageMakerDDPStrategy(NLPDDPStrategy):
    """
    FSDP plugin for Pytorch Lightning with the support forsharding_strategy tensor-parallelism.
    SageMakerDDPStrategy deals with
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
        self.use_smp = use_smp
        self.smp_config_dict = smp_config_dict

        # # Set the mixed precision recipe TODO: Uncomment this once we moved FSDP setup back to strategy
        # kwargs["mixed_precision"] = self._set_mixed_precision_recipe(
        #     precision, grad_reduce_dtype, set_buffer_dtype=set_buffer_dtype
        # )

        # Init from original PT-Lightning policy to avoid megatron specific initialization
        super(NLPDDPStrategy, self).__init__(**kwargs)

    def _setup_model(self, model: torch.nn.Module) -> torch.nn.Module:
        """Wraps the model into a :class:`~torch.distributed.fsdp.fully_sharded_data_parallel.FullyShardedDataParallel`
        module.
        Over write original PT-Lightning _setup_model function to add deferred_initialization related patches.
        """

        # TODO: Check we need DDP wrapper here

        return model

    def setup(self, trainer: "pl.Trainer") -> None:
        super(NLPDDPStrategy, self).setup(trainer)

        if torch.distributed.get_rank() == 0:
            print(f"strategy model is {self.model}")
            print(f"strategy lt model is {self.lightning_module}")

    def setup_environment(self) -> None:
        """
        Setup distributed for SMP, and setup nemo distributing variables
        """
        # Init from original PT-Lightning policy to avoid megatron specific initialization
        super(NLPDDPStrategy, self).setup_environment()

        # Initialize smp, todo: check whether we still need this for HF case
        tsm.init(self.smp_config_dict)

        # Setup nemo distributed variables, not actually initialize megatron distributed backend
        initialize_model_parallel_for_nemo(
            world_size=self.world_size,
            global_rank=self.global_rank,
            local_rank=self.local_rank,
            tensor_model_parallel_size=self.smp_config_dict["tensor_parallel_degree"],
            seed=self.seed,
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
