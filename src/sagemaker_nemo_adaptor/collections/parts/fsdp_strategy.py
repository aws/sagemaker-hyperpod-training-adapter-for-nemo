from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Union

import torch
import torch.sagemaker as tsm
from nemo.collections.nlp.parts import utils_funcs
from nemo.collections.nlp.parts.nlp_overrides import NLPFSDPStrategy
from omegaconf.dictconfig import DictConfig
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, StateDictType
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
        use_smp: bool = True,
        smp_config_dict: Dict = None,
        # FSDP args
        **kwargs: Union[Any, Dict[str, Any]],
    ) -> None:
        self.cfg = cfg
        self.use_smp = use_smp
        self.smp_config_dict = smp_config_dict

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
            tsm.init(self.smp_config_dict)

        # Setup nemo distributed variables, not actually initialize megatron distributed backend
        initialize_model_parallel_for_nemo(
            world_size=self.world_size,
            global_rank=self.global_rank,
            local_rank=self.local_rank,
            tensor_model_parallel_size=self.smp_config_dict["tensor_parallel_degree"] if self.use_smp else 1,
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

    def lightning_module_state_dict(self) -> Dict[str, Any]:
        """
        Store the model state dict in one of full or sharded format.
        """
        assert isinstance(self.checkpoint_io, SageMakerCheckpointIO)
        typ = self.checkpoint_io.checkpoint_type
        if typ == SageMakerCheckpointType.LOCAL:
            return self.local_model_state_dict
        if typ == SageMakerCheckpointType.SHARDED:
            return self.sharded_model_state_dict
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

    def optimizer_state(self, optimizer: torch.optim.Optimizer) -> Dict[str, torch.Tensor]:
        """
        Store the full optimizer state dict in one of full or sharded format.
        """
        assert isinstance(self.checkpoint_io, SageMakerCheckpointIO)
        typ = self.checkpoint_io.checkpoint_type
        if typ == SageMakerCheckpointType.LOCAL:
            return self.local_optimizer_state_dict(optimizer)
        if typ == SageMakerCheckpointType.SHARDED:
            return self.sharded_optimizer_state_dict(optimizer)
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
        self.checkpoint_io.save_checkpoint(checkpoint, filepath, storage_options)

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
