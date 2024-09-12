from typing import Any, Dict, Optional

import pytorch_lightning as pl
import torch.distributed as dist
import torch.sagemaker.distributed.checkpoint.state_dict_saver as saver
from lightning_fabric.plugins import CheckpointIO
from lightning_fabric.utilities.types import _PATH
from nemo.utils import logging
from torch.sagemaker.distributed.checkpoint.filesystem import (
    DistributedFileSystemReader,
)
from torch.sagemaker.distributed.checkpoint.state_dict_loader import load
from torch.sagemaker.distributed.checkpoint.state_dict_utils import init_optim_state

from sagemaker_nemo_adaptor.utils.app_state import SageMakerAppState


class SageMakerLocalCheckpointIO(CheckpointIO):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.app_state = SageMakerAppState()

    def save_checkpoint(
        self,
        checkpoint: Dict[str, Any],
        path: _PATH,
        storage_options: Optional[Any] = None,
    ) -> None:
        saver.async_save(
            checkpoint,
            checkpoint_id=path,
            process_group=self.app_state.current_replication_group,
            coordinator_rank=self.app_state.replication_coordinator_rank,
        )

    def load_checkpoint(
        self,
        path: _PATH,
        trainer: "pl.Trainer" = None,
        map_location: Optional[Any] = None,
    ) -> Dict[str, Any]:
        for optimizer in trainer.optimizers:
            init_optim_state(optimizer, skip_empty_param=True)
        state_dict = trainer._checkpoint_connector.dump_checkpoint(weights_only=False)
        load(
            state_dict=state_dict,
            process_group=self.app_state.current_replication_group,
            coordinator_rank=self.app_state.replication_coordinator_rank,
            storage_reader=DistributedFileSystemReader(path),
        )
        trainer.datamodule.load_state_dict(state_dict)
        if dist.get_rank() == 0:
            logging.info(f"Loaded Local checkpoint")

    def remove_checkpoint(self, path: _PATH) -> None:
        raise NotImplementedError("SageMakerLocalCheckpointIO.remove_checkpoint not implemented")
