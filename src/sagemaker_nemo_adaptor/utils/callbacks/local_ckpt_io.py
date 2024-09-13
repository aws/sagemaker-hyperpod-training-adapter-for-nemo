from typing import Any, Dict, Optional

import pytorch_lightning as pl
import torch.sagemaker.distributed.checkpoint.state_dict_saver as saver
from lightning_fabric.utilities.types import _PATH
from nemo.utils import logging
from torch.sagemaker.distributed.checkpoint.async_utils import AsyncCallsQueue
from torch.sagemaker.distributed.checkpoint.filesystem import (
    DistributedFileSystemReader,
)
from torch.sagemaker.distributed.checkpoint.state_dict_loader import load
from torch.sagemaker.distributed.checkpoint.state_dict_utils import init_optim_state

from sagemaker_nemo_adaptor.utils.app_state import SageMakerAppState
from sagemaker_nemo_adaptor.utils.callbacks.base_ckpt_io import (
    SageMakerBaseCheckpointIO,
)


class SageMakerLocalCheckpointIO(SageMakerBaseCheckpointIO):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.app_state = SageMakerAppState()
        self.queue = AsyncCallsQueue()

    def save_checkpoint(
        self,
        checkpoint: Dict[str, Any],
        path: _PATH,
        storage_options: Optional[Any] = None,
    ) -> None:
        self.queue.maybe_finalize_async_calls(blocking=True)
        saver.async_save(
            checkpoint,
            checkpoint_id=path,
            process_group=self.app_state.current_replication_group,
            coordinator_rank=self.app_state.replication_coordinator_rank,
            queue=self.queue,
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
        self.load_data_module_and_lr_schedulers(trainer, state_dict)
        logging.info(f"Loaded Local checkpoint")
        return state_dict

    def remove_checkpoint(self, path: _PATH) -> None:
        raise NotImplementedError("SageMakerLocalCheckpointIO.remove_checkpoint not implemented")

    def teardown(self):
        self.queue.maybe_finalize_async_calls(blocking=True)
