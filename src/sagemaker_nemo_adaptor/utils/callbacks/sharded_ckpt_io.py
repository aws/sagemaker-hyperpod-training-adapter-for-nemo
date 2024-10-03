from typing import Any, Dict, Optional

import pytorch_lightning as pl
import torch.distributed as dist
import torch.sagemaker.distributed.checkpoint.state_dict_loader as loader
import torch.sagemaker.distributed.checkpoint.state_dict_saver as saver
from lightning.fabric.utilities.cloud_io import get_filesystem
from lightning_fabric.utilities.types import _PATH
from nemo.utils import logging
from torch.sagemaker.distributed.checkpoint.filesystem import (
    DistributedFileSystemWriter,
)

from sagemaker_nemo_adaptor.utils.callbacks.base_ckpt_io import (
    SageMakerBaseCheckpointIO,
)


class SageMakerShardedCheckpointIO(SageMakerBaseCheckpointIO):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)

    def save_checkpoint(
        self,
        checkpoint: Dict[str, Any],
        path: _PATH,
        storage_options: Optional[Any] = None,
    ) -> None:
        trainer = storage_options
        assert isinstance(trainer, pl.Trainer)
        group = self.app_state.fsdp_process_group
        saver.maybe_finalize_async_calls(blocking=True, process_group=group)
        if self.app_state.is_fsdp_action_rank:
            storage_writer = DistributedFileSystemWriter(path)
            saver.async_save(
                checkpoint,
                storage_writer=storage_writer,
                process_group=self.app_state.fsdp_process_group,
                coordinator_rank=self.app_state.fsdp_coordinator_rank,
                force_check_all_plans=False,
                wait_error_handling=False,
            )

    def load_checkpoint(
        self,
        path: _PATH,
        trainer: "pl.Trainer",
        map_location: Optional[Any] = None,
    ) -> Dict[str, Any]:
        assert trainer, "Bad parameter, trainer is empty"
        state_dict = trainer._checkpoint_connector.dump_checkpoint(False)
        state_dict.pop("optimizer_states")
        loader.load(
            state_dict,
            checkpoint_id=path,
            process_group=self.app_state.fsdp_process_group,
            coordinator_rank=self.app_state.fsdp_coordinator_rank,
        )
        self.load_data_module_and_lr_schedulers(trainer, state_dict)
        logging.info(f"Loaded Sharded checkpoint")
        return state_dict

    def remove_checkpoint(self, path: _PATH) -> None:
        if dist.get_rank() != 0:
            return
        fs = get_filesystem(path)
        if fs.exists(path):
            fs.rm(path, recursive=True)
            logging.info(f"Removed checkpoint: {path}")

    def teardown(self):
        saver.maybe_finalize_async_calls(blocking=True, process_group=self.app_state.fsdp_process_group)
