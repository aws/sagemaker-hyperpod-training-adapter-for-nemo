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

from typing import Any, Dict, Optional

import pytorch_lightning as pl
import torch.distributed as dist
import torch.sagemaker.distributed.checkpoint.state_dict_loader as loader
import torch.sagemaker.distributed.checkpoint.state_dict_saver as saver
from lightning.fabric.utilities.cloud_io import get_filesystem
from lightning_fabric.utilities.types import _PATH
from nemo.utils import logging
from torch.sagemaker.distributed.checkpoint.async_utils import AsyncCallsQueue
from torch.sagemaker.distributed.checkpoint.filesystem import (
    DistributedFileSystemWriter,
)

from hyperpod_nemo_adapter.utils.callbacks.base_ckpt_io import SageMakerBaseCheckpointIO


class SageMakerShardedCheckpointIO(SageMakerBaseCheckpointIO):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.queue = AsyncCallsQueue()

    def save_checkpoint(
        self,
        checkpoint: Dict[str, Any],
        path: _PATH,
        storage_options: Optional[Any] = None,
    ) -> None:
        trainer = storage_options
        assert isinstance(trainer, pl.Trainer)
        group = self.app_state.fsdp_process_group
        self.queue.maybe_finalize_async_calls(blocking=True, process_group=group)

        if self.app_state.is_fsdp_action_rank:
            storage_writer = DistributedFileSystemWriter(path)
            saver.async_save(
                checkpoint,
                storage_writer=storage_writer,
                process_group=self.app_state.fsdp_process_group,
                coordinator_rank=self.app_state.fsdp_coordinator_rank,
                queue=self.queue,
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
        self.queue.maybe_finalize_async_calls(blocking=True, process_group=self.app_state.fsdp_process_group)
