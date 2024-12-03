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
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

import pytorch_lightning as pl
import torch
import torch.distributed as dist
import torch.sagemaker.distributed.checkpoint.state_dict_saver as saver
from lightning_fabric.utilities.types import _PATH
from nemo.utils import logging
from s3torchconnectorclient._mountpoint_s3_client import S3Exception
from torch import multiprocessing as mp
from torch.sagemaker import state
from torch.sagemaker.distributed.checkpoint.async_utils import AsyncCallsQueue
from torch.sagemaker.distributed.checkpoint.filesystem import (
    DistributedFileSystemWriter,
    path_to_file_system,
)
from torch.sagemaker.distributed.checkpoint.s3_filesystem import (
    S3FileSystem,
    format_s3_path,
    is_s3_uri,
    retry_with_jitter,
)
from torch.sagemaker.distributed.checkpoint.state_dict_loader import load
from torch.sagemaker.distributed.checkpoint.state_dict_utils import init_optim_state

from hyperpod_nemo_adapter.utils.app_state import SageMakerAppState
from hyperpod_nemo_adapter.utils.callbacks.base_ckpt_io import SageMakerBaseCheckpointIO


def _subdir():
    app_state = SageMakerAppState()
    tp_rank = f"tp{state.tp_rank}"
    ep_rank = f"ep{state.ep_rank}"
    fsdp_rank = f"fsdp{dist.get_rank(app_state.fsdp_process_group)}"
    return "_".join([tp_rank, ep_rank, fsdp_rank])


def _subdirs():
    app_state = SageMakerAppState()
    for i in range(state.tp_size):
        for j in range(state.ep_size):
            for k in range(dist.get_world_size(app_state.fsdp_process_group)):
                yield f"tp{i}_ep{j}_fsdp{k}"


class _Profiler:
    def __init__(self):
        ctx = mp.get_context("fork")
        self.start = ctx.Manager().Value("d", -1.0)
        self.end = ctx.Manager().Value("d", -1.0)
        self._duration = -1.0

    @property
    def duration(self):
        return self._duration

    def update(self):
        start = self.start.value
        end = self.end.value
        duration = end - start
        self._duration = duration


class SageMakerLocalCheckpointIO(SageMakerBaseCheckpointIO):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.app_state = SageMakerAppState()
        self.queue = AsyncCallsQueue()
        self.profiler = _Profiler()
        self._ckpt_preprocessing_duration = 0

    @property
    def io_duration(self):
        return self.profiler.duration

    @property
    def ckpt_preprocessing_duration(self):
        return self._ckpt_preprocessing_duration

    @staticmethod
    @retry_with_jitter
    def write_local_metadata(training_step, writer):
        if writer.is_coordinator:
            path = writer.fs.concat_path(writer.path, ".local.metadata")
            with writer.fs.create_stream(path, "wb") as stream:
                metadata = {"step": training_step}
                torch.save(metadata, stream)

    @staticmethod
    def on_start(start, writer):
        start.value = time.perf_counter()

    @staticmethod
    def on_end(end, writer):
        end.value = time.perf_counter()

    def wait(self):
        self.queue.maybe_finalize_async_calls(blocking=True, skip_sync=True)
        self.profiler.update()

    def save_checkpoint(
        self,
        checkpoint: Dict[str, Any],
        path: _PATH,
        storage_options: Optional[Any] = None,
    ) -> None:
        trainer = storage_options
        assert isinstance(trainer, pl.Trainer)
        self.wait()

        # hooks
        on_start = functools.partial(
            SageMakerLocalCheckpointIO.on_start,
            self.profiler.start,
        )
        on_end = functools.partial(
            SageMakerLocalCheckpointIO.on_end,
            self.profiler.end,
        )
        hook = functools.partial(
            SageMakerLocalCheckpointIO.write_local_metadata,
            trainer.global_step,
        )

        path = os.path.join(path, _subdir())
        storage_writer = DistributedFileSystemWriter(
            path,
            pre_write_hooks=[on_start],
            post_write_hooks=[hook, on_end],
        )
        s = time.perf_counter()
        saver.async_save(
            checkpoint,
            storage_writer=storage_writer,
            process_group=self.app_state.current_replication_group,
            coordinator_rank=self.app_state.replication_coordinator_rank,
            queue=self.queue,
            force_check_all_plans=False,
            wait_error_handling=False,
        )
        self._ckpt_preprocessing_duration = time.perf_counter() - s

    def load_checkpoint(
        self,
        path: _PATH,
        trainer: "pl.Trainer" = None,
        map_location: Optional[Any] = None,
    ) -> Dict[str, Any]:
        for optimizer in trainer.optimizers:
            init_optim_state(optimizer, skip_empty_param=True)
        state_dict = trainer._checkpoint_connector.dump_checkpoint(weights_only=False)
        device = torch.cuda.current_device()
        for step, directory in SageMakerLocalCheckpointIO.list_checkpoint_dirs(path):
            success = self._load(trainer, state_dict, directory)
            state = step if success else -1
            tensor = torch.tensor([state], device=device)
            dist.all_reduce(tensor, op=dist.ReduceOp.MIN)
            state = tensor.item()

            if state >= 0:
                # The following line should be always true because all ranks
                # read all same folders. If the following line raises an
                # assertion error, the error may be from hardware configuration
                # such as some compute nodes' mount point is incorrect.
                assert state == step
                logging.info(f"Loaded Local checkpoint ({state}, {os.path.dirname(directory)})")
                return state_dict

        raise FileNotFoundError("No checkpoint be found")

    def _load(self, trainer, state_dict, path):
        try:
            load(
                state_dict=state_dict,
                checkpoint_id=path,
                process_group=self.app_state.current_replication_group,
                coordinator_rank=self.app_state.replication_coordinator_rank,
            )
            self.load_data_module_and_lr_schedulers(trainer, state_dict)
            return True
        except Exception as e:
            logging.warning(f"Load checkpoint fail. error: {e}")
            return False

    @retry_with_jitter
    @staticmethod
    def list_checkpoint_dirs(path):
        path = Path(path)
        path = format_s3_path(path) if is_s3_uri(path) else path
        fs = path_to_file_system(path)
        dirs = SageMakerLocalCheckpointIO.listdir(fs, path)
        candidate = []
        for f in dirs:
            step = SageMakerLocalCheckpointIO.load_step(fs, f)
            if step >= 0:
                candidate.append((step, f))

        candidate.sort(reverse=True)
        for s, f in candidate:
            yield s, fs.concat_path(f, _subdir())

    @staticmethod
    def listdir(fs, path):
        if isinstance(fs, S3FileSystem):
            return [f"s3://{f}" for f in fs.fs.listdir(path, detail=False)]
        return [fs.concat_path(path, f) for f in os.scandir(path)]

    @staticmethod
    def load_step(fs, root):
        prev = None
        for d in _subdirs():
            file = fs.concat_path(root, d)
            path = fs.concat_path(file, ".local.metadata")
            meta = SageMakerLocalCheckpointIO.load_local_metadata(fs, path)
            step = meta.get("step", -1)
            if step < 0:
                return -1
            if not prev:
                prev = step
                continue
            if prev != step:
                return -1
        return step

    @staticmethod
    @retry_with_jitter
    def load_local_metadata(fs, file):
        try:
            logging.debug(f"Loading local metadata: {file}")
            with fs.create_stream(file, "rb") as f:
                return torch.load(f)
        except S3Exception as e:
            msg = str(e)
            if "Service error: The key does not exist" in msg:
                return {}
            if "Service error: The bucket does not exist" in msg:
                return {}
            raise
        except Exception as e:
            logging.warning(f"torch.load({file}) fail. error: {e}")
            return {}

    def remove_checkpoint(self, path: _PATH) -> None:
        raise NotImplementedError("SageMakerLocalCheckpointIO.remove_checkpoint not implemented")

    def teardown(self):
        self.queue.maybe_finalize_async_calls(blocking=True)
