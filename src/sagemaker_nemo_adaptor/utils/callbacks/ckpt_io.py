from typing import Any, Dict, Optional

import pytorch_lightning as pl
import torch.sagemaker.distributed.checkpoint.state_dict_saver as saver
from lightning_fabric.plugins import CheckpointIO, TorchCheckpointIO
from lightning_fabric.utilities.types import _PATH
from nemo.utils import logging

from sagemaker_nemo_adaptor.constants import SageMakerCheckpointType
from sagemaker_nemo_adaptor.utils.callbacks.local_ckpt_io import (
    SageMakerLocalCheckpointIO,
)
from sagemaker_nemo_adaptor.utils.callbacks.sharded_ckpt_io import (
    SageMakerShardedCheckpointIO,
)


class SageMakerCheckpointIO(CheckpointIO):
    def __init__(self, *a, **kw):
        self._checkpoint_type = SageMakerCheckpointType.SHARDED
        sharded_checkpoint_io = SageMakerShardedCheckpointIO(*a, **kw)
        local_checkpoint_io = SageMakerLocalCheckpointIO(*a, **kw)
        full_checkpoint_io = TorchCheckpointIO(*a, **kw)
        self._checkpoint_io = {
            SageMakerCheckpointType.SHARDED: sharded_checkpoint_io,
            SageMakerCheckpointType.LOCAL: local_checkpoint_io,
            SageMakerCheckpointType.FULL: full_checkpoint_io,
        }

    def save_checkpoint(
        self,
        checkpoint: Dict[str, Any],
        path: _PATH,
        storage_options: Optional[Any] = None,
    ) -> None:
        typ = self._checkpoint_type
        if typ not in self._checkpoint_io:
            raise NotImplementedError(f"Checkpoint type {typ} not implemented")
        logging.info(f"save_checkpoint: {path}")
        checkpoint_io = self._checkpoint_io[typ]
        return checkpoint_io.save_checkpoint(checkpoint, path, storage_options)

    def load_checkpoint(
        self,
        path: _PATH,
        trainer: "pl.Trainer",
        map_location: Optional[Any] = None,
    ) -> Dict[str, Any]:
        typ = self._checkpoint_type
        if typ not in self._checkpoint_io:
            raise NotImplementedError(f"Checkpoint type {typ} not implemented")
        checkpoint_io = self._checkpoint_io[typ]
        return checkpoint_io.load_checkpoint(path, trainer, map_location)

    def remove_checkpoint(self, path: _PATH) -> None:
        typ = self._checkpoint_type
        if typ not in self._checkpoint_io:
            raise NotImplementedError(f"Checkpoint type {typ} not implemented")
        return self._checkpoint_io[typ].remove_checkpoint(path)

    def wait(self):
        saver.maybe_finalize_async_calls(blocking=True)

    @property
    def checkpoint_type(self):
        return self._checkpoint_type

    @checkpoint_type.setter
    def checkpoint_type(self, typ: SageMakerCheckpointType):
        self._checkpoint_type = typ

    def get_checkpoint_io(self, typ):
        assert typ in self._checkpoint_io, f"{typ} is not supported"
        return self._checkpoint_io[typ]
