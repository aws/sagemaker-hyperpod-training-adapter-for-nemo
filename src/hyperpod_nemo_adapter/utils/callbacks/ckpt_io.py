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
from lightning_fabric.plugins import CheckpointIO
from lightning_fabric.utilities.types import _PATH
from nemo.utils import logging

from hyperpod_nemo_adapter.constants import (
    OPTIMIZER_KEY_PREFIX,
    SageMakerCheckpointType,
)
from hyperpod_nemo_adapter.utils.callbacks.full_ckpt_io import SageMakerFullCheckpointIO
from hyperpod_nemo_adapter.utils.callbacks.local_ckpt_io import (
    SageMakerLocalCheckpointIO,
)
from hyperpod_nemo_adapter.utils.callbacks.peft_ckpt_io import (
    SageMakerPeftFullCheckpointIO,
    SageMakerPeftShardedCheckpointIO,
)
from hyperpod_nemo_adapter.utils.callbacks.sharded_ckpt_io import (
    SageMakerShardedCheckpointIO,
)


class SageMakerCheckpointIO(CheckpointIO):
    def __init__(self, *a, **kw):
        self._checkpoint_type = SageMakerCheckpointType.SHARDED
        sharded_checkpoint_io = SageMakerShardedCheckpointIO(*a, **kw)
        local_checkpoint_io = SageMakerLocalCheckpointIO(*a, **kw)
        full_checkpoint_io = SageMakerFullCheckpointIO(*a, **kw)
        peft_full_checkpoint_io = SageMakerPeftFullCheckpointIO(*a, **kw)
        peft_sharded_checkpoint_io = SageMakerPeftShardedCheckpointIO(*a, **kw)
        self._checkpoint_io = {
            SageMakerCheckpointType.SHARDED: sharded_checkpoint_io,
            SageMakerCheckpointType.LOCAL: local_checkpoint_io,
            SageMakerCheckpointType.FULL: full_checkpoint_io,
            SageMakerCheckpointType.PEFT_FULL: peft_full_checkpoint_io,
            SageMakerCheckpointType.PEFT_SHARDED: peft_sharded_checkpoint_io,
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
        logging.info(f"save {self._checkpoint_type} checkpoint: {path}")
        # Only sharded checkpointing needs unique optimizer key
        if "optimizer_states" in checkpoint and (
            self._checkpoint_type == SageMakerCheckpointType.SHARDED
            or self._checkpoint_type == SageMakerCheckpointType.PEFT_SHARDED
        ):
            optimizers = checkpoint.pop("optimizer_states")
            for i, optim in enumerate(optimizers):
                checkpoint[f"{OPTIMIZER_KEY_PREFIX}_{i}"] = optim
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

    def __getitem__(self, typ: SageMakerCheckpointType) -> CheckpointIO:
        return self._checkpoint_io[typ]

    @property
    def checkpoint_type(self):
        return self._checkpoint_type

    @checkpoint_type.setter
    def checkpoint_type(self, typ: SageMakerCheckpointType):
        self._checkpoint_type = typ

    def teardown(self, trainer):
        checkpoint_io = self._checkpoint_io[SageMakerCheckpointType.SHARDED]
        checkpoint_io.teardown()
        checkpoint_io = self._checkpoint_io[SageMakerCheckpointType.LOCAL]
        checkpoint_io.teardown()
        checkpoint_io = self._checkpoint_io[SageMakerCheckpointType.PEFT_SHARDED]
        checkpoint_io.teardown()
