import shutil
from typing import Any, Dict, Optional

import torch.sagemaker.distributed.checkpoint.state_dict_saver as saver
from lightning_fabric.plugins import CheckpointIO
from lightning_fabric.utilities.types import _PATH
from torch.sagemaker.distributed.checkpoint.filesystem import (
    DistributedFileSystemReader,
)


class SageMakerShardedCheckpointIO(CheckpointIO):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)

    def save_checkpoint(
        self,
        checkpoint: Dict[str, Any],
        path: _PATH,
        storage_options: Optional[Any] = None,
    ) -> None:
        saver.async_save(checkpoint, checkpoint_id=path)

    def load_checkpoint(
        self,
        path: _PATH,
        map_location: Optional[Any] = None,
    ) -> Dict[str, Any]:
        ...

    def remove_checkpoint(self, path: _PATH) -> None:
        shutil.rmtree(path, ignore_errors=True)
