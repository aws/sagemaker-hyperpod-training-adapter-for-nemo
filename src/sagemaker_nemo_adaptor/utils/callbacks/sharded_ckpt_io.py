import shutil
from typing import Any, Dict, Optional

import pytorch_lightning as pl
import torch.sagemaker.distributed.checkpoint.state_dict_loader as loader
import torch.sagemaker.distributed.checkpoint.state_dict_saver as saver
from lightning_fabric.plugins import CheckpointIO
from lightning_fabric.utilities.types import _PATH


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
        trainer: "pl.Trainer",
        map_location: Optional[Any] = None,
    ) -> Dict[str, Any]:
        assert trainer, "Bad parameter, trainer is empty"
        state_dict = trainer._checkpoint_connector.dump_checkpoint(False)
        return loader.load(state_dict, checkpoint_id=path)

    def remove_checkpoint(self, path: _PATH) -> None:
        shutil.rmtree(path, ignore_errors=True)
