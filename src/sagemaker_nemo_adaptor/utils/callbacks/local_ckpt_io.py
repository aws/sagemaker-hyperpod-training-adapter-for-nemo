from typing import Any, Dict, Optional

import pytorch_lightning as pl
from lightning_fabric.plugins import CheckpointIO
from lightning_fabric.utilities.types import _PATH


class SageMakerLocalCheckpointIO(CheckpointIO):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)

    def save_checkpoint(
        self,
        checkpoint: Dict[str, Any],
        path: _PATH,
        storage_options: Optional[Any] = None,
    ) -> None:
        raise NotImplementedError("SageMakerLocalCheckpointIO.save_checkpoint not implemented")

    def load_checkpoint(
        self,
        path: _PATH,
        trainer: "pl.Trainer" = None,
        map_location: Optional[Any] = None,
    ) -> Dict[str, Any]:
        raise NotImplementedError("SageMakerLocalCheckpointIO.load_checkpoint not implemented")

    def remove_checkpoint(self, path: _PATH) -> None:
        raise NotImplementedError("SageMakerLocalCheckpointIO.remove_checkpoint not implemented")
