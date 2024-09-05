from typing import Any, Dict, Optional

import torch.distributed as dist
from lightning_fabric.plugins import TorchCheckpointIO
from lightning_fabric.utilities.types import _PATH


class SageMakerFullCheckpointIO(TorchCheckpointIO):
    def save_checkpoint(self, checkpoint: Dict[str, Any], path: _PATH, storage_options: Optional[Any] = None) -> None:
        if dist.get_rank() == 0:
            return super().save_checkpoint(checkpoint, path)
