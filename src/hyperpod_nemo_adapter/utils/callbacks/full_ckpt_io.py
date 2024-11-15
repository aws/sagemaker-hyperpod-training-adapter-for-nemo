import os
from typing import Any, Dict, Optional

import torch.distributed as dist
from lightning_fabric.plugins import TorchCheckpointIO
from lightning_fabric.utilities.types import _PATH


class SageMakerFullCheckpointIO(TorchCheckpointIO):
    def save_checkpoint(self, checkpoint: Dict[str, Any], path: _PATH, storage_options: Optional[Any] = None) -> None:
        if dist.get_rank() == 0:
            trainer = storage_options
            # Save full model in huggingface format. pytorch_model.bin is used during from_pretrined.
            super().save_checkpoint(checkpoint["state_dict"], os.path.join(path, "pytorch_model.bin"))
            if trainer:
                trainer.strategy.model.model_config.save_pretrained(path)
