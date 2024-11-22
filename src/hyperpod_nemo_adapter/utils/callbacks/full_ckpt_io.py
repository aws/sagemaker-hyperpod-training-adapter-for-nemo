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
