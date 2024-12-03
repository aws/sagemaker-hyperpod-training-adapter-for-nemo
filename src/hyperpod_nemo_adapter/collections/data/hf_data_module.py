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

from omegaconf import DictConfig
from pytorch_lightning import Trainer
from transformers import default_data_collator

from hyperpod_nemo_adapter.collections.data.base import BaseDataModule
from hyperpod_nemo_adapter.collections.data.datasets import (
    HuggingFacePretrainingDataset,
)
from hyperpod_nemo_adapter.utils.log_utils import Logger

_logger = Logger().get_logger()


class HuggingFaceDataModule(BaseDataModule):
    """
    Lightning DataModule for HuggingFace Pretraining dataset pipelining
    """

    def __init__(self, cfg: DictConfig, trainer: Trainer):
        super().__init__(cfg=cfg, trainer=trainer, collate_fn=default_data_collator)

    def train_dataloader(self):
        input_path = self.cfg.model.data.train_dir
        trainset = HuggingFacePretrainingDataset(input_path=input_path, partition="train")
        self._train_ds = trainset.dataset
        return self._build_dataloader(self._train_ds, batch_size=self.cfg.model.train_batch_size)

    def val_dataloader(self):
        val_dir = self.cfg.model.data.val_dir
        if not val_dir:
            return None
        valset = HuggingFacePretrainingDataset(input_path=val_dir, partition="val")
        self._validation_ds = valset.dataset
        return self._build_dataloader(self._validation_ds, batch_size=self.cfg.model.val_batch_size)

    def get_batch(self, data):
        return data["input_ids"], data["attention_mask"], data["labels"]

    def get_val_batch(self, data):
        return self.get_batch(data)
