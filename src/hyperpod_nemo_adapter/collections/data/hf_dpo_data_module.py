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

from hyperpod_nemo_adapter.collections.data.base import BaseDataModule
from hyperpod_nemo_adapter.collections.data.datasets.hf_dpo_dataset import (
    HuggingFaceDPODataset,
)
from hyperpod_nemo_adapter.utils.dpo_utils import DataCollatorForPreference
from hyperpod_nemo_adapter.utils.log_utils import Logger

_logger = Logger().get_logger()


class HuggingFaceDPODataModule(BaseDataModule):
    """
    Lightning DataModule for HuggingFace DPO data pipelining.
    """

    def __init__(self, cfg: DictConfig, trainer: Trainer):
        super().__init__(cfg=cfg, trainer=trainer, collate_fn=DataCollatorForPreference(pad_token_id=0))

    def train_dataloader(self):
        input_path = self.cfg.model.data.train_dir
        trainset = HuggingFaceDPODataset(input_path=input_path, partition="train")
        self._train_ds = trainset.dataset
        return self._build_dataloader(self._train_ds, batch_size=self.cfg.model.train_batch_size)

    def val_dataloader(self):
        val_dir = self.cfg.model.data.val_dir
        if not val_dir:
            return None
        valset = HuggingFaceDPODataset(input_path=val_dir, partition="val")
        self._validation_ds = valset.dataset
        return self._build_dataloader(self._validation_ds, batch_size=self.cfg.model.val_batch_size)

    def get_batch(self, data):
        return (
            data["prompt_input_ids"],
            data["prompt_attention_mask"],
            data["chosen_input_ids"],
            data["chosen_attention_mask"],
            data["rejected_input_ids"],
            data["rejected_attention_mask"],
        )

    def get_val_batch(self, data):
        return self.get_batch(data)
