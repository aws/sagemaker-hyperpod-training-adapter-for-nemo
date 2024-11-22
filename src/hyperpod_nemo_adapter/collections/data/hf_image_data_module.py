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
from transformers import AutoProcessor, default_data_collator

from hyperpod_nemo_adapter.collections.data.base import BaseDataModule
from hyperpod_nemo_adapter.collections.data.datasets import (
    HuggingFacePretrainingVisionDataset,
)
from hyperpod_nemo_adapter.collections.data.vision_dataset import (
    OCRVQADataCollator,
    get_custom_dataset,
)
from hyperpod_nemo_adapter.utils.log_utils import Logger

_logger = Logger().get_logger()


class HuggingFaceVisionDataModule(BaseDataModule):
    """
    Lightning DataModule for HuggingFace Pretraining dataset pipelining
    """

    def __init__(self, cfg: DictConfig, trainer: Trainer):
        super().__init__(cfg=cfg, trainer=trainer, collate_fn=default_data_collator)

    def get_dataloader(self, training=True):
        assert (
            self.cfg.model.hf_model_name_or_path is not None
        ), "Currently HuggingFaceVisionDataModule only support to run with hf_model_name_or_path"
        if self.cfg.model.data.train_dir is None:
            _logger.info(f"train_dir is not provided, using ocrvqa data for testing")
            if training:
                self._train_ds = get_custom_dataset("train")
            else:
                self._validation_ds = get_custom_dataset("test")
        else:
            input_path = self.cfg.model.data.train_dir if training else self.cfg.model.data.val_dir
            if not input_path:
                return None
            dataset = HuggingFacePretrainingVisionDataset(
                input_path=input_path, partition="train" if training else "val"
            )
            if training:
                self._train_ds = dataset.dataset
            else:
                self._validation_ds = dataset.dataset

        token = self.cfg.model.get("hf_access_token", None)
        processor = AutoProcessor.from_pretrained(self.cfg.model.hf_model_name_or_path, token=token)
        processor.tokenizer.padding_side = "right"
        data_collator = OCRVQADataCollator(processor)
        self.collate_fn = data_collator
        if training:
            return self._build_dataloader(self._train_ds, batch_size=self.cfg.model.train_batch_size)
        else:
            return self._build_dataloader(self._validation_ds, batch_size=self.cfg.model.val_batch_size)

    def train_dataloader(self):
        return self.get_dataloader()

    def val_dataloader(self):
        return self.get_dataloader(training=False)

    def get_batch(self, data):
        return data

    def get_val_batch(self, data):
        return self.get_batch(data)
