from omegaconf import DictConfig
from pytorch_lightning import Trainer
from transformers import default_data_collator

from sagemaker_nemo_adaptor.collections.data.base import BaseDataModule
from sagemaker_nemo_adaptor.collections.data.datasets import (
    HuggingFacePretrainingDataset,
)
from sagemaker_nemo_adaptor.utils.log_utils import Logger

_logger = Logger().get_logger()


class HuggingFaceDataModule(BaseDataModule):
    """
    Lightning DataModule for HuggingFace Pretraining dataset pipelining
    """

    def __init__(self, cfg: DictConfig, trainer: Trainer):
        super().__init__(cfg=cfg, trainer=trainer, collate_fn=default_data_collator)

    def train_dataloader(self):
        trainset = HuggingFacePretrainingDataset(input_path=self.cfg.model.data.train_dir)
        self._train_ds = trainset.dataset

        return self._build_dataloader(self._train_ds)

    def val_dataloader(self):
        if self.cfg.model.data.val_dir:
            valset = HuggingFacePretrainingDataset(input_path=self.cfg.model.data.val_dir, partition="val")
            self._validation_ds = valset.dataset

            return self._build_dataloader(self._validation_ds)
        else:
            return None

    def get_batch(self, data):
        return data["input_ids"], data["attention_mask"], data["labels"]

    def get_val_batch(self, data):
        return self.get_batch(data)
