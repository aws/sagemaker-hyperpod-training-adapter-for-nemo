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
        input_path = self.cfg.model.data.train_dir
        trainset = HuggingFacePretrainingDataset(input_path=input_path, partition="train")
        self._train_ds = trainset.dataset
        return self._build_dataloader(self._train_ds)

    def val_dataloader(self):
        val_dir = self.cfg.model.data.val_dir
        if not val_dir:
            return None
        valset = HuggingFacePretrainingDataset(input_path=val_dir, partition="val")
        self._validation_ds = valset.dataset
        return self._build_dataloader(self._validation_ds)

    def get_batch(self, data):
        return data["input_ids"], data["attention_mask"], data["labels"]

    def get_val_batch(self, data):
        return self.get_batch(data)
