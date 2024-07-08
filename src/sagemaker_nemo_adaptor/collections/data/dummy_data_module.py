from sagemaker_nemo_adaptor.collections.data.base import BaseDataModule
from sagemaker_nemo_adaptor.collections.data.datasets import DummyDataset


class DummyDataModule(BaseDataModule):
    """
    Lightning DataModule for synthetic data pipelining
    """

    def train_dataloader(self):
        self._train_ds = DummyDataset(vocab_size=self.cfg.vocab_size, seqlen=self.cfg.max_context_width)
        return self._build_dataloader(self._train_ds)

    def val_dataloader(self):
        """We're not doing validation for synthetic data"""
        return None

    def get_batch(self, data):
        return data[0], data[1], data[0]

    def get_val_batch(self, data):
        return self.get_batch(data)
