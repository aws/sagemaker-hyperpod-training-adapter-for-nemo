from sagemaker_nemo_adaptor.collections.data.base import BaseDataModule
from sagemaker_nemo_adaptor.collections.data.datasets import GPTPretrainingDataset


class GPTDataModule(BaseDataModule):
    """
    Lightning DataModule for GPT Pretraining dataset pipelining
    """

    def train_dataloader(self):
        self._train_ds = GPTPretrainingDataset(
            input_paths=self.cfg.data.train_dir,
            max_context_width=self.cfg.max_context_width,
            zipped=self.cfg.data.zipped_data,
        )
        return self._build_dataloader(self._train_ds)

    def val_dataloader(self):
        if self.cfg.data.val_dir:
            self._validation_ds = GPTPretrainingDataset(
                input_paths=self.cfg.data.val_dir,
                max_context_width=self.cfg.max_context_width,
                zipped=self.cfg.zipped.data,
            )
            return self._build_dataloader(self._validation_ds)
        else:
            return None

    def get_batch(self, data):
        return data[0], data[1], data[0]

    def get_val_batch(self, data):
        return self.get_batch(data)
