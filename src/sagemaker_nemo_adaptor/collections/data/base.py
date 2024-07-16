import torch
import torch.distributed as dist
from nemo.utils import AppState
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.core.datamodule import LightningDataModule
from torch.utils.data import DataLoader


class BaseDataModule(LightningDataModule):
    """
    General Lightning DataModule class for SageMaker adaptor, it deals with
    1. Provide general function of build dataloader with sampler
    2. Setup data parallel parameters
    3. (TODO: WIP) Compute the processed batches for checkpointing and throughput calculation.
    """

    def __init__(self, cfg: DictConfig, trainer: Trainer):
        super().__init__()
        self.cfg = cfg
        self.trainer = trainer

    def setup(self, stage=None):
        super().setup(stage)

        app_state = AppState()
        self.dp_size = app_state.data_parallel_size
        self.dp_rank = app_state.data_parallel_rank

        # TODO: implement checkpoint save/load
        # resume_checkpoint_path = self.trainer._checkpoint_connector.resume_from_checkpoint_fit_path
        # self.init_consumed_samples = (
        #     self._extract_consumed_samples_from_ckpt(resume_checkpoint_path) if resume_checkpoint_path else 0
        # )

    def _build_dataloader(
        self,
        dataset,
        resume_from_sequence_number=0,
        num_workers=0,
        shuffle=False,
        collate_fn=None,
    ):
        """
        Build sampler and dataloader
        TODO: resume_from_sequence_number is related to checkpoint save/load, revisit when implement
        TODO: These arguments should be configurable through recipe, could be configured in datamodule/recipe checker
        """
        # TODO: set sampler.epoch to correctly shuffle across epochs, else same order will be used for
        # all epochs not relevant now as we have no epochs
        sampler = torch.utils.data.DistributedSampler(
            dataset,
            shuffle=shuffle,
            seed=self.cfg.seed,
            rank=self.dp_rank,
            num_replicas=self.dp_size,
            drop_last=True,
        )

        kwargs = {
            "sampler": sampler,
            "batch_size": self.cfg.train_batch_size,
            "num_workers": num_workers,
            "collate_fn": collate_fn,
            "pin_memory": True,
            "drop_last": True,
        }

        if resume_from_sequence_number > 0:
            dataloader = SkipDataLoader(dataset, resume_from_sequence_number=resume_from_sequence_number, **kwargs)
        else:
            dataloader = torch.utils.data.DataLoader(dataset, **kwargs)
        return dataloader

    def compute_consumed_samples(self, steps_since_resume=0):
        # TODO: implement checkpoint save/load
        pass

    def _extract_consumed_samples_from_ckpt(self, ckpt_path):
        # TODO: implement checkpoint save/load
        pass

    def get_batch(self, data):
        """
        Pre-process input batch before train forward step, should be implemented in specific dm class
        """
        raise NotImplementedError

    def get_val_batch(self, data):
        """
        Pre-process input batch before validation forward step, should be implemented in specific dm class
        """
        raise NotImplementedError

    def train_dataloader(self):
        raise NotImplementedError

    def val_dataloader(self):
        raise NotImplementedError

    def test_dataloader(self):
        raise NotImplementedError


# Adapted from accelerate's SkipDataLoader to skip certain number of sequences instead of batches
# https://github.com/huggingface/accelerate/blob/80da9cfb09bb3cc9f1b385cb55d6b90d025a5fd9/src/accelerate/data_loader.py#L858C1-L878C28
class SkipDataLoader(DataLoader):
    """
    Subclass of a PyTorch `DataLoader` that will skip the first batches.

    Args:
        dataset (`torch.utils.data.dataset.Dataset`):
            The dataset to use to build this datalaoder.
        skip_batches (`int`, *optional*, defaults to 0):
            The number of batches to skip at the beginning.
        kwargs:
            All other keyword arguments to pass to the regular `DataLoader` initialization.
    """

    def __init__(self, *args, resume_from_sequence_number=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.resume_from_sequence_number = resume_from_sequence_number
        self.cur_seq_index = 0

    def __iter__(self):
        for batch in super().__iter__():
            num_seq = int(self.batch_size)

            if self.cur_seq_index + num_seq > self.resume_from_sequence_number % (len(self) * self.batch_size):
                yield batch
            else:
                if dist.get_rank() == 0:
                    print(
                        f"Dataloader skipping {num_seq} sequences in this batch as starting from {self.resume_from_sequence_number} sequences"
                    )

            self.cur_seq_index += num_seq
