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

from typing import Callable, Optional

import torch
from nemo.utils import logging
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.core.datamodule import LightningDataModule
from torch.utils.data import DataLoader

from hyperpod_nemo_adapter.constants import (
    DEFAULT_SEED,
    TRAIN_SEQUENCE_NUMBER,
    VAL_SEQUENCE_NUMBER,
)
from hyperpod_nemo_adapter.utils.app_state import SageMakerAppState


class BaseDataModule(LightningDataModule):
    """
    General Lightning DataModule class for SageMaker adapter, it deals with
    1. Provide general function of build dataloader with sampler
    2. Setup data parallel parameters
    """

    def __init__(self, cfg: DictConfig, trainer: Trainer, collate_fn: Optional[Callable] = None):
        super().__init__()
        self.cfg = cfg
        self.trainer = trainer
        self.collate_fn = collate_fn

    def setup(self, stage=None):
        super().setup(stage)

        app_state = SageMakerAppState()
        self.dp_size = app_state.data_parallel_size
        self.dp_rank = app_state.data_parallel_rank

    @property
    def seed(self):
        return self.cfg.model.seed if self.cfg.model.seed else DEFAULT_SEED

    def _build_dataloader(
        self,
        dataset,
        batch_size=1,
        num_workers=0,
        shuffle=False,
    ):
        """
        Build sampler and dataloader
        """
        sampler = torch.utils.data.DistributedSampler(
            dataset,
            shuffle=shuffle,
            seed=self.seed,
            rank=self.dp_rank,
            num_replicas=self.dp_size,
            drop_last=True,
        )

        kwargs = {
            "sampler": sampler,
            "batch_size": batch_size,
            "num_workers": num_workers,
            "collate_fn": self.collate_fn,
            "pin_memory": True,
            "drop_last": True,
        }

        return SkipDataLoader(dataset, **kwargs)

    def _retrieve_squence_index(self, dataloaders):
        """Given a (list) dataloader, retrieve the current sequence number from it."""
        if isinstance(dataloaders, list):
            sequence_nums = [d.cur_seq_index for d in dataloaders]
        else:
            sequence_nums = [dataloaders.cur_seq_index]
        return sequence_nums

    def state_dict(self):
        """Generate state_dict to save during checkpoint."""
        state_dict = {"seed": self.seed}
        if self.trainer.train_dataloader:
            state_dict[TRAIN_SEQUENCE_NUMBER] = self._retrieve_squence_index(self.trainer.train_dataloader)

        if self.trainer.val_dataloaders:
            state_dict[VAL_SEQUENCE_NUMBER] = self._retrieve_squence_index(self.trainer.val_dataloaders)
        return state_dict

    def _resume_squence_index(self, sequence_nums, dataloaders):
        """
        Given a (list) dataloader and sequence_nums, set the resume_from_sequence_number in
        the dataloader so that it will skip the batches.
        """
        if isinstance(dataloaders, list):
            assert len(sequence_nums) == len(dataloaders), (
                f"sequence_nums length " f"{len(sequence_nums)} is not equal to dataloaders length {len(dataloaders)}"
            )
            for i in range(len(dataloaders)):
                dataloaders[i].resume_from_sequence_number = sequence_nums[i]
        else:
            assert (
                len(sequence_nums) == 1
            ), f"Only have one dataloader but sequence_nums has lenght of {len(sequence_nums)}"
            dataloaders.resume_from_sequence_number = sequence_nums[0]

    def load_state_dict(self, state_dict):
        """
        Load the state_dict into dataloaders if they exist.
        """
        class_name = self.__class__.__qualname__
        assert class_name in state_dict, f"{class_name} is not in state_dict"

        seed = state_dict[class_name].get("seed", None)
        assert seed == self.seed, f"Seed in state_dict[{class_name}] is {seed} but current seed is {self.seed}"

        if self.trainer.train_dataloader:
            assert (
                TRAIN_SEQUENCE_NUMBER in state_dict[class_name]
            ), f"Could not find {TRAIN_SEQUENCE_NUMBER} is not in state_dict[{class_name}]"
            self._resume_squence_index(state_dict[class_name][TRAIN_SEQUENCE_NUMBER], self.trainer.train_dataloader)

        if self.trainer.val_dataloaders:
            assert (
                VAL_SEQUENCE_NUMBER in state_dict[class_name]
            ), f"Could not find {VAL_SEQUENCE_NUMBER} is not in state_dict[{class_name}]"
            self._resume_squence_index(state_dict[class_name][VAL_SEQUENCE_NUMBER], self.trainer.val_dataloaders)

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
        resume_from_sequence_number (`int`, *optional*, defaults to 0):
            The number of batches to skip at the beginning.
        kwargs:
            All other keyword arguments to pass to the regular `DataLoader` initialization.
    """

    def __init__(self, *args, resume_from_sequence_number=0, **kwargs):
        super().__init__(*args, **kwargs)
        self._resume_from_sequence_number = resume_from_sequence_number
        self.cur_seq_index = 0

    @property
    def resume_from_sequence_number(self):
        return self._resume_from_sequence_number

    @resume_from_sequence_number.setter
    def resume_from_sequence_number(self, sequence_number):
        self._resume_from_sequence_number = sequence_number

    def __iter__(self):
        for batch in super().__iter__():
            num_seq = int(self.batch_size)
            self.cur_seq_index += num_seq

            if self.cur_seq_index > self._resume_from_sequence_number % (len(self) * self.batch_size):
                yield batch
            elif self.cur_seq_index + num_seq > self._resume_from_sequence_number:
                logging.info(
                    f"Dataloader skipped {self.cur_seq_index} sequences in this batch as starting from {self._resume_from_sequence_number} sequences"
                )
