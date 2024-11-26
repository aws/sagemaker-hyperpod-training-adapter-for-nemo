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

from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.core.datamodule import LightningDataModule
from torch.distributed import init_process_group
from torch.utils.data import DataLoader

# local modules
from hyperpod_nemo_adapter.collections.data.base import BaseDataModule, SkipDataLoader

MODULE_PATH = "hyperpod_nemo_adapter.collections.data.base"


def build_base_data_module(config: DictConfig = {}, trainer: Trainer = None):
    if trainer is None:
        trainer = Trainer()

    return BaseDataModule(config, trainer)


class TestBaseDataModule:
    class MockedAppState:
        def __init__(self):
            self.count = 0
            self.data_parallel_size = 11
            self.data_parallel_rank = 12

        def __call__(self):
            self.count += 1

    def test_is_LighteningDataModule_subclass(self):
        assert issubclass(BaseDataModule, LightningDataModule)

    def test_init(self):
        config: DictConfig = {}
        trainer = Trainer()
        base = build_base_data_module(config, trainer)
        assert base.cfg == config
        assert base.trainer == trainer

    def test_setup(self, mocker):
        # setup
        mocked_app_state = self.MockedAppState()
        patch = mocker.patch(f"{MODULE_PATH}.SageMakerAppState", return_value=mocked_app_state)
        base = build_base_data_module()

        # action
        base.setup()

        # assertions
        patch.assert_called_once()
        assert base.dp_size == mocked_app_state.data_parallel_size
        assert base.dp_rank == mocked_app_state.data_parallel_rank


class TestSkipDataLoader:
    test_dataset = [
        {
            "text": "Beginners BBQ Class Taking Place in Missoula!\nDo you want to get better at making delicious BBQ?",
            "timestamp": "2019-04-25T12:57:54Z",
            "url": "https://klyq.com/beginners-bbq-class-taking-place-in-missoula/",
        }
    ]

    def test_is_DataLoader_subclass(self):
        assert issubclass(SkipDataLoader, DataLoader)

    def test_init(self):
        sdl = SkipDataLoader(self.test_dataset, resume_from_sequence_number=7)
        assert sdl.resume_from_sequence_number == 7
        assert sdl.cur_seq_index == 0

    def test_iterable(self):
        # setup
        data_entry_count = 7
        concat_dataset = [self.test_dataset[0].copy() for _ in range(data_entry_count)]
        resume_sequence_number = 3
        sdl = SkipDataLoader(concat_dataset, resume_from_sequence_number=resume_sequence_number)

        # required before iterating over instance
        # https://pytorch.org/tutorials/beginner/ddp_series_multigpu.html#constructing-the-process-group
        # https://pytorch.org/docs/stable/distributed.html#tcp-initialization
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"
        init_process_group(backend="gloo", rank=0, world_size=1)

        # assertions
        assert sum(bool(data) for data in sdl) == data_entry_count - resume_sequence_number
        assert sdl.cur_seq_index == data_entry_count

        # teardown
        del os.environ["MASTER_ADDR"]
        del os.environ["MASTER_PORT"]
