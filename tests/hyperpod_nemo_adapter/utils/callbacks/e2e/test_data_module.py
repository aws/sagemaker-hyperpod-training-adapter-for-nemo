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
import shutil

import pytest
import torch
import torch.distributed as dist
from lightning.pytorch.callbacks import Callback
from test_utils import TestCheckpoint, assert_state_dict_equal, skip_if_lt_x_gpu

from hyperpod_nemo_adapter.collections.data.datasets import DummyDataset
from hyperpod_nemo_adapter.collections.data.dummy_data_module import DummyDataModule
from hyperpod_nemo_adapter.collections.model.nlp import SageMakerLlamaModel
from hyperpod_nemo_adapter.collections.parts import SageMakerTrainerBuilder
from hyperpod_nemo_adapter.constants import SageMakerCheckpointType


class DummyTestDataset(DummyDataset):
    """Test Dataset.
    Instead of using DummyDataset, we need something deterministic.
    """

    def __getitem__(self, index):
        return index * torch.ones((self.seqlen,), dtype=torch.long), self.mask


class DummyTestDataModuleTest(DummyDataModule):
    """
    Lightning Test DataModule.
    """

    def train_dataloader(self):
        vocab_size = 1024 if self.cfg.model.get("vocab_size", None) is None else self.cfg.model.vocab_size
        self._train_ds = DummyTestDataset(vocab_size=vocab_size, seqlen=self.cfg.model.max_context_width)
        return self._build_dataloader(self._train_ds, batch_size=self.cfg.model.train_batch_size)


class BatchRetriever(Callback):
    """Retrieve the state_dict from the given step."""

    def __init__(
        self,
    ):
        self._last_step_batch = None

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        self._last_step_batch = batch

    @property
    def last_step_batch(self):
        return self._last_step_batch


class TestDataModule(TestCheckpoint):

    @skip_if_lt_x_gpu(8)
    @pytest.mark.parametrize(
        # Skip full checkpoint type as in full mode, only weights are saved.
        "checkpoint_type",
        [
            SageMakerCheckpointType.SHARDED,
            SageMakerCheckpointType.LOCAL,
        ],
    )
    def test_data_module_save_load(self, checkpoint_type, temp_dir):
        # Config set up
        config = self.config()
        config.exp_manager.exp_dir = temp_dir
        config.exp_manager.checkpoint_dir = os.path.join(temp_dir, "checkpoints")
        config.model.vocab_size = 256
        self.update_checkpoint_config_with_type(config, checkpoint_type)
        config.trainer.max_steps = 7

        if checkpoint_type == SageMakerCheckpointType.SHARDED:
            # Train 7 steps but only save checkpoint @ 5.
            # The last step data will be retrieved and compared.
            config.exp_manager.checkpoint_callback_params.save_top_k = 10
            config.exp_manager.checkpoint_callback_params.save_last = False
            config.exp_manager.checkpoint_callback_params.every_n_train_steps = 5

        # Create the trainer and use testdata module instead.
        trainer, _ = SageMakerTrainerBuilder(config).create_trainer()
        model_module = SageMakerLlamaModel(config.model, trainer, use_smp_model=config.use_smp_model)
        test_data_module = DummyTestDataModuleTest(config, trainer)
        batch_retriever = BatchRetriever()
        trainer.callbacks.append(batch_retriever)
        trainer.fit(model_module, datamodule=test_data_module)
        old_batch = batch_retriever.last_step_batch

        if checkpoint_type == SageMakerCheckpointType.SHARDED:
            # Latest checkpoint is saved @ step 5, so extra 2 step will be run.
            sharded_checkpoint_dir = os.path.join(config.exp_manager.checkpoint_dir, "sharded")
            lastest_checkpoint = list(os.scandir(sharded_checkpoint_dir))[-1]
            config.exp_manager.resume_from_checkpoint = lastest_checkpoint.path
        else:
            # Remove the latest checkpoint, so that and older one will be loaded.
            # An extra step after checkpoing loading will be run.
            lastest_checkpoint_id = (config.trainer.max_steps - 1) % 3
            remove_path = os.path.join(config.exp_manager.checkpoint_dir, "local", str(lastest_checkpoint_id))
            shutil.rmtree(remove_path, ignore_errors=True)
            print(f"Remove checkpoint: {remove_path}")
            dist.barrier()

        # Create another trainer and use testdata module instead.
        trainer, _ = SageMakerTrainerBuilder(config).create_trainer()
        model_module = SageMakerLlamaModel(config.model, trainer, use_smp_model=config.use_smp_model)
        test_data_module = DummyTestDataModuleTest(config, trainer)
        batch_retriever = BatchRetriever()
        trainer.callbacks.append(batch_retriever)
        trainer.fit(model_module, datamodule=test_data_module)
        new_batch = batch_retriever.last_step_batch
        # Compare the fetched batch data at the last step between two runs..
        assert_state_dict_equal(old_batch, new_batch)
        dist.barrier()
