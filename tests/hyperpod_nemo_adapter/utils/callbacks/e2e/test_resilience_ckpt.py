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
from copy import deepcopy
from pathlib import Path

import pytest
import pytorch_lightning as pl
import torch.distributed as dist
from lightning.pytorch.callbacks import Callback
from nemo.utils import logging
from test_utils import (
    TestCheckpoint,
    assert_state_dict_equal,
    assert_values,
    skip_if_lt_x_gpu,
)
from torch.sagemaker.distributed.checkpoint.filesystem import (
    DistributedFileSystemWriter,
)
from torch.sagemaker.distributed.checkpoint.state_dict_utils import (
    compute_auto_checkpoint_interval,
)

from hyperpod_nemo_adapter.constants import SageMakerCheckpointType
from hyperpod_nemo_adapter.utils.callbacks.local_ckpt_io import (
    SageMakerLocalCheckpointIO,
)


class ResilienceIntervalRetriever(Callback):
    """Accumulate training timings and write timings.

    Then cauculate the interval to see if it matches the interval from resilience callback.

    Note this callback is append to the last in trainer.callbacks, ie: should be called after checkpoint callback.

    The logic diff is:
    1. accumulate all train step timings + ckpt timings instead of updating each step.
    2. Find the local max/min
    2. use all_reduce to get global min train time and max write time.
    """

    def __init__(self, warmup_steps, drop_n_warmup_steps):
        self.warmup_start = drop_n_warmup_steps
        self.ckpt_warmup_end = self.warmup_start + warmup_steps
        self.step_warmup_end = self.ckpt_warmup_end + warmup_steps
        self.step_durations = []
        self.step_ckpt_durations = []
        self.ckpt_preprocessing_durations = []
        self.ckpt_io_durations = []
        self.step = 0

    def on_train_batch_end(
        self,
        trainer: "pl.Trainer",
        *args,
        **kwargs,
    ) -> None:
        self.step += 1
        self.retrieve(trainer)

    def on_train_batch_start(
        self,
        trainer: "pl.Trainer",
        *args,
        **kwargs,
    ) -> None:
        typ = SageMakerCheckpointType.LOCAL
        checkpoint_io = trainer.strategy.checkpoint_io[typ]
        if self.step <= self.warmup_start - 1:
            return
        if self.step <= self.ckpt_warmup_end - 1:
            self.ckpt_preprocessing_durations.append(checkpoint_io.ckpt_preprocessing_duration)

    def retrieve(self, trainer):
        decision_maker = trainer.checkpoint_callback._interval_decision_maker
        step_duration = decision_maker._end - decision_maker._start
        if step_duration == 0:
            return
        if self.step <= self.warmup_start:
            return
        typ = SageMakerCheckpointType.LOCAL
        checkpoint_io = trainer.strategy.checkpoint_io[typ]
        if self.step <= self.ckpt_warmup_end:
            self.step_ckpt_durations.append(step_duration)
            self.ckpt_io_durations.append(checkpoint_io.io_duration)
            return
        if self.step <= self.step_warmup_end:
            self.step_durations.append(step_duration)
            return

    def calculate_interval(self):
        # Merge all durations
        interval = compute_auto_checkpoint_interval(
            self.step_durations,
            self.step_ckpt_durations,
            self.ckpt_preprocessing_durations,
            self.ckpt_io_durations,
        )
        return interval.interval


class StateDictRetriever(Callback):
    """Retrieve the state_dict from the given step."""

    def __init__(self, retrieve_step, checkpoint_type=SageMakerCheckpointType.LOCAL):
        self._checkpoint_type = checkpoint_type
        self.retrieve_step = retrieve_step
        self._retrieve_state_dict = None

    def on_train_batch_end(
        self,
        trainer: "pl.Trainer",
        *args,
        **kwargs,
    ) -> None:
        if trainer.global_step == self.retrieve_step:
            trainer.strategy.checkpoint_io.checkpoint_type = self._checkpoint_type
            self._retrieve_state_dict = deepcopy(trainer._checkpoint_connector.dump_checkpoint(weights_only=False))
            if dist.get_rank() == 0:
                logging.info(f"Retrieve state dict at step {self.retrieve_step}.")

    @property
    def retrieve_state_dict(self):
        return self._retrieve_state_dict


class TestResilienceCheckpoint(TestCheckpoint):

    @skip_if_lt_x_gpu(8)
    @pytest.mark.parametrize(
        "model_type",
        [("llama"), ("mistral"), ("mixtral")],
    )
    def test_resilience_save_and_load(self, model_type, temp_dir):
        # Config set up
        ports = self.find_free_network_ports()
        ports = self.broadcast_ports(ports)
        config = self.config(model_type=model_type)
        config.exp_manager.exp_dir = temp_dir
        config.exp_manager.checkpoint_dir = os.path.join(temp_dir, "checkpoints")
        self.update_checkpoint_config_with_type(config, SageMakerCheckpointType.LOCAL)

        sample = self.generate_sample(config)

        self.reset_state_and_groups(ports[0])
        trainer, data_module, model_module, old_outputs = self.create_and_fit(
            config, model_type=model_type, sample=sample
        )
        # Make sure we only save/load with resilicence checkpoint, and the other types of state_dict
        # are still equal.
        old_state_dicts = self.retrieve_state_dicts(trainer)

        # Check saved checkpoint files.
        assert os.path.exists(os.path.join(config.exp_manager.checkpoint_dir, "local"))
        model_config = config.model
        fsdp_degree = model_config.get("shard_degree", 1)
        tp_degree = model_config.get("tensor_model_parallel_degree", 1)
        ep_degree = model_config.get("expert_model_parallel_degree", 1)
        total_degree = fsdp_degree * tp_degree * ep_degree
        assert len(list(os.scandir(os.path.join(config.exp_manager.checkpoint_dir, "local", "0")))) == total_degree

        del trainer, data_module, model_module

        # Create a new trainer and load the checkpoint
        # No checkpoint path needs to be set during loading, as it should auto resume.
        self.reset_state_and_groups(ports[1])
        logging.info("Creating a new trainer and loading the checkpoint")
        trainer, data_module, model_module, new_outputs = self.create_and_fit(
            config, model_type=model_type, sample=sample
        )
        new_state_dicts = self.retrieve_state_dicts(trainer)

        for old_state_dict, new_state_dict in zip(old_state_dicts, new_state_dicts):
            self.check_correctness(old_state_dict, new_state_dict, data_module.__class__.__qualname__)
        assert_state_dict_equal(old_outputs, new_outputs)
        dist.barrier()

    @skip_if_lt_x_gpu(8)
    def test_resilience_interval(self, temp_dir):
        ports = self.find_free_network_ports()
        ports = self.broadcast_ports(ports)
        self.reset_state_and_groups(ports[0])
        config = self.config()
        config.exp_manager.exp_dir = temp_dir
        config.exp_manager.checkpoint_dir = os.path.join(temp_dir, "checkpoints")
        self.update_checkpoint_config_with_type(config, SageMakerCheckpointType.LOCAL)

        self.reset_state_and_groups(ports[1])
        # Set up for resilience checkpoint with dynamic interval
        config.trainer.max_steps = 16
        config.exp_manager.auto_checkpoint.warmup_steps = 4
        config.exp_manager.auto_checkpoint.drop_n_warmup_steps = 2

        # Insert the ResilienceIntervalRetriever callback.
        auto_checkpoint = config.exp_manager.auto_checkpoint
        interval_retriever = ResilienceIntervalRetriever(
            warmup_steps=auto_checkpoint.warmup_steps,
            drop_n_warmup_steps=auto_checkpoint.drop_n_warmup_steps,
        )
        trainer, _, _, _ = self.create_and_fit(config, callbacks=[interval_retriever])
        trainer.strategy.checkpoint_io.checkpoint_type = SageMakerCheckpointType.LOCAL

        # Check that resilience callback intervals is the same as manual calculated one.
        # There is only one checkpoint callback.
        resilience_callback = trainer.checkpoint_callback
        resilience_interval_old = resilience_callback._every_n_train_steps
        resilience_callback_state_dict_old = resilience_callback.state_dict()
        assert_values(resilience_interval_old, interval_retriever.calculate_interval(), "")

        # Create a new trainer and load the checkpoint
        # No checkpoint path needs to be set during loading, as it should auto resume.
        trainer, _, _, _ = self.create_and_fit(config)
        trainer.strategy.checkpoint_io.checkpoint_type = SageMakerCheckpointType.LOCAL
        resilience_callback = trainer.checkpoint_callback
        resilience_interval_new = resilience_callback._every_n_train_steps
        resilience_callback_state_dict_new = resilience_callback.state_dict()

        # Check resilience_callback_state_dict and intervals are the same.
        assert_values(resilience_interval_old, resilience_interval_new)
        assert_state_dict_equal(resilience_callback_state_dict_old, resilience_callback_state_dict_new)
        dist.barrier()

    @skip_if_lt_x_gpu(8)
    def test_load_with_corrupted_checkpoint(self, temp_dir):
        """Simulate one node fails."""
        ports = self.find_free_network_ports()
        ports = self.broadcast_ports(ports)
        self.reset_state_and_groups(ports[0])
        config = self.config()
        config.exp_manager.exp_dir = temp_dir
        config.exp_manager.checkpoint_dir = os.path.join(temp_dir, "checkpoints")
        self.update_checkpoint_config_with_type(config, SageMakerCheckpointType.LOCAL)

        # Insert the StateDictRetriever callback.
        state_dict_retriever = StateDictRetriever(retrieve_step=1)
        trainer, data_module, model_module, _ = self.create_and_fit(config, callbacks=state_dict_retriever)

        # Two steps are run. Then we remove one of the directory.
        if dist.get_rank() == 0:
            trainer.strategy.checkpoint_io.checkpoint_type = SageMakerCheckpointType.LOCAL
            remove_dir = os.path.join(config.exp_manager.checkpoint_dir, "local", "1", "tp0_ep0_fsdp0")
            shutil.rmtree(remove_dir, ignore_errors=True)
        dist.barrier()
        del trainer, data_module, model_module

        self.reset_state_and_groups(ports[1])
        # After removing, the latest_checkpoint will be skipped, and loaded the next one. In this case,
        # it is from .../local/0/ which is saved at global_step 1, the state_dict is retrieved by
        # state_dict_retriever
        # set the max_steps to be 1 here so that it won't continue training after loading the checkpoint.
        config.trainer.max_steps = 1
        trainer, data_module, _, _ = self.create_and_fit(config)
        new_state_dict = self.retrieve_state_dicts(trainer, checkpoint_types=[SageMakerCheckpointType.LOCAL])[0]

        # Check resilience_callback_state_dict and intervals are the same.
        self.check_correctness(
            new_state_dict, state_dict_retriever.retrieve_state_dict, data_module.__class__.__qualname__
        )
        dist.barrier()

    @skip_if_lt_x_gpu(8)
    def test_slow_write_checkpoint(self, temp_dir):
        """Simulate if one of the checkpoint is behind."""
        ports = self.find_free_network_ports()
        ports = self.broadcast_ports(ports)
        self.reset_state_and_groups(ports[0])
        config = self.config()
        config.exp_manager.exp_dir = temp_dir
        config.exp_manager.checkpoint_dir = os.path.join(temp_dir, "checkpoints")
        self.update_checkpoint_config_with_type(config, SageMakerCheckpointType.LOCAL)

        # Insert the StateDictRetriever callback.
        state_dict_retriever = StateDictRetriever(retrieve_step=1)
        trainer, data_module, _, _ = self.create_and_fit(config, callbacks=state_dict_retriever)

        # Overwrite one of the latest(globa_step 2) checkpoint's local.metadata with global_step 1.
        if dist.get_rank() == 0:
            modify_path = os.path.join(config.exp_manager.checkpoint_dir, "local", "1", "tp0_ep0_fsdp0")
            storage_writer = DistributedFileSystemWriter(modify_path)
            storage_writer.set_up_storage_writer(True)
            modify_path = Path(modify_path)
            SageMakerLocalCheckpointIO.write_local_metadata(1, storage_writer)
        dist.barrier()

        self.reset_state_and_groups(ports[1])
        # After modifying, the latest_checkpoint will be skipped, and loaded the next one. In this case,
        # it is from .../local/0/ which is saved at global_step 1, the state_dict is retrieved by
        # state_dict_retriever
        config.trainer.max_steps = 1
        trainer, data_module, _, _ = self.create_and_fit(config)
        trainer.strategy.checkpoint_io.checkpoint_type = SageMakerCheckpointType.LOCAL
        new_state_dict = trainer._checkpoint_connector.dump_checkpoint(weights_only=False)

        # Check resilience_callback_state_dict and intervals are the same.
        self.check_correctness(
            state_dict_retriever.retrieve_state_dict, new_state_dict, data_module.__class__.__qualname__
        )
        dist.barrier()

    @skip_if_lt_x_gpu(8)
    def test_resilience_max_save(self, temp_dir):
        # Config set up
        ports = self.find_free_network_ports()
        ports = self.broadcast_ports(ports)
        self.reset_state_and_groups(ports[0])
        config = self.config()
        config.exp_manager.exp_dir = temp_dir
        config.exp_manager.checkpoint_dir = os.path.join(temp_dir, "checkpoints")
        self.update_checkpoint_config_with_type(config, SageMakerCheckpointType.LOCAL)
        config.trainer.max_steps = 7

        trainer, _, _, _ = self.create_and_fit(
            config,
        )

        # Check the number of saved checkpoint files.
        assert (
            len(list(os.scandir(os.path.join(config.exp_manager.checkpoint_dir, "local"))))
            == trainer.checkpoint_callback.save_top_k
        )
        dist.barrier()

    @skip_if_lt_x_gpu(8)
    def test_resilience_save_calls(self, temp_dir):
        # Config set up
        ports = self.find_free_network_ports()
        ports = self.broadcast_ports(ports)
        self.reset_state_and_groups(ports[0])
        config = self.config()
        config.exp_manager.exp_dir = temp_dir
        config.exp_manager.checkpoint_dir = os.path.join(temp_dir, "checkpoints")
        self.update_checkpoint_config_with_type(config, SageMakerCheckpointType.LOCAL)
        config.trainer.max_steps = 7
        config.exp_manager.auto_checkpoint.warmup_steps = 3
        config.exp_manager.auto_checkpoint.drop_n_warmup_steps = 2

        trainer, _, _, _ = self.create_and_fit(
            config,
        )
        assert (
            len(trainer.checkpoint_callback._interval_decision_maker.ckpt_io_durations)
            == config.exp_manager.auto_checkpoint.warmup_steps
        )
        assert (
            len(trainer.checkpoint_callback._interval_decision_maker.step_durations)
            == config.trainer.max_steps
            - config.exp_manager.auto_checkpoint.warmup_steps
            - config.exp_manager.auto_checkpoint.drop_n_warmup_steps
        )
        dist.barrier()
