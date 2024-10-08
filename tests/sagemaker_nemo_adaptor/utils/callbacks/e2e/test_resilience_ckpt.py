import math
import os
import shutil
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import pytest
import pytorch_lightning as pl
import torch
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

from sagemaker_nemo_adaptor.utils.temp_utils import enable_dummy_sm_env

enable_dummy_sm_env()  # Need to be called before torch sagemaker is imported

from pathlib import Path

from sagemaker_nemo_adaptor.constants import SageMakerCheckpointType
from sagemaker_nemo_adaptor.utils.callbacks.local_ckpt_io import (
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

    def __init__(self, warmup_steps, drop_n_warmup_steps, interval_guard):
        self._warmup_steps = warmup_steps
        self._drop_n_warmup_steps = drop_n_warmup_steps
        self._interval_guard = interval_guard
        self._training_step_timings = []
        self._ckpt_write_timings = []

    def on_train_batch_end(
        self,
        trainer: "pl.Trainer",
        *args,
        **kwargs,
    ) -> None:
        self.retrieve_train_time(trainer)
        self.retrieve_write_time(trainer)

    def retrieve_train_time(self, trainer):
        global_step = trainer.global_step
        if global_step > self._warmup_steps + 1 or global_step < self._drop_n_warmup_steps:
            return
        decision_maker = trainer.checkpoint_callback._interval_decision_maker
        self._training_step_timings.append(decision_maker._end - decision_maker._start)

    def retrieve_write_time(self, trainer):
        global_step = trainer.global_step
        if global_step > self._warmup_steps + 1 or global_step < self._drop_n_warmup_steps:
            return

        typ = SageMakerCheckpointType.LOCAL
        checkpoint_io = trainer.strategy.checkpoint_io[typ]
        max_ckpt_duration = checkpoint_io.profiler.max_duration
        self._ckpt_write_timings.append(max_ckpt_duration)

    def calculate_interval(self):
        local_min_train_time = min(self._training_step_timings)
        local_max_write_time = max(self._ckpt_write_timings)
        train_time = torch.tensor(local_min_train_time, dtype=torch.float16, device=torch.cuda.current_device())
        write_time = torch.tensor(local_max_write_time, dtype=torch.float16, device=torch.cuda.current_device())

        dist.all_reduce(train_time, op=dist.ReduceOp.MIN)
        dist.all_reduce(write_time, op=dist.ReduceOp.MAX)
        global_min_train_time = train_time.item()
        global_max_write_time = write_time.item()
        interval = int(math.ceil(global_max_write_time / global_min_train_time * self._interval_guard))
        return int(max(interval, 1))


class ResilienceStateDictRetriever(Callback):
    """Retrieve the state_dict from the given step."""

    def __init__(self, retrieve_step):
        self.retrieve_step = retrieve_step
        self._retrieve_state_dict = None

    def on_train_batch_end(
        self,
        trainer: "pl.Trainer",
        *args,
        **kwargs,
    ) -> None:
        if trainer.global_step == self.retrieve_step:
            trainer.strategy.checkpoint_io.checkpoint_type = SageMakerCheckpointType.LOCAL
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
        config = self.config(model_type=model_type)
        config.exp_manager.exp_dir = temp_dir
        config.exp_manager.checkpoint_dir = os.path.join(temp_dir, "checkpoints")
        self.update_checkpoint_config_with_type(config, SageMakerCheckpointType.LOCAL)

        sample = self.generate_sample(config)

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
        logging.info("Creating a new trainer and loading the checkpoint")
        trainer, data_module, model_module, new_outputs = self.create_and_fit(
            config, model_type=model_type, sample=sample
        )
        new_state_dicts = self.retrieve_state_dicts(trainer)

        for old_state_dict, new_state_dicts in zip(old_state_dicts, new_state_dicts):
            self.check_correctness(old_state_dict, new_state_dicts, data_module.__class__.__qualname__)
        assert_state_dict_equal(old_outputs, new_outputs)
        dist.barrier()

    @skip_if_lt_x_gpu(8)
    def test_resilience_interval(self, temp_dir):

        config = self.config()
        config.exp_manager.exp_dir = temp_dir
        config.exp_manager.checkpoint_dir = os.path.join(temp_dir, "checkpoints")
        self.update_checkpoint_config_with_type(config, SageMakerCheckpointType.LOCAL)

        # Set up for resilience checkpoint with dynamic interval
        config.trainer.max_steps = 6
        config.exp_manager.auto_checkpoint.warmup_steps = 4
        config.exp_manager.auto_checkpoint.drop_n_warmup_steps = 1
        config.exp_manager.auto_checkpoint.interval_guard = 1.25

        # Insert the ResilienceIntervalRetriever callback.
        auto_checkpoint = config.exp_manager.auto_checkpoint
        interval_retriever = ResilienceIntervalRetriever(
            warmup_steps=auto_checkpoint.warmup_steps,
            drop_n_warmup_steps=auto_checkpoint.drop_n_warmup_steps,
            interval_guard=auto_checkpoint.interval_guard,
        )
        trainer, _, _, _ = self.create_and_fit(config, interval_retriever)
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

        config = self.config()
        config.exp_manager.exp_dir = temp_dir
        config.exp_manager.checkpoint_dir = os.path.join(temp_dir, "checkpoints")
        self.update_checkpoint_config_with_type(config, SageMakerCheckpointType.LOCAL)

        # Insert the ResilienceStateDictRetriever callback.
        state_dict_retriever = ResilienceStateDictRetriever(retrieve_step=1)
        trainer, data_module, model_module, _ = self.create_and_fit(config, state_dict_retriever)

        # Two steps are run. Then we remove one of the directory.
        if dist.get_rank() == 0:
            trainer.strategy.checkpoint_io.checkpoint_type = SageMakerCheckpointType.LOCAL
            remove_dir = os.path.join(config.exp_manager.checkpoint_dir, "local", "1", "tp0_ep0_fsdp0")
            shutil.rmtree(remove_dir, ignore_errors=True)
        dist.barrier()
        del trainer, data_module, model_module

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

        config = self.config()
        config.exp_manager.exp_dir = temp_dir
        config.exp_manager.checkpoint_dir = os.path.join(temp_dir, "checkpoints")
        self.update_checkpoint_config_with_type(config, SageMakerCheckpointType.LOCAL)

        # Insert the ResilienceStateDictRetriever callback.
        state_dict_retriever = ResilienceStateDictRetriever(retrieve_step=1)
        trainer, data_module, _, _ = self.create_and_fit(config, state_dict_retriever)

        # Overwrite one of the latest(globa_step 2) checkpoint's local.metadata with global_step 1.
        if dist.get_rank() == 0:
            modify_path = os.path.join(config.exp_manager.checkpoint_dir, "local", "1", "tp0_ep0_fsdp0")
            storage_writer = DistributedFileSystemWriter(modify_path)
            storage_writer.set_up_storage_writer(True)
            modify_path = Path(modify_path)
            SageMakerLocalCheckpointIO.write_local_metadata(1, storage_writer)
        dist.barrier()
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

    @skip_if_lt_x_gpu(8)
    @patch("sagemaker_nemo_adaptor.utils.callbacks.checkpoint.SageMakerModelCheckpointBase._save")
    def test_resilience_save_calls(self, mock_save, temp_dir):
        # Config set up
        config = self.config()
        config.exp_manager.exp_dir = temp_dir
        config.exp_manager.checkpoint_dir = os.path.join(temp_dir, "checkpoints")
        self.update_checkpoint_config_with_type(config, SageMakerCheckpointType.LOCAL)
        config.trainer.max_steps = 7

        self.create_and_fit(
            config,
        )
        assert mock_save.call_count == config.trainer.max_steps
