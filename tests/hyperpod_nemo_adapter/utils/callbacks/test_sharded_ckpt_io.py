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
import tempfile
from unittest.mock import MagicMock, patch

import pytest
import pytorch_lightning as pl
from lightning.pytorch.trainer.connectors.checkpoint_connector import (
    _CheckpointConnector,
)

from hyperpod_nemo_adapter.utils.callbacks.sharded_ckpt_io import (
    SageMakerShardedCheckpointIO,
)


class TestSageMakerShardedCheckpointIO:

    @pytest.fixture
    def checkpoint_io(
        self,
    ):
        checkpoint_io = SageMakerShardedCheckpointIO()
        app_state_mock = MagicMock()
        checkpoint_io.app_state = app_state_mock
        return checkpoint_io

    @pytest.fixture
    def trainer_mock(
        self,
    ):
        trainer_mock = MagicMock(spec=pl.Trainer)
        trainer_mock._checkpoint_connector = MagicMock(spec=_CheckpointConnector)
        state_dict = {"state_dict": {"a": 1, "b": 2}, "optimizer_states": {"step": 0}}
        trainer_mock._checkpoint_connector.dump_checkpoint.return_value = state_dict
        return trainer_mock

    @pytest.fixture
    def temp_dir(
        self,
    ):
        temp_dir = tempfile.mkdtemp()
        yield temp_dir

    @patch("hyperpod_nemo_adapter.utils.callbacks.sharded_ckpt_io.DistributedFileSystemWriter")
    @patch("hyperpod_nemo_adapter.utils.callbacks.sharded_ckpt_io.saver.async_save")
    def test_save_checkpoint(self, mock_async_save, mock_storage_writer, checkpoint_io, trainer_mock, temp_dir):
        path = f"{temp_dir}/checkpoint.ckpt"
        checkpoint_io.save_checkpoint(
            trainer_mock._checkpoint_connector.dump_checkpoint.return_value, path, trainer_mock
        )
        mock_async_save.assert_called_once_with(
            trainer_mock._checkpoint_connector.dump_checkpoint.return_value,
            storage_writer=mock_storage_writer.return_value,
            process_group=checkpoint_io.app_state.fsdp_process_group,
            coordinator_rank=checkpoint_io.app_state.fsdp_coordinator_rank,
            queue=checkpoint_io.queue,
            force_check_all_plans=False,
            wait_error_handling=False,
        )

    @patch("hyperpod_nemo_adapter.utils.callbacks.sharded_ckpt_io.loader.load")
    @patch(
        "hyperpod_nemo_adapter.utils.callbacks.base_ckpt_io.SageMakerBaseCheckpointIO.load_data_module_and_lr_schedulers"
    )
    def test_load_checkpoint(
        self, mock_load_data_module_and_lr_schedulers, mock_load, checkpoint_io, trainer_mock, temp_dir
    ):
        path = f"{temp_dir}/checkpoint.ckpt"
        state_dict = trainer_mock._checkpoint_connector.dump_checkpoint.return_value
        checkpoint_io.load_checkpoint(path, trainer_mock)

        trainer_mock._checkpoint_connector.dump_checkpoint.assert_called_once_with(False)
        mock_load.assert_called_once_with(
            state_dict,
            checkpoint_id=path,
            process_group=checkpoint_io.app_state.fsdp_process_group,
            coordinator_rank=checkpoint_io.app_state.fsdp_coordinator_rank,
        )
        mock_load_data_module_and_lr_schedulers.assert_called_once_with(trainer_mock, state_dict)

    def test_remove_checkpoint(self, checkpoint_io, temp_dir):
        checkpoint_dir = f"{temp_dir}/"
        checkpoint_file = os.path.join(checkpoint_dir, "checkpoint.ckpt")
        with open(checkpoint_file, "w") as f:
            f.write("test")

        checkpoint_io.remove_checkpoint(checkpoint_dir)
        assert not os.path.exists(checkpoint_dir)

    @patch("hyperpod_nemo_adapter.utils.callbacks.sharded_ckpt_io.AsyncCallsQueue.maybe_finalize_async_calls")
    def test_teardown(self, mock_finalize_async_calls, checkpoint_io):
        checkpoint_io.teardown()
        mock_finalize_async_calls.assert_called_once_with(
            blocking=True, process_group=checkpoint_io.app_state.fsdp_process_group
        )
