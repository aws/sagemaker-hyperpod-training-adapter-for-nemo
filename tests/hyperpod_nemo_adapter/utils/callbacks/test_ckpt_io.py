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

from unittest.mock import patch

import pytest

from hyperpod_nemo_adapter.constants import SageMakerCheckpointType
from hyperpod_nemo_adapter.utils.callbacks.ckpt_io import SageMakerCheckpointIO
from hyperpod_nemo_adapter.utils.callbacks.full_ckpt_io import SageMakerFullCheckpointIO
from hyperpod_nemo_adapter.utils.callbacks.local_ckpt_io import (
    SageMakerLocalCheckpointIO,
)
from hyperpod_nemo_adapter.utils.callbacks.sharded_ckpt_io import (
    SageMakerShardedCheckpointIO,
)


class TestSageMakerCheckpointIO:
    @pytest.fixture
    def checkpoint_io(
        self,
    ):
        return SageMakerCheckpointIO()

    @pytest.fixture
    def trainer_mock(self, mocker):
        return mocker.MagicMock()

    def test_init(self, checkpoint_io):
        assert checkpoint_io.checkpoint_type == SageMakerCheckpointType.SHARDED
        assert isinstance(checkpoint_io._checkpoint_io[SageMakerCheckpointType.SHARDED], SageMakerShardedCheckpointIO)
        assert isinstance(checkpoint_io._checkpoint_io[SageMakerCheckpointType.LOCAL], SageMakerLocalCheckpointIO)
        assert isinstance(checkpoint_io._checkpoint_io[SageMakerCheckpointType.FULL], SageMakerFullCheckpointIO)

    @patch("hyperpod_nemo_adapter.utils.callbacks.sharded_ckpt_io.SageMakerShardedCheckpointIO.save_checkpoint")
    def test_save_checkpoint(self, mock_sharded_ckpt_io_save, checkpoint_io):
        checkpoint = {"state_dict": {"key": "value"}}
        path = "/path/to/checkpoint"
        storage_options = None
        checkpoint_io.save_checkpoint(checkpoint, path, storage_options)

        mock_sharded_ckpt_io_save.assert_called_once_with(checkpoint, path, storage_options)

    @patch("hyperpod_nemo_adapter.utils.callbacks.sharded_ckpt_io.SageMakerShardedCheckpointIO.load_checkpoint")
    def test_load_checkpoint(self, mock_sharded_ckpt_io_load, checkpoint_io, trainer_mock):
        path = "/path/to/checkpoint"
        map_location = None

        checkpoint_io.load_checkpoint(path, trainer_mock, map_location)
        mock_sharded_ckpt_io_load.assert_called_once_with(path, trainer_mock, map_location)

    @patch("hyperpod_nemo_adapter.utils.callbacks.sharded_ckpt_io.SageMakerShardedCheckpointIO.remove_checkpoint")
    def test_remove_checkpoint(self, mock_sharded_ckpt_io_remove, checkpoint_io):
        path = "/path/to/checkpoint"

        checkpoint_io.remove_checkpoint(path)
        mock_sharded_ckpt_io_remove.assert_called_once_with(path)

    def test_checkpoint_type_property(self, checkpoint_io):
        assert checkpoint_io.checkpoint_type == SageMakerCheckpointType.SHARDED
        checkpoint_io.checkpoint_type = SageMakerCheckpointType.LOCAL
        assert checkpoint_io.checkpoint_type == SageMakerCheckpointType.LOCAL

    @patch("hyperpod_nemo_adapter.utils.callbacks.sharded_ckpt_io.SageMakerShardedCheckpointIO.teardown")
    @patch("hyperpod_nemo_adapter.utils.callbacks.local_ckpt_io.SageMakerLocalCheckpointIO.teardown")
    @patch("hyperpod_nemo_adapter.utils.callbacks.peft_ckpt_io.SageMakerPeftShardedCheckpointIO.teardown")
    def test_teardown(
        self,
        mock_sharded_ckpt_io_teardown,
        mock_local_ckpt_io_teardown,
        mock_peft_sharded_ckpt_io_teardown,
        checkpoint_io,
    ):
        checkpoint_io.teardown(None)
        mock_sharded_ckpt_io_teardown.assert_called_once()
        mock_local_ckpt_io_teardown.assert_called_once()
        mock_peft_sharded_ckpt_io_teardown.assert_called_once()
