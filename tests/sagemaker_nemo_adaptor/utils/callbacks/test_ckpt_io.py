from unittest.mock import patch

import pytest

from sagemaker_nemo_adaptor.constants import SageMakerCheckpointType
from sagemaker_nemo_adaptor.utils.callbacks.ckpt_io import SageMakerCheckpointIO
from sagemaker_nemo_adaptor.utils.callbacks.full_ckpt_io import (
    SageMakerFullCheckpointIO,
)
from sagemaker_nemo_adaptor.utils.callbacks.local_ckpt_io import (
    SageMakerLocalCheckpointIO,
)
from sagemaker_nemo_adaptor.utils.callbacks.sharded_ckpt_io import (
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

    @patch("sagemaker_nemo_adaptor.utils.callbacks.sharded_ckpt_io.SageMakerShardedCheckpointIO.save_checkpoint")
    def test_save_checkpoint(self, mock_sharded_ckpt_io_save, checkpoint_io):
        checkpoint = {"state_dict": {"key": "value"}}
        path = "/path/to/checkpoint"
        storage_options = None
        checkpoint_io.save_checkpoint(checkpoint, path, storage_options)

        mock_sharded_ckpt_io_save.assert_called_once_with(checkpoint, path, storage_options)

    @patch("sagemaker_nemo_adaptor.utils.callbacks.sharded_ckpt_io.SageMakerShardedCheckpointIO.load_checkpoint")
    def test_load_checkpoint(self, mock_sharded_ckpt_io_load, checkpoint_io, trainer_mock):
        path = "/path/to/checkpoint"
        map_location = None

        checkpoint_io.load_checkpoint(path, trainer_mock, map_location)
        mock_sharded_ckpt_io_load.assert_called_once_with(path, trainer_mock, map_location)

    @patch("sagemaker_nemo_adaptor.utils.callbacks.sharded_ckpt_io.SageMakerShardedCheckpointIO.remove_checkpoint")
    def test_remove_checkpoint(self, mock_sharded_ckpt_io_remove, checkpoint_io):
        path = "/path/to/checkpoint"

        checkpoint_io.remove_checkpoint(path)
        mock_sharded_ckpt_io_remove.assert_called_once_with(path)

    def test_checkpoint_type_property(self, checkpoint_io):
        assert checkpoint_io.checkpoint_type == SageMakerCheckpointType.SHARDED
        checkpoint_io.checkpoint_type = SageMakerCheckpointType.LOCAL
        assert checkpoint_io.checkpoint_type == SageMakerCheckpointType.LOCAL

    @patch("sagemaker_nemo_adaptor.utils.callbacks.sharded_ckpt_io.SageMakerShardedCheckpointIO.teardown")
    @patch("sagemaker_nemo_adaptor.utils.callbacks.local_ckpt_io.SageMakerLocalCheckpointIO.teardown")
    def test_teardown(self, mock_sharded_ckpt_io_teardown, mock_local_ckpt_io_teardown, checkpoint_io):
        checkpoint_io.teardown(None)
        mock_sharded_ckpt_io_teardown.assert_called_once()
        mock_local_ckpt_io_teardown.assert_called_once()
