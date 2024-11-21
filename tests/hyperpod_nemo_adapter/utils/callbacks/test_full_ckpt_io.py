import os
from unittest.mock import MagicMock, patch

import pytest
from lightning_fabric.plugins import TorchCheckpointIO

from hyperpod_nemo_adapter.utils.callbacks.full_ckpt_io import SageMakerFullCheckpointIO


class TestSageMakerFullCheckpointIO:
    @pytest.fixture
    def checkpoint_io(
        self,
    ):
        return SageMakerFullCheckpointIO()

    @pytest.fixture
    def mock_get_rank(sefl, monkeypatch):
        mock_rank = MagicMock()
        monkeypatch.setattr("hyperpod_nemo_adapter.utils.callbacks.full_ckpt_io.dist.get_rank", mock_rank)
        return mock_rank

    @patch("hyperpod_nemo_adapter.utils.callbacks.full_ckpt_io.dist.get_rank")
    def test_save_checkpoint_non_zero_rank(self, mock_get_rank, checkpoint_io):
        mock_get_rank.return_value = 1
        checkpoint = {"state_dict": {"key": "value"}}
        path = "/path/to/checkpoint"
        storage_options = None

        with patch.object(TorchCheckpointIO, "save_checkpoint") as mock_super_save_checkpoint:
            checkpoint_io.save_checkpoint(checkpoint, path, storage_options)
            mock_super_save_checkpoint.assert_not_called()

    @patch("hyperpod_nemo_adapter.utils.callbacks.full_ckpt_io.dist.get_rank")
    def test_save_checkpoint_rank_zero(self, mock_get_rank, checkpoint_io):
        mock_get_rank.return_value = 0
        checkpoint = {"state_dict": {"key": "value"}}
        path = "/path/to/checkpoint"
        storage_options = None

        with patch.object(TorchCheckpointIO, "save_checkpoint") as mock_super_save_checkpoint:
            checkpoint_io.save_checkpoint(checkpoint, path, storage_options)
            mock_super_save_checkpoint.assert_called_once_with(
                checkpoint["state_dict"], os.path.join(path, "pytorch_model.bin")
            )
