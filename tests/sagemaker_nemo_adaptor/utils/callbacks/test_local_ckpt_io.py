from unittest.mock import MagicMock, patch

import pytest
import pytorch_lightning as pl
from lightning.pytorch.trainer.connectors.checkpoint_connector import (
    _CheckpointConnector,
)

from sagemaker_nemo_adaptor.utils.app_state import SageMakerAppState
from sagemaker_nemo_adaptor.utils.callbacks.local_ckpt_io import (
    SageMakerLocalCheckpointIO,
)


class TestSageMakerLocalCheckpointIO:

    @pytest.fixture
    def checkpoint_io(
        self,
    ):
        return SageMakerLocalCheckpointIO()

    @pytest.fixture
    def app_state(
        self,
    ):
        return SageMakerAppState()

    @pytest.fixture
    def trainer_mock(
        self,
    ):
        trainer = MagicMock(spec=pl.Trainer)
        trainer._checkpoint_connector = MagicMock(spec=_CheckpointConnector)
        trainer._checkpoint_connector.dump_checkpoint.return_value = {"state_dict": {"a": 1, "b": 2}}
        return trainer

    @patch("sagemaker_nemo_adaptor.utils.callbacks.local_ckpt_io.saver.async_save")
    def test_save_checkpoint(self, mock_async_save, checkpoint_io, app_state, trainer_mock):
        checkpoint_io.app_state = app_state
        path = "path/to/checkpoint"
        checkpoint_io.save_checkpoint(trainer_mock._checkpoint_connector.dump_checkpoint.return_value, path)
        mock_async_save.assert_called_once_with(
            trainer_mock._checkpoint_connector.dump_checkpoint.return_value,
            checkpoint_id=path,
            process_group=app_state.current_replication_group,
            coordinator_rank=app_state.replication_coordinator_rank,
            queue=checkpoint_io.queue,
        )

    @patch("sagemaker_nemo_adaptor.utils.callbacks.local_ckpt_io.load")
    @patch("sagemaker_nemo_adaptor.utils.callbacks.local_ckpt_io.DistributedFileSystemReader")
    @patch(
        "sagemaker_nemo_adaptor.utils.callbacks.base_ckpt_io.SageMakerBaseCheckpointIO.load_data_module_and_lr_schedulers"
    )
    def test_load_checkpoint(
        self, mock_load_data_module_and_lr_schedulers, mock_reader, mock_load, checkpoint_io, app_state
    ):
        path = "path/to/checkpoint"
        trainer = MagicMock(spec=pl.Trainer)
        trainer._checkpoint_connector = MagicMock(spec=_CheckpointConnector)
        state_dict = {"state_dict": {"a": 1, "b": 2}}
        trainer._checkpoint_connector.dump_checkpoint.return_value = state_dict

        checkpoint_io.load_checkpoint(path, trainer)

        mock_reader.assert_called_once_with(path)
        mock_load.assert_called_once_with(
            state_dict=trainer._checkpoint_connector.dump_checkpoint.return_value,
            process_group=app_state.current_replication_group,
            coordinator_rank=app_state.replication_coordinator_rank,
            storage_reader=mock_reader.return_value,
        )
        mock_load_data_module_and_lr_schedulers.assert_called_once_with(trainer, state_dict)

    def test_remove_checkpoint(self, checkpoint_io):
        with pytest.raises(NotImplementedError):
            checkpoint_io.remove_checkpoint("path/to/checkpoint")
