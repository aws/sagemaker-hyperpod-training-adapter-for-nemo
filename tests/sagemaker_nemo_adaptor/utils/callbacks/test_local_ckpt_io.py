from unittest.mock import MagicMock

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

    def test_remove_checkpoint(self, checkpoint_io):
        with pytest.raises(NotImplementedError):
            checkpoint_io.remove_checkpoint("path/to/checkpoint")
