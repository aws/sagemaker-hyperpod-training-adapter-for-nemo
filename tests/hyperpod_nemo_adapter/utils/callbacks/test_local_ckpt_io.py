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

from unittest.mock import MagicMock

import pytest
import pytorch_lightning as pl
from lightning.pytorch.trainer.connectors.checkpoint_connector import (
    _CheckpointConnector,
)

from hyperpod_nemo_adapter.utils.app_state import SageMakerAppState
from hyperpod_nemo_adapter.utils.callbacks.local_ckpt_io import (
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
