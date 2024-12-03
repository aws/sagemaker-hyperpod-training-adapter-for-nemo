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
from contextlib import nullcontext

import pytest
from test_utils import TestCheckpoint

from hyperpod_nemo_adapter.collections.parts import SageMakerTrainerBuilder
from hyperpod_nemo_adapter.utils.callbacks.checkpoint import (
    SageMakerCheckpoint,
    SageMakerCheckpointPeft,
    SageMakerModelCheckpointResilience,
)
from hyperpod_nemo_adapter.utils.callbacks.ckpt_io import SageMakerCheckpointIO


class TestCheckpointCreation(TestCheckpoint):

    @pytest.mark.parametrize(
        "save_top_k, sharded_save_last, auto_checkpoint, every_n_train_steps, save_full_last, peft_type",
        [  # all off
            (0, False, False, 0, False, None),
            # one on
            (10, False, False, 0, False, None),
            (0, True, False, 0, False, None),
            (0, False, True, 0, False, None),
            (0, False, False, 10, False, None),
            (0, False, False, 0, True, None),
            # two on
            # sharded + resilience
            (10, True, True, 0, False, None),
            # sharded + full
            (10, True, False, 10, True, None),
            # resilience + full
            (0, False, True, 10, True, None),
            # all on
            (10, True, True, 0, True, None),
            # if peft, then all other will be off.
            (10, True, True, 0, True, "lora"),
        ],
    )
    def test_callback_io_creation(
        self, save_top_k, sharded_save_last, auto_checkpoint, every_n_train_steps, save_full_last, peft_type, tmp_path
    ):
        """Test that the callback io creation works as expected."""
        config = self.config()
        config.exp_manager.exp_dir = tmp_path
        config.exp_manager.checkpoint_dir = os.path.join(tmp_path, "checkpoints")
        config = self.update_checkpoint_config(
            config, (save_top_k, sharded_save_last, auto_checkpoint, every_n_train_steps, save_full_last, peft_type)
        )
        is_sharded = save_top_k > 0 or sharded_save_last
        is_resilience = auto_checkpoint
        is_full = every_n_train_steps > 0 or save_full_last
        is_peft = peft_type != None
        should_raise_error = is_resilience and is_sharded
        context = nullcontext() if not should_raise_error else pytest.raises(AssertionError)

        with context:
            trainer, _ = SageMakerTrainerBuilder(config).create_trainer()

        if should_raise_error:
            return

        # test checkpoint callbacks
        if is_peft:
            assert len(trainer.checkpoint_callbacks) == 1
            assert isinstance(trainer.checkpoint_callbacks[0], SageMakerCheckpointPeft)
        else:
            assert len(trainer.checkpoint_callbacks) == sum([is_resilience, is_full or is_sharded])
            if is_resilience:
                assert isinstance(trainer.checkpoint_callbacks[0], SageMakerModelCheckpointResilience)
            elif is_sharded or is_full:
                assert isinstance(trainer.checkpoint_callbacks[0], SageMakerCheckpoint)

        # test checkpoint IO.
        if is_resilience or is_full or is_sharded:
            assert isinstance(trainer.strategy.checkpoint_io, SageMakerCheckpointIO)
