import os

import pytest
from test_utils import TestCheckpoint

from sagemaker_nemo_adaptor.utils.temp_utils import enable_dummy_sm_env

enable_dummy_sm_env()  # Need to be called before torch sagemaker is imported

from sagemaker_nemo_adaptor.collections.parts import SageMakerTrainerBuilder
from sagemaker_nemo_adaptor.utils.callbacks.checkpoint import (
    SageMakerCheckpoint,
    SageMakerCheckpointPeft,
    SageMakerModelCheckpointResilience,
)
from sagemaker_nemo_adaptor.utils.callbacks.ckpt_io import SageMakerCheckpointIO


class TestCheckpointCreation(TestCheckpoint):

    def update_checkpoint_config(self, config, checkpoint_param):
        """Update the checkpoint config to use the same model config as the training config."""
        save_top_k, sharded_save_last, auto_checkpoint, every_n_train_steps, save_full_last, peft_type = (
            checkpoint_param
        )
        # sharded
        config.exp_manager.checkpoint_callback_params.save_top_k = save_top_k
        config.exp_manager.checkpoint_callback_params.save_last = sharded_save_last

        # resilience
        config.exp_manager.auto_checkpoint.enabled = auto_checkpoint

        # full
        config.exp_manager.export_full_model.every_n_train_steps = every_n_train_steps
        config.exp_manager.export_full_model.save_last = save_full_last

        # peft
        config.model.peft.peft_type = peft_type

        return config

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
    @pytest.mark.parametrize("temp_dir", ["/tmp/test_callback_io_creation"], indirect=True)
    def test_callback_io_creation(
        self, save_top_k, sharded_save_last, auto_checkpoint, every_n_train_steps, save_full_last, peft_type, temp_dir
    ):
        """Test that the callback io creation works as expected."""
        config = self.config()
        config.exp_manager.exp_dir = temp_dir
        config.exp_manager.checkpoint_dir = os.path.join(temp_dir, "checkpoints")
        config = self.update_checkpoint_config(
            config, (save_top_k, sharded_save_last, auto_checkpoint, every_n_train_steps, save_full_last, peft_type)
        )
        trainer, _ = SageMakerTrainerBuilder(config).create_trainer()

        is_sharded = save_top_k > 0 or sharded_save_last
        is_resilience = auto_checkpoint
        is_full = every_n_train_steps > 0 or save_full_last
        is_peft = peft_type != None

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
