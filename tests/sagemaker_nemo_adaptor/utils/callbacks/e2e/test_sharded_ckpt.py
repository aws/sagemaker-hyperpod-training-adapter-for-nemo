import os

import pytest
import torch.distributed as dist
from nemo.utils import logging
from test_utils import TestCheckpoint
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu

from scripts.utils import enable_dummy_sm_env

enable_dummy_sm_env()  # Need to be called before torch sagemaker is imported

from sagemaker_nemo_adaptor.constants import SageMakerCheckpointType


@skip_if_lt_x_gpu(8)
class TestShardedCheckpoint(TestCheckpoint):

    def turn_on_sharded_only(self, config):
        # turn off auto checkpointing
        config.exp_manager.auto_checkpoint.enabled = False

        # turn on sharded checkpointing
        config.exp_manager.checkpoint_callback_params.save_last = True
        config.exp_manager.checkpoint_callback_params.save_top_k = 3
        config.exp_manager.checkpoint_callback_params.every_n_train_steps = 5

        # turn off full checkpointing
        config.exp_manager.export_full_model.every_n_train_steps = 0
        config.exp_manager.export_full_model.save_last = False

    @pytest.mark.parametrize("temp_dir", ["/tmp/test_sharded_save_and_load"], indirect=True)
    def test_sharded_save_and_load(self, temp_dir):
        # Config set up
        config = self.config()
        config.exp_manager.exp_dir = temp_dir
        config.exp_manager.checkpoint_dir = os.path.join(temp_dir, "checkpoints")
        self.turn_on_sharded_only(config)

        trainer, data_module, model_module = self.create_and_fit(config)
        trainer.strategy.checkpoint_io.checkpoint_type = SageMakerCheckpointType.SHARDED
        old_state_dict = trainer._checkpoint_connector.dump_checkpoint(weights_only=False)

        # Check saved checkpoint files.
        sharded_checkpoint_dir = os.path.join(config.exp_manager.checkpoint_dir, "sharded")
        assert os.path.exists(sharded_checkpoint_dir)
        checkpoint_callback_params = config.exp_manager.checkpoint_callback_params
        num_checkpoints_save = config.trainer.max_steps // checkpoint_callback_params.every_n_train_steps
        # Check if extra last step is saved.
        if checkpoint_callback_params.save_last:
            num_checkpoints_save += int(config.trainer.max_steps % checkpoint_callback_params.every_n_train_steps > 0)
        if num_checkpoints_save > checkpoint_callback_params.save_top_k:
            num_checkpoints_save = checkpoint_callback_params.save_top_k
        assert len(list(os.scandir(sharded_checkpoint_dir))) == num_checkpoints_save

        model_config = config.model
        tp_degree = model_config.get("tensor_model_parallel_degree", 1)
        ep_degree = model_config.get("expert_model_parallel_degree", 1)
        total_degree = tp_degree * ep_degree
        lastest_checkpoint = list(os.scandir(sharded_checkpoint_dir))[-1]
        assert len(list(os.scandir(lastest_checkpoint))) == total_degree

        del trainer, data_module, model_module

        # Create a new trainer and load the checkpoint
        config.exp_manager.resume_from_checkpoint = lastest_checkpoint.path
        logging.info("Creating a new trainer and loading the checkpoint")
        trainer, data_module, model_module = self.create_and_fit(config)
        trainer.strategy.checkpoint_io.checkpoint_type = SageMakerCheckpointType.SHARDED
        new_state_dict = trainer._checkpoint_connector.dump_checkpoint(weights_only=False)

        self.check_correctness(old_state_dict, new_state_dict, data_module.__class__.__qualname__)
        dist.barrier()
