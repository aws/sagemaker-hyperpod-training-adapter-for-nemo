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
class TestResilienceCheckpoint(TestCheckpoint):

    def turn_on_resilience_only(self, config):
        config.exp_manager.auto_checkpoint.enabled = True
        config.exp_manager.auto_checkpoint.warmup_steps = 0
        config.exp_manager.auto_checkpoint.drop_n_warmup_steps = 0
        config.exp_manager.auto_checkpoint.interval_guard = 1.25
        # turn off sharded checkpointing
        config.exp_manager.checkpoint_callback_params.save_last = False
        config.exp_manager.checkpoint_callback_params.save_top_k = 0

        # turn off full checkpointing
        config.exp_manager.export_full_model.every_n_train_steps = 0
        config.exp_manager.export_full_model.save_last = False

    @pytest.mark.parametrize("temp_dir", ["/tmp/test_resilience_save_and_load"], indirect=True)
    def test_resilience_save_and_load(self, temp_dir):
        # Config set up
        config = self.config()
        config.exp_manager.exp_dir = temp_dir
        config.exp_manager.checkpoint_dir = os.path.join(temp_dir, "checkpoints")
        self.turn_on_resilience_only(config)

        trainer, data_module, model_module = self.create_and_fit(config)
        trainer.strategy.checkpoint_io.checkpoint_type = SageMakerCheckpointType.LOCAL
        old_state_dict = trainer._checkpoint_connector.dump_checkpoint(weights_only=False)

        # Check saved checkpoint files.
        assert os.path.exists(os.path.join(config.exp_manager.checkpoint_dir, "local"))
        model_config = config.model
        fsdp_degree = model_config.get("shard_degree", 1)
        tp_degree = model_config.get("tensor_model_parallel_degree", 1)
        ep_degree = model_config.get("expert_model_parallel_degree", 1)
        total_degree = fsdp_degree * tp_degree * ep_degree
        assert len(list(os.scandir(os.path.join(config.exp_manager.checkpoint_dir, "local", "0")))) == total_degree

        del trainer, data_module, model_module

        # Create a new trainer and load the checkpoint
        # No checkpoint path needs to be set during loading, as it should auto resume.
        logging.info("Creating a new trainer and loading the checkpoint")
        trainer, data_module, model_module = self.create_and_fit(config)
        trainer.strategy.checkpoint_io.checkpoint_type = SageMakerCheckpointType.LOCAL
        new_state_dict = trainer._checkpoint_connector.dump_checkpoint(weights_only=False)

        self.check_correctness(old_state_dict, new_state_dict, data_module.__class__.__qualname__)
        dist.barrier()
