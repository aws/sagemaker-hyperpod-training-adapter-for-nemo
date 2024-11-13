import os

import pytest
import torch.distributed as dist
from nemo.utils import logging
from test_utils import TestCheckpoint, assert_state_dict_equal, skip_if_lt_x_gpu

from sagemaker_nemo_adaptor.constants import SageMakerCheckpointType


class TestLRSchedulder(TestCheckpoint):

    def retrieve_lrs(self, trainer):
        # get the last computed lr and next step lr.
        lrs = []
        next_lrs = []
        for cfg in trainer.lr_scheduler_configs:
            lrs.append(cfg.scheduler.get_last_lr())
            cfg.scheduler.step()
            next_lrs.append(cfg.scheduler.get_last_lr())
        return lrs, next_lrs

    @skip_if_lt_x_gpu(8)
    @pytest.mark.parametrize(
        # Skip full checkpoint type as in full mode, only weights are saved.
        "checkpoint_type",
        [
            SageMakerCheckpointType.SHARDED,
            SageMakerCheckpointType.LOCAL,
        ],
    )
    def test_lr_scheduler_save_load(self, checkpoint_type, temp_dir):
        """
        Test lr scheduler save and load.
        lr scheduler latest lr will be retrieved and next step lr will be computed.
        """
        # Config set up
        ports = self.find_free_network_ports()
        ports = self.broadcast_ports(ports)
        self.reset_state_and_groups(ports[0])
        config = self.config()
        config.exp_manager.exp_dir = temp_dir
        config.exp_manager.checkpoint_dir = os.path.join(temp_dir, "checkpoints")
        self.update_checkpoint_config_with_type(config, checkpoint_type)

        trainer, data_module, model_module, old_outputs = self.create_and_fit(config)
        # get the lrs after training.
        old_lrs, old_next_lrs = self.retrieve_lrs(trainer)

        del trainer, data_module, model_module

        # Create a new trainer and load the checkpoint
        self.reset_state_and_groups(ports[1])
        logging.info("Creating a new trainer and loading the checkpoint")
        if checkpoint_type == SageMakerCheckpointType.SHARDED:
            sharded_checkpoint_dir = os.path.join(config.exp_manager.checkpoint_dir, "sharded")
            lastest_checkpoint = list(os.scandir(sharded_checkpoint_dir))[-1]
            config.exp_manager.resume_from_checkpoint = lastest_checkpoint.path
        trainer, data_module, model_module, new_outputs = self.create_and_fit(config)

        # get the lrs after loading.
        new_lrs, new_next_lrs = self.retrieve_lrs(trainer)
        assert_state_dict_equal(old_lrs, new_lrs)
        assert_state_dict_equal(old_next_lrs, new_next_lrs)
        dist.barrier()
