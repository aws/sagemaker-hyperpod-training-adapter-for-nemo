import os

import torch.distributed as dist
from nemo.utils import logging
from test_utils import TestCheckpoint, skip_if_lt_x_gpu

from sagemaker_nemo_adaptor.utils.temp_utils import enable_dummy_sm_env

enable_dummy_sm_env()  # Need to be called before torch sagemaker is imported

from sagemaker_nemo_adaptor.constants import SageMakerCheckpointType


class TestFullCheckpoint(TestCheckpoint):

    def turn_on_full_only(self, config):
        # turn off auto checkpointing
        config.exp_manager.auto_checkpoint.enabled = False

        # turn on generic checkpointing
        config.exp_manager.checkpoint_callback_params.save_last = False
        config.exp_manager.checkpoint_callback_params.save_top_k = 0

        # turn off full checkpointing
        config.exp_manager.export_full_model.every_n_train_steps = 5
        config.exp_manager.export_full_model.save_last = True

    @skip_if_lt_x_gpu(8)
    def test_full_save_and_load(self, temp_dir):
        # Config set up
        config = self.config()
        config.exp_manager.exp_dir = temp_dir
        config.exp_manager.checkpoint_dir = os.path.join(temp_dir, "checkpoints")
        self.turn_on_full_only(config)

        trainer, data_module, model_module = self.create_and_fit(config)
        trainer.strategy.checkpoint_io.checkpoint_type = SageMakerCheckpointType.FULL
        old_state_dict = trainer._checkpoint_connector.dump_checkpoint(weights_only=True)

        full_checkpoint_dir = os.path.join(config.exp_manager.checkpoint_dir, "full")
        assert os.path.exists(full_checkpoint_dir)
        export_full_model = config.exp_manager.export_full_model
        num_checkpoints_save = config.trainer.max_steps // export_full_model.every_n_train_steps
        # Check if extra last step is saved.
        if export_full_model.save_last:
            num_checkpoints_save += int(config.trainer.max_steps % export_full_model.every_n_train_steps > 0)
        all_saved_full_checkpoints = list(os.scandir(full_checkpoint_dir))
        assert len(all_saved_full_checkpoints) == num_checkpoints_save

        lastest_checkpoint = all_saved_full_checkpoints[-1]
        all_files = list(os.scandir(lastest_checkpoint))
        # There should only be 2 files in one full checkpoint dir.
        # one is pytorch_model.bin, the other is config.json
        assert len(all_files) == 2
        assert "pytorch_model.bin" in [v.name for v in all_files]
        assert "config.json" in [v.name for v in all_files]

        del trainer, data_module, model_module

        # Create a new trainer and load the checkpoint
        # A save full checkpoint can be only loaded through config.model.hf_model_name_or_path
        # Since in full mode, we don't save global step after reaching max_steps, we will need        # to set the max_steps to 0. Otherwise, it will train from scratch and weights will change.
        config.model.hf_model_name_or_path = lastest_checkpoint.path
        config.model.do_finetune = True
        config.trainer.max_steps = 0
        logging.info("Creating a new trainer and loading the checkpoint")
        trainer, data_module, model_module = self.create_and_fit(config)

        trainer.strategy.checkpoint_io.checkpoint_type = SageMakerCheckpointType.FULL
        new_state_dict = trainer._checkpoint_connector.dump_checkpoint(weights_only=True)
        self.check_correctness(old_state_dict, new_state_dict, data_module_key="", is_full=True)
        dist.barrier()
