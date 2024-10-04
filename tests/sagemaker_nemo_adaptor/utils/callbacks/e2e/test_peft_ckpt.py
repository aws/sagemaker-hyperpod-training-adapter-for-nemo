import os

import pytest
import torch
import torch.distributed as dist
from nemo.utils import logging
from test_utils import TestCheckpoint, assert_state_dict_equal, create_temp_dir
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu

from sagemaker_nemo_adaptor.utils.temp_utils import enable_dummy_sm_env

enable_dummy_sm_env()  # Need to be called before torch sagemaker is imported

from sagemaker_nemo_adaptor.constants import SageMakerCheckpointType


class TestPeftCheckpoint(TestCheckpoint):
    @pytest.fixture(scope="function", autouse=True)
    def cleanup_gpu_resources(self):
        """
        A pytest fixture that cleans up the GPU resources after each test.
        """
        yield

        logging.info(f"Cleaning up GPU resources")
        torch.cuda.empty_cache()

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

    def turn_on_full_only(self, config):
        # turn off auto checkpointing
        config.exp_manager.auto_checkpoint.enabled = False

        # turn off generic checkpointing
        config.exp_manager.checkpoint_callback_params.save_last = False
        config.exp_manager.checkpoint_callback_params.save_top_k = 0

        # turn on full checkpointing
        config.exp_manager.export_full_model.every_n_train_steps = 5
        config.exp_manager.export_full_model.save_last = True

    def run_full_save(self, temp_dir):
        """
        Helper method to save full checkpoint to be used as the base model for PEFT
        """
        # Config set up
        config = self.config()
        config.trainer.max_steps = 1
        config.exp_manager.exp_dir = temp_dir
        config.exp_manager.checkpoint_dir = os.path.join(temp_dir, "checkpoints")
        self.turn_on_full_only(config)

        trainer, data_module, model_module, _ = self.create_and_fit(config)
        trainer.strategy.checkpoint_io.checkpoint_type = SageMakerCheckpointType.FULL

        full_checkpoint_dir = os.path.join(config.exp_manager.checkpoint_dir, "full")
        assert os.path.exists(full_checkpoint_dir)
        all_saved_full_checkpoints = list(os.scandir(full_checkpoint_dir))

        latest_checkpoint = all_saved_full_checkpoints[-1]

        del trainer, data_module, model_module

        logging.info(f"saved model to {latest_checkpoint.path}")
        return latest_checkpoint.path


@skip_if_lt_x_gpu(8)
class TestPeftShardedCheckpoint(TestPeftCheckpoint):

    @create_temp_dir
    def test_lora_sharded_save_and_load(self, temp_dir):
        config = self.config(config_name="smp_llama_config_lora")
        self.run_peft_sharded_save_and_load(temp_dir, config)

    @create_temp_dir
    def test_qlora_sharded_save_and_load(self, temp_dir):
        config = self.config(config_name="smp_llama_config_lora")
        config.model.peft.peft_type = "qlora_4bit"
        self.run_peft_sharded_save_and_load(temp_dir, config)

    def run_peft_sharded_save_and_load(self, temp_dir, config):
        # Save full checkpoint to be used as the base model for PEFT
        pretrained_path = self.run_full_save(temp_dir)
        config.model.hf_model_name_or_path = pretrained_path
        # Config set up
        config.exp_manager.exp_dir = temp_dir
        config.exp_manager.checkpoint_dir = os.path.join(temp_dir, "checkpoints")
        self.turn_on_sharded_only(config)

        # Turn on fine tuning
        config.model.do_finetune = True

        trainer, data_module, model_module, _ = self.create_and_fit(config)
        trainer.strategy.checkpoint_io.checkpoint_type = SageMakerCheckpointType.PEFT_SHARDED
        old_state_dict = trainer._checkpoint_connector.dump_checkpoint(weights_only=False)
        old_model_weights = trainer.strategy.sharded_model_state_dict

        # Check saved checkpoint files.
        sharded_checkpoint_dir = os.path.join(config.exp_manager.checkpoint_dir, "peft_sharded")
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
        # Need to correct by 3 to account for the adapter weights files that will be saved
        assert len(list(os.scandir(lastest_checkpoint))) == total_degree + 3

        del trainer, data_module, model_module

        # Create a new trainer and load the checkpoint
        config.exp_manager.resume_from_checkpoint = lastest_checkpoint.path
        logging.info("Creating a new trainer and loading the checkpoint")
        trainer, data_module, model_module, _ = self.create_and_fit(config)
        trainer.strategy.checkpoint_io.checkpoint_type = SageMakerCheckpointType.PEFT_SHARDED
        new_state_dict = trainer._checkpoint_connector.dump_checkpoint(weights_only=False)
        new_model_weights = trainer.strategy.sharded_model_state_dict

        self.check_correctness(old_state_dict, new_state_dict, data_module.__class__.__qualname__)
        # Check model weights separately
        assert_state_dict_equal(old_model_weights, new_model_weights)

        del trainer, data_module, model_module
        dist.barrier()


@skip_if_lt_x_gpu(8)
class TestPeftFullCheckpoint(TestPeftCheckpoint):

    @create_temp_dir
    def test_lora_full_save_and_load(self, temp_dir):
        config = self.config(config_name="smp_llama_config_lora")
        self.run_peft_full_save_and_load(temp_dir, config)

    @create_temp_dir
    def test_qlora_full_save_and_load(self, temp_dir):
        config = self.config(config_name="smp_llama_config_lora")
        config.model.peft.peft_type = "qlora_4bit"
        self.run_peft_full_save_and_load(temp_dir, config)

    def run_peft_full_save_and_load(self, temp_dir, config):
        # Save full checkpoint to be used as the base model for PEFT
        pretrained_path = self.run_full_save(temp_dir)
        config.model.hf_model_name_or_path = pretrained_path

        # Config set up
        config.exp_manager.exp_dir = temp_dir
        config.exp_manager.checkpoint_dir = os.path.join(temp_dir, "checkpoints")
        self.turn_on_full_only(config)

        # Turn on fine tuning
        config.model.do_finetune = True

        trainer, data_module, model_module, _ = self.create_and_fit(config)
        trainer.strategy.checkpoint_io.checkpoint_type = SageMakerCheckpointType.PEFT_FULL

        full_checkpoint_dir = os.path.join(config.exp_manager.checkpoint_dir, "peft_full")
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
        # There should only be 3 files for the adapter weights + one final-model dir in one peft_full checkpoint dir.
        # adapter_config.json, adapter_model.safetensors, final-model, README.md
        assert len(all_files) == 4
        assert "adapter_config.json" in [v.name for v in all_files]
        assert "adapter_model.safetensors" in [v.name for v in all_files]
        assert "final-model" in [v.name for v in all_files]

        # Check files in the final-model dir
        # There should be a config.json, generation_config.json, and at least 1 .safetensors file
        final_model_dir = os.path.join(lastest_checkpoint, "final-model")
        all_final_model_files = list(os.scandir(final_model_dir))
        assert "config.json" in [v.name for v in all_final_model_files]
        assert "generation_config.json" in [v.name for v in all_final_model_files]
        assert any(".safetensors" in v.name for v in all_final_model_files)

        del trainer, data_module, model_module

        # Create a new trainer and load the checkpoint
        # A save full checkpoint can be only loaded through config.model.hf_model_name_or_path
        config.model.hf_model_name_or_path = final_model_dir
        config.model.do_finetune = True
        logging.info("Creating a new trainer and loading the checkpoint")
        trainer, data_module, model_module, _ = self.create_and_fit(config)
        # TODO: figure out how to check correctness of the fully merged model after loading

        del trainer, data_module, model_module
        dist.barrier()
