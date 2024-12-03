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

import pytest
import torch
import torch.distributed as dist
from nemo.utils import logging
from test_utils import TestCheckpoint, assert_state_dict_equal, skip_if_lt_x_gpu

from hyperpod_nemo_adapter.constants import SageMakerCheckpointType


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
        config = self.config(model_type="llama")
        config.trainer.max_steps = 1
        config.exp_manager.exp_dir = temp_dir
        config.exp_manager.checkpoint_dir = os.path.join(temp_dir, "checkpoints")
        self.turn_on_full_only(config)

        trainer, data_module, model_module, _ = self.create_and_fit(config, model_type="llama")
        trainer.strategy.checkpoint_io.checkpoint_type = SageMakerCheckpointType.FULL

        full_checkpoint_dir = os.path.join(config.exp_manager.checkpoint_dir, "full")
        assert os.path.exists(full_checkpoint_dir)
        all_saved_full_checkpoints = list(os.scandir(full_checkpoint_dir))

        latest_checkpoint = all_saved_full_checkpoints[-1]

        del trainer, data_module, model_module

        logging.info(f"saved model to {latest_checkpoint.path}")
        return latest_checkpoint.path


class TestPeftShardedCheckpoint(TestPeftCheckpoint):

    @skip_if_lt_x_gpu(8)
    @pytest.mark.parametrize(
        "peft_type",
        [("lora"), ("qlora_4bit")],
    )
    def test_peft_sharded_save_and_load(self, temp_dir, peft_type):
        ports = self.find_free_network_ports()
        ports = self.broadcast_ports(ports)
        # Save full checkpoint to be used as the base model for PEFT
        pretrained_path = self.run_full_save(temp_dir)
        # Config set up
        config = self.config(model_type="llama_lora")
        config.model.hf_model_name_or_path = pretrained_path
        config.model.peft.peft_type = peft_type
        config.exp_manager.exp_dir = temp_dir
        config.exp_manager.checkpoint_dir = os.path.join(temp_dir, "checkpoints")
        self.turn_on_sharded_only(config)

        # Turn on fine tuning
        config.model.do_finetune = True

        self.reset_state_and_groups(ports[0])
        trainer, data_module, model_module, _ = self.create_and_fit(config, model_type="llama_lora")
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

        self.reset_state_and_groups(ports[1])
        # Create a new trainer and load the checkpoint
        config.exp_manager.resume_from_checkpoint = lastest_checkpoint.path
        logging.info("Creating a new trainer and loading the checkpoint")
        trainer, data_module, model_module, _ = self.create_and_fit(config, model_type="llama_lora")
        trainer.strategy.checkpoint_io.checkpoint_type = SageMakerCheckpointType.PEFT_SHARDED
        new_state_dict = trainer._checkpoint_connector.dump_checkpoint(weights_only=False)
        new_model_weights = trainer.strategy.sharded_model_state_dict

        self.check_correctness(old_state_dict, new_state_dict, data_module.__class__.__qualname__)
        # Check model weights separately
        assert_state_dict_equal(old_model_weights, new_model_weights)

        del trainer, data_module, model_module
        dist.barrier()


class TestPeftFullCheckpoint(TestPeftCheckpoint):

    @skip_if_lt_x_gpu(8)
    @pytest.mark.parametrize(
        "peft_type",
        [("lora"), ("qlora_4bit")],
    )
    def test_peft_full_save_and_load(self, temp_dir, peft_type):
        ports = self.find_free_network_ports()
        ports = self.broadcast_ports(ports)
        # Save full checkpoint to be used as the base model for PEFT
        pretrained_path = self.run_full_save(temp_dir)

        # Config set up
        config = self.config(model_type="llama_lora")
        config.model.hf_model_name_or_path = pretrained_path
        config.model.peft.peft_type = peft_type
        config.exp_manager.exp_dir = temp_dir
        config.exp_manager.checkpoint_dir = os.path.join(temp_dir, "checkpoints")
        self.turn_on_full_only(config)

        # Turn on fine tuning
        config.model.do_finetune = True

        self.reset_state_and_groups(ports[0])
        trainer, data_module, model_module, _ = self.create_and_fit(config, model_type="llama_lora")
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

        latest_checkpoint = all_saved_full_checkpoints[-1]
        all_files = list(os.scandir(latest_checkpoint))
        # There should only be 3 files for the adapter weights + one final-model dir in one peft_full checkpoint dir.
        # adapter_config.json, adapter_model.safetensors, final-model, README.md
        assert len(all_files) == 4
        assert "adapter_config.json" in [v.name for v in all_files]
        assert "adapter_model.safetensors" in [v.name for v in all_files]
        assert "final-model" in [v.name for v in all_files]

        # Check files in the final-model dir
        # There should be a config.json, generation_config.json, and at least 1 .safetensors file
        final_model_dir = os.path.join(latest_checkpoint, "final-model")
        all_final_model_files = list(os.scandir(final_model_dir))
        assert "config.json" in [v.name for v in all_final_model_files]
        assert "generation_config.json" in [v.name for v in all_final_model_files]
        assert any(".safetensors" in v.name for v in all_final_model_files)

        # Save the fully merged model state_dict for comparison
        if dist.get_rank() == 0:
            old_merged_model = trainer.strategy.checkpoint_io._checkpoint_io[
                SageMakerCheckpointType.PEFT_FULL
            ]._merge_and_upload_peft_model(trainer, latest_checkpoint, upload_to_storage=False)
            old_state_dict = old_merged_model.state_dict()
        else:
            # full state_dict only gets loaded onto rank 0
            old_state_dict = {}

        del trainer, data_module, model_module

        self.reset_state_and_groups(ports[1])
        # Create a new trainer and load the full checkpoint
        # Set up non peft config to load in the full checkpoint
        config.model.peft.peft_type = None
        # A save full checkpoint can be only loaded through config.model.hf_model_name_or_path
        config.model.hf_model_name_or_path = final_model_dir
        config.model.do_finetune = True
        config.trainer.max_steps = 0
        config.model.optim.sched.warmup_steps = 1
        # disable final checkpoint
        config.exp_manager.export_full_model.save_last = False
        logging.info("Creating a new trainer and loading the checkpoint")
        trainer, data_module, model_module, _ = self.create_and_fit(config, model_type="llama_lora")

        trainer.strategy.checkpoint_io.checkpoint_type = SageMakerCheckpointType.FULL
        new_state_dict = trainer._checkpoint_connector.dump_checkpoint(weights_only=True)
        new_state_dict = new_state_dict["state_dict"]

        assert_state_dict_equal(new_state_dict, old_state_dict)

        del trainer, data_module, model_module
        dist.barrier()
