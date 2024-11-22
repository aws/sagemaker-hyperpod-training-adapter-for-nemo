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
from unittest.mock import patch

import pytest
import torch.distributed as dist
from nemo.utils import logging
from test_utils import TestCheckpoint, assert_state_dict_equal, skip_if_lt_x_gpu

from hyperpod_nemo_adapter.constants import SageMakerCheckpointType


class TestFullCheckpoint(TestCheckpoint):

    @skip_if_lt_x_gpu(8)
    @pytest.mark.parametrize(
        "model_type",
        [("llama"), ("mistral"), ("mixtral")],
    )
    def test_full_save_and_load(self, temp_dir, model_type):
        ports = self.find_free_network_ports()
        ports = self.broadcast_ports(ports)
        # Config set up
        config = self.config(model_type=model_type)
        config.exp_manager.exp_dir = temp_dir
        config.exp_manager.checkpoint_dir = os.path.join(temp_dir, "checkpoints")
        self.update_checkpoint_config_with_type(config, SageMakerCheckpointType.FULL)

        sample = self.generate_sample(config)
        self.reset_state_and_groups(ports[0])
        trainer, data_module, model_module, old_outputs = self.create_and_fit(
            config, model_type=model_type, sample=sample
        )
        old_state_dict = self.retrieve_state_dicts(trainer, checkpoint_types=[SageMakerCheckpointType.FULL])[0]

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

        self.reset_state_and_groups(ports[1])
        # Create a new trainer and load the checkpoint
        # A save full checkpoint can be only loaded through config.model.hf_model_name_or_path
        # Since in full mode, we don't save global step after reaching max_steps, we will need
        # to set the max_steps to 0. Otherwise, it will train from scratch and weights will change.
        config.model.hf_model_name_or_path = lastest_checkpoint.path
        config.model.do_finetune = True
        config.trainer.max_steps = 0
        config.model.optim.sched.warmup_steps = 1
        logging.info("Creating a new trainer and loading the checkpoint")
        trainer, data_module, model_module, new_outputs = self.create_and_fit(
            config, model_type=model_type, sample=sample
        )

        new_state_dict = self.retrieve_state_dicts(trainer, checkpoint_types=[SageMakerCheckpointType.FULL])[0]

        self.check_correctness(old_state_dict, new_state_dict, data_module_key="", is_full=True)

        assert_state_dict_equal(old_outputs, new_outputs)
        dist.barrier()

    @skip_if_lt_x_gpu(8)
    @pytest.mark.parametrize(
        "save_full_last, every_n_train_steps",
        [
            (False, 2),
            (True, 2),
        ],
    )
    def test_full_max_save(self, save_full_last, every_n_train_steps, temp_dir):
        # Config set up
        ports = self.find_free_network_ports()
        ports = self.broadcast_ports(ports)
        self.reset_state_and_groups(ports[0])
        config = self.config()
        config.exp_manager.exp_dir = temp_dir
        config.exp_manager.checkpoint_dir = os.path.join(temp_dir, "checkpoints")
        self.update_checkpoint_config_with_type(config, SageMakerCheckpointType.FULL)
        max_steps = 7
        config.trainer.max_steps = max_steps

        config.exp_manager.export_full_model.every_n_train_steps = every_n_train_steps
        config.exp_manager.export_full_model.save_last = save_full_last

        self.create_and_fit(
            config,
        )

        full_checkpoint_dir = os.path.join(config.exp_manager.checkpoint_dir, "full")
        assert os.path.exists(full_checkpoint_dir)
        num_checkpoints_save = config.trainer.max_steps // every_n_train_steps
        # Check if extra last step is saved.
        if save_full_last:
            num_checkpoints_save += int(max_steps % every_n_train_steps > 0)
        assert len(list(os.scandir(full_checkpoint_dir))) == num_checkpoints_save
        dist.barrier()

    @skip_if_lt_x_gpu(8)
    @pytest.mark.parametrize(
        "max_steps, save_full_last, every_n_train_steps",
        [
            (7, True, 2),
            (7, False, 2),
            (6, True, 3),
        ],
    )
    @patch("hyperpod_nemo_adapter.utils.callbacks.checkpoint.SageMakerModelCheckpointBase._save")
    def test_full_save_calls(self, mock_save, max_steps, save_full_last, every_n_train_steps, temp_dir):
        # Config set up
        ports = self.find_free_network_ports()
        ports = self.broadcast_ports(ports)
        self.reset_state_and_groups(ports[0])
        config = self.config()
        config.exp_manager.exp_dir = temp_dir
        config.exp_manager.checkpoint_dir = os.path.join(temp_dir, "checkpoints")
        self.update_checkpoint_config_with_type(config, SageMakerCheckpointType.FULL)

        config.trainer.max_steps = max_steps
        config.exp_manager.export_full_model.every_n_train_steps = every_n_train_steps
        config.exp_manager.export_full_model.save_last = save_full_last

        expected_call_counts = int(config.trainer.max_steps / every_n_train_steps)
        if max_steps % every_n_train_steps != 0 and save_full_last:
            expected_call_counts += 1
        self.create_and_fit(config)
        assert mock_save.call_count == expected_call_counts

        dist.barrier()
