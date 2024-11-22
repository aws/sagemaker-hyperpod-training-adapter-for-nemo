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


class TestShardedCheckpoint(TestCheckpoint):

    @skip_if_lt_x_gpu(8)
    @pytest.mark.parametrize(
        "model_type",
        [("llama"), ("mistral"), ("mixtral")],
    )
    def test_sharded_save_and_load(self, temp_dir, model_type):
        # Config set up
        ports = self.find_free_network_ports()
        ports = self.broadcast_ports(ports)
        config = self.config(model_type=model_type)
        config.exp_manager.exp_dir = temp_dir
        config.exp_manager.checkpoint_dir = os.path.join(temp_dir, "checkpoints")
        self.update_checkpoint_config_with_type(config, SageMakerCheckpointType.SHARDED)

        sample = self.generate_sample(config)

        self.reset_state_and_groups(ports[0])
        trainer, data_module, model_module, old_outputs = self.create_and_fit(
            config, model_type=model_type, sample=sample
        )
        old_state_dict = self.retrieve_state_dicts(trainer, checkpoint_types=[SageMakerCheckpointType.SHARDED])[0]

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

        self.reset_state_and_groups(ports[1])
        # Create a new trainer and load the checkpoint
        config.exp_manager.resume_from_checkpoint = lastest_checkpoint.path
        logging.info("Creating a new trainer and loading the checkpoint")
        trainer, data_module, model_module, new_outputs = self.create_and_fit(
            config, model_type=model_type, sample=sample
        )
        new_state_dict = self.retrieve_state_dicts(trainer, checkpoint_types=[SageMakerCheckpointType.SHARDED])[0]

        self.check_correctness(old_state_dict, new_state_dict, data_module.__class__.__qualname__)
        assert_state_dict_equal(old_outputs, new_outputs)
        dist.barrier()

    @skip_if_lt_x_gpu(8)
    @pytest.mark.parametrize(
        "save_top_k, sharded_save_last, every_n_train_steps",
        [
            (2, False, 2),
            (2, True, 2),
        ],
    )
    def test_sharded_max_save(self, save_top_k, sharded_save_last, every_n_train_steps, temp_dir):
        # Config set up
        ports = self.find_free_network_ports()
        ports = self.broadcast_ports(ports)
        self.reset_state_and_groups(ports[0])
        config = self.config()
        config.exp_manager.exp_dir = temp_dir
        config.exp_manager.checkpoint_dir = os.path.join(temp_dir, "checkpoints")
        self.update_checkpoint_config_with_type(config, SageMakerCheckpointType.SHARDED)
        max_steps = 7
        config.trainer.max_steps = max_steps

        config.exp_manager.checkpoint_callback_params.save_top_k = save_top_k
        config.exp_manager.checkpoint_callback_params.save_last = sharded_save_last
        config.exp_manager.checkpoint_callback_params.every_n_train_steps = every_n_train_steps

        self.create_and_fit(
            config,
        )

        sharded_checkpoint_dir = os.path.join(config.exp_manager.checkpoint_dir, "sharded")
        assert os.path.exists(sharded_checkpoint_dir)
        num_checkpoints_save = max_steps // every_n_train_steps
        if num_checkpoints_save > save_top_k:
            num_checkpoints_save = save_top_k
        # Check if extra last step is saved.
        if sharded_save_last:
            num_checkpoints_save += int(max_steps % every_n_train_steps > 0)
        assert len(list(os.scandir(sharded_checkpoint_dir))) == num_checkpoints_save
        dist.barrier()

    @skip_if_lt_x_gpu(8)
    @pytest.mark.parametrize(
        "max_steps, sharded_save_last, every_n_train_steps",
        [
            (7, True, 2),
            (7, False, 2),
            (6, True, 3),
            (2, False, 0),
        ],
    )
    @patch("hyperpod_nemo_adapter.utils.callbacks.checkpoint.SageMakerModelCheckpointBase._save")
    def test_sharded_save_calls(self, mock_save, max_steps, sharded_save_last, every_n_train_steps, temp_dir):
        # Config set up
        ports = self.find_free_network_ports()
        ports = self.broadcast_ports(ports)
        self.reset_state_and_groups(ports[0])
        config = self.config()
        config.exp_manager.exp_dir = temp_dir
        config.exp_manager.checkpoint_dir = os.path.join(temp_dir, "checkpoints")
        self.update_checkpoint_config_with_type(config, SageMakerCheckpointType.SHARDED)

        config.trainer.max_steps = max_steps
        config.exp_manager.checkpoint_callback_params.save_last = sharded_save_last
        config.exp_manager.checkpoint_callback_params.every_n_train_steps = every_n_train_steps

        if every_n_train_steps != 0:
            expected_call_counts = int(config.trainer.max_steps / every_n_train_steps)
            if max_steps % every_n_train_steps != 0 and sharded_save_last:
                expected_call_counts += 1
        else:
            expected_call_counts = 0
        self.create_and_fit(config)
        assert mock_save.call_count == expected_call_counts
        dist.barrier()
