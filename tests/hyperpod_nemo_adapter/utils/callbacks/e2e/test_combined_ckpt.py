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

import torch.distributed as dist
from nemo.utils import logging
from test_resilience_ckpt import StateDictRetriever
from test_utils import (
    TestCheckpoint,
    assert_state_dict_equal,
    assert_values,
    skip_if_lt_x_gpu,
)
from torch.sagemaker.distributed.checkpoint.filesystem import (
    DistributedFileSystemWriter,
)

from hyperpod_nemo_adapter.constants import SageMakerCheckpointType
from hyperpod_nemo_adapter.utils.callbacks.local_ckpt_io import (
    SageMakerLocalCheckpointIO,
)


class TestCombinedCheckpoint(TestCheckpoint):

    @skip_if_lt_x_gpu(8)
    def test_combined_save_and_load(self, temp_dir):
        # Config set up
        ports = self.find_free_network_ports()
        ports = self.broadcast_ports(ports)
        self.reset_state_and_groups(ports[0])
        config = self.config()
        config.exp_manager.exp_dir = temp_dir
        config.exp_manager.checkpoint_dir = os.path.join(temp_dir, "checkpoints")
        # train total with 3 steps, with step 2 state dict retreieved.
        # sharded will saved @ step 2,
        # local will save @ every step
        # full will save @ step 3
        # Turn on all checkpoints.
        # save_top_k, sharded_save_last, auto_checkpoint, every_n_train_steps, save_full_last, peft_type
        self.update_checkpoint_config(config, (0, False, True, 5, True, None))
        config.exp_manager.checkpoint_callback_params.every_n_train_steps = 2
        config.trainer.max_steps = 3

        state_dict_retriever = StateDictRetriever(retrieve_step=3, checkpoint_type=SageMakerCheckpointType.SHARDED)
        trainer, data_module, model_module, old_outputs = self.create_and_fit(config, callbacks=[state_dict_retriever])

        # Check LOCAL saved checkpoint files.
        assert os.path.exists(os.path.join(config.exp_manager.checkpoint_dir, "local"))
        model_config = config.model
        fsdp_degree = model_config.get("shard_degree", 1)
        tp_degree = model_config.get("tensor_model_parallel_degree", 1)
        ep_degree = model_config.get("expert_model_parallel_degree", 1)
        total_degree = fsdp_degree * tp_degree * ep_degree
        assert len(list(os.scandir(os.path.join(config.exp_manager.checkpoint_dir, "local", "0")))) == total_degree

        # Check FULL saved checkpoint files.
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

        # Stop training and further
        self.reset_state_and_groups(ports[1])
        logging.info("Creating a new trainer and loading the checkpoint")
        trainer, data_module, model_module, new_outputs = self.create_and_fit(config)
        new_state_dict = self.retrieve_state_dicts(trainer, checkpoint_types=[SageMakerCheckpointType.SHARDED])[0]

        self.check_correctness(
            new_state_dict, state_dict_retriever.retrieve_state_dict, data_module.__class__.__qualname__
        )
        dist.barrier()
