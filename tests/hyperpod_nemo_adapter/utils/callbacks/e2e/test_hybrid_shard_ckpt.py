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
import time

import pytest
import torch.distributed as dist
from test_utils import (
    _WORLD_SIZE,
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


class TestHybridShardCheckpoint(TestCheckpoint):

    def set_hybrid_degree(self, config, tp=1, ep=1, fsdp=8, world_size=_WORLD_SIZE):
        config.model.shard_degree = fsdp
        config.model.tensor_model_parallel_degree = tp
        config.model.expert_model_parallel_degree = ep
        config.trainer.devices = list(range(world_size))

    def run_hybrid_test(
        self, model_type, checkpoint_type, tp1, fsdp1, ep1, world_size1, tp2, fsdp2, ep2, world_size2, tmp_dir
    ):
        """Test for resilience hybrid shard in llama.

        It will:
            1. training with shard degrees (tp1, fsdp1, world_size1)
            2. load with shard degrees (tp2, fsdp2, world_size2)

        Note:
            1. Here we always start with full world size, so that shared tmp file are
            boradcasted to all ranks.
            2. We use different address and port for each trainer to avoid NCCL hang.
            3. TSM State needs to reset and process group needs to be destroyed(or TSM will always get the
                same world size).
            4. Make sure the rank that does not get involved in second training exit gracefully.
            5. The last trainer need to retrains full world size. ie: 8
        """
        ports = self.find_free_network_ports()
        ports = self.broadcast_ports(ports)
        self.reset_state_and_groups(ports[0])

        rank = int(os.environ.get("LOCAL_RANK", "0"))
        # During SHARDED mode, shard degree will change so we cannot retrieve the state_dict each rank
        # and compare. Instead, we will get the full state_dict as whole.
        retrieve_checpoint_type = (
            SageMakerCheckpointType.FULL if checkpoint_type == SageMakerCheckpointType.SHARDED else checkpoint_type
        )
        if rank < world_size1:
            config = self.config(model_type=model_type)
            self.set_hybrid_degree(config, tp=tp1, fsdp=fsdp1, ep=ep1, world_size=world_size1)
            config.exp_manager.exp_dir = tmp_dir
            config.exp_manager.checkpoint_dir = os.path.join(tmp_dir, "checkpoints")
            self.update_checkpoint_config_with_type(config, checkpoint_type)
            trainer, data_module, model_module, old_outputs = self.create_and_fit(config)
            old_state_dict = self.retrieve_state_dicts(trainer, checkpoint_types=[retrieve_checpoint_type])[0]

        self.reset_state_and_groups(ports[1])
        rank = int(os.environ.get("LOCAL_RANK", "0"))
        # Config set up
        if rank < world_size2:

            config = self.config(model_type=model_type)
            config.exp_manager.exp_dir = tmp_dir
            config.exp_manager.checkpoint_dir = os.path.join(tmp_dir, "checkpoints")
            if checkpoint_type == SageMakerCheckpointType.SHARDED:
                sharded_checkpoint_dir = os.path.join(config.exp_manager.checkpoint_dir, "sharded")
                while not os.path.exists(sharded_checkpoint_dir):
                    # Some ranks were not involved in first training, so we need to wait for them to finish
                    time.sleep(5)
                lastest_checkpoint = list(os.scandir(sharded_checkpoint_dir))[-1]
                config.exp_manager.resume_from_checkpoint = lastest_checkpoint.path
            self.set_hybrid_degree(config, tp=tp2, fsdp=fsdp2, ep=ep2, world_size=world_size2)
            self.update_checkpoint_config_with_type(config, checkpoint_type)

            trainer, data_module, model_module, old_outputs = self.create_and_fit(config)
            new_state_dict = self.retrieve_state_dicts(trainer, checkpoint_types=[retrieve_checpoint_type])[0]
            dist.barrier()
        if (rank < world_size1 and retrieve_checpoint_type != SageMakerCheckpointType.FULL) or rank == 0:
            self.check_correctness(old_state_dict, new_state_dict, data_module.__class__.__qualname__)

    @skip_if_lt_x_gpu(8)
    @pytest.mark.parametrize(
        "tp1, fsdp1, world_size1, tp2, fsdp2, world_size2",
        [  # test TP + FSDP + DP : (tp2,fsdp2,dp1) -> (tp2,fsdp2,dp2)
            (2, 2, 4, 2, 2, 8),
            # test FSDP + DP: (fsdp4,dp1) -> (fsdp4,dp2)
            (1, 4, 4, 1, 4, 8),
            # test TP + DP: (tp2,dp2) -> (tp2,dp4)
            # (2, 1, 4, 2, 1, 8),
        ],
    )
    def test_resilience_hybrid_llama(self, tp1, fsdp1, world_size1, tp2, fsdp2, world_size2, temp_dir):
        "Test resilience hybrid sharding on llama with fixed shard degree(tp + fsdp) but different dp."
        self.run_hybrid_test(
            model_type="llama",
            checkpoint_type=SageMakerCheckpointType.LOCAL,
            tp1=tp1,
            fsdp1=fsdp1,
            ep1=1,
            world_size1=world_size1,
            tp2=tp2,
            fsdp2=fsdp2,
            ep2=1,
            world_size2=world_size2,
            tmp_dir=temp_dir,
        )

    @skip_if_lt_x_gpu(8)
    def test_resilience_hybrid_shard_mixtral(self, temp_dir):
        "Test resilience hybrid sharding on mixtral with fixed shard degree(ep + fsdp) but different dp."
        # test EP + FSDP + DP : (ep2,fsdp2,dp1) ->(ep2,fsdp2,dp2)
        self.run_hybrid_test(
            model_type="mixtral",
            checkpoint_type=SageMakerCheckpointType.LOCAL,
            tp1=1,
            fsdp1=2,
            ep1=2,
            world_size1=4,
            tp2=1,
            fsdp2=2,
            ep2=2,
            world_size2=8,
            tmp_dir=temp_dir,
        )

    @skip_if_lt_x_gpu(8)
    @pytest.mark.parametrize(
        "tp1, fsdp1, world_size1, tp2, fsdp2, world_size2",
        [  # test FSDP + DP : (fsdp2, dp2) -> (fsdp4,dp2)
            (1, 2, 4, 1, 4, 8),
            # test FSDP + DP: (fsdp2,dp3) -> (fsdp4,dp2)
            (1, 2, 6, 1, 4, 8),
            # test TP + FSDP: (tp2,fsdp2) -> (tp2,fsdp4)
            (2, 2, 4, 2, 4, 8),
        ],
    )
    @skip_if_lt_x_gpu(8)
    def test_sharded_hybrid_shard_llama(self, tp1, fsdp1, world_size1, tp2, fsdp2, world_size2, temp_dir):
        "Test sharded hybrid sharding on llama with different shard degree(tp + fsdp) and dp."
        self.run_hybrid_test(
            model_type="llama",
            checkpoint_type=SageMakerCheckpointType.SHARDED,
            tp1=tp1,
            fsdp1=fsdp1,
            ep1=1,
            world_size1=world_size1,
            tp2=tp2,
            fsdp2=fsdp2,
            ep2=1,
            world_size2=world_size2,
            tmp_dir=temp_dir,
        )

    @skip_if_lt_x_gpu(8)
    @pytest.mark.parametrize(
        "ep1, fsdp1, world_size1, ep2, fsdp2, world_size2",
        [
            # test EP + FSDP + DP : (ep2,fsdp2,dp1) ->(ep2,fsdp2,dp2)
            (2, 2, 4, 2, 2, 8),
            # test EP + FSDP + DP : (ep2,fsdp2,dp1) ->(ep2,fsdp2,dp2)
            (2, 3, 6, 2, 4, 8),
        ],
    )
    @skip_if_lt_x_gpu(8)
    def test_sharded_hybrid_shard_mixtral(self, ep1, fsdp1, world_size1, ep2, fsdp2, world_size2, temp_dir):
        "Test sharded hybrid sharding on mixtral with different shard degree(ep + fsdp) and dp."
        self.run_hybrid_test(
            model_type="mixtral",
            checkpoint_type=SageMakerCheckpointType.SHARDED,
            tp1=1,
            fsdp1=fsdp1,
            ep1=ep1,
            world_size1=world_size1,
            tp2=1,
            fsdp2=fsdp2,
            ep2=ep2,
            world_size2=world_size2,
            tmp_dir=temp_dir,
        )
