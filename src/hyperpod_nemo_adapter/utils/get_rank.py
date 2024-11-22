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

import torch
import torch.distributed as dist
import torch.sagemaker as tsm
from nemo.utils.env_var_parsing import get_envint
from torch.sagemaker.utils.process_group_utils import get_global_ranks


def is_global_rank_zero():
    """Helper function to determine if the current process is global_rank 0 (the main process),
    TODO: Add support for getting EKS rank
    """
    # Try to get the pytorch RANK env var
    # RANK is set by torch.distributed.launch
    rank = get_envint("RANK", None)
    if rank is not None:
        return rank == 0

    # Try to get the SLURM global rank env var
    # SLURM_PROCID is set by SLURM
    slurm_rank = get_envint("SLURM_PROCID", None)
    if slurm_rank is not None:
        return slurm_rank == 0

    # if neither pytorch and SLURM env vars are set
    # check NODE_RANK/GROUP_RANK and LOCAL_RANK env vars
    # asume global_rank is zero if undefined
    node_rank = get_envint("NODE_RANK", get_envint("GROUP_RANK", 0))
    local_rank = get_envint("LOCAL_RANK", 0)
    return node_rank == 0 and local_rank == 0


def get_rank():
    """Helper function that returns torch.distributed.get_rank() if DDP has been initialized otherwise it returns 0."""

    if is_global_rank_zero():
        return 0
    else:
        return torch.distributed.get_rank()


def is_action_rank(global_rank):
    return tsm.state.ranker.get_rep_rank(global_rank) == 0


def get_coordinator_rank(process_group):
    return min(get_global_ranks(process_group))


def get_current_replication_group(global_rank):
    replication_group = None
    replication_ranks = None
    for ranks in tsm.state.ranker.get_rep_groups():
        group = dist.new_group(ranks)
        if global_rank in ranks:
            replication_group = group
            replication_ranks = ranks
    assert replication_group, f"{global_rank} Replication group not found"
    assert replication_ranks, f"{global_rank} Replication ranks not found"
    return min(replication_ranks), replication_group
