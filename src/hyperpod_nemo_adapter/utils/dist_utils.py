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

from transformers import set_seed

from hyperpod_nemo_adapter.utils.app_state import SageMakerAppState


def initialize_model_parallel_for_nemo(
    world_size, global_rank, local_rank, tensor_model_parallel_size=1, context_parallel_size=1, seed=None
):
    # updating NeMo globals
    app_state = SageMakerAppState()
    app_state.global_rank = global_rank
    app_state.world_size = world_size
    app_state.local_rank = local_rank
    app_state.tensor_model_parallel_size = tensor_model_parallel_size

    try:
        import torch.sagemaker as tsm

        tp_rank = tsm.state.tp_rank
        app_state.tensor_model_parallel_rank = tp_rank
    except:
        # HF case
        pass

    app_state.model_parallel_size = tensor_model_parallel_size
    app_state.data_parallel_size = world_size // (tensor_model_parallel_size * context_parallel_size)
    app_state.data_parallel_rank = global_rank // (tensor_model_parallel_size * context_parallel_size)

    _set_random_seed(seed)

    app_state._is_megatron_initialized = True


def _set_random_seed(seed):
    set_seed(seed)
