from nemo.utils import AppState
from transformers import set_seed


def initialize_model_parallel_for_nemo(world_size, global_rank, local_rank, tensor_model_parallel_size=1, seed=None):
    # updating NeMo globals
    # TODO: Update EP,PP when applicable
    app_state = AppState()
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
    app_state.data_parallel_size = world_size // tensor_model_parallel_size
    app_state.data_parallel_rank = global_rank // tensor_model_parallel_size

    _set_random_seed(seed)

    app_state._is_megatron_initialized = True


def _set_random_seed(seed):
    set_seed(seed)
