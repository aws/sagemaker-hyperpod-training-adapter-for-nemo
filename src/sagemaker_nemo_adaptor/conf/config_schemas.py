from typing import Literal, Optional, Union

from pydantic import BaseModel, ConfigDict

from sagemaker_nemo_adaptor.constants import ModelType

"""
BASE CLASSES
"""


class SageMakerParallelConfig(BaseModel):
    tensor_model_parallel_degree: int
    expert_model_parallel_degree: int
    # context_model_parallel_degree: int


class BaseModelConfig(BaseModel):
    # needed to disallow protected namespace "model_"
    model_config: ConfigDict = ConfigDict(protected_namespaces=())

    model_type: ModelType

    train_batch_size: int
    moe: bool
    sequence_parallel: bool
    activation_checkpointing: bool
    activation_loading_horizon: int
    delayed_param: bool
    offload_activations: bool
    seed: int
    grad_clip: float
    hf_pretrained_model: Optional[str]

    # FSDP Configs
    sharding_strategy: str
    forward_prefetch: bool
    shard_degree: int
    backward_fetch_policy: str
    auto_wrap_policy: str
    limit_all_gathers: bool
    use_orig_param: bool

    # model architecture
    max_context_width: int
    max_position_embeddings: Optional[int]
    num_layers: int
    hidden_width: int
    num_heads: int
    llama_intermediate_size: int
    initializer_range: float
    pad_token_id: int
    layernorm_epsilon: float
    attention_bias: bool
    vocab_size: int
    activation: Literal["gelu"]
    num_key_value_heads: Optional[int]
    use_flash_attention: bool

    # Transformer Engine
    transformer_engine: bool
    fp8: bool
    fp8_amax_history_len: int
    fp8_amax_compute_algo: Literal["max", "most_recent"]

    # finetune
    do_finetune: bool
    finetune_with_pretrained_weights: bool

    # Rubik calls it `pretrained_model_weights` but we opted to follow the name used by HF
    pretrained_model_name_or_path: Optional[str]

    precision: Union[str, int]

    lr_decay_iters: int

    log_reduced_training_loss: bool


class BaseTrainerConfig(BaseModel):
    devices: int
    num_nodes: int
    accelerator: Literal["gpu"]
    #   https://github.com/Lightning-AI/pytorch-lightning/blob/828fd998961f6a60f92c35254bb94d6e049ad069/src/lightning/fabric/plugins/precision/precision.py#L36
    precision: Union[str, int]
    max_steps: int
    log_every_n_steps: int

    accumulate_grad_batches: int
    gradient_clip_val: float


class BaseConfig(BaseModel):
    use_smp: bool
    distributed_backend: Literal["smddp", "nccl"]

    # child configs
    # model: Optional[BaseModel] = None
    trainer: BaseTrainerConfig


"""
LLAMA V3 CONFIGS
"""


class LlamaV3ModelConfigWithSMP(BaseModelConfig, SageMakerParallelConfig):
    pass


class LlamaV3Config(BaseConfig):
    model: BaseModelConfig


class LlamaV3ConfigWithSMP(BaseConfig):
    model: LlamaV3ModelConfigWithSMP


"""
CONFIG SCHEMAS
"""

# Based on Hugging Face
HF_SCHEMAS = {ModelType.LLAMA_V3.value: LlamaV3Config}

# With Rubik optimizations
SMP_SCHEMAS = {ModelType.LLAMA_V3.value: LlamaV3ConfigWithSMP}
