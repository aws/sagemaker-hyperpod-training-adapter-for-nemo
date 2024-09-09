from typing import Any, Literal, Optional, Union

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)
from pydantic_core.core_schema import FieldValidationInfo

from sagemaker_nemo_adaptor.constants import (
    GPUS_PER_NODE,
    ModelType,
    SageMakerMonitorMode,
)
from sagemaker_nemo_adaptor.utils.general_utils import is_power_of_two
from sagemaker_nemo_adaptor.utils.log_utils import Logger

_logger = Logger().get_logger()


"""
HELPER FUNCTIONS
"""


def validate_degrees_le_world_size(
    shard_degree: Optional[int],
    tensor_model_parallel_degree: Optional[int],
    expert_model_parallel_degree: Optional[int],
    num_nodes: Optional[int],
) -> None:
    """
    Check that the degrees are <= world size so the data doesn't exceed the hardware capacity.
    """
    # default param values to 1 if they are missing
    sd = shard_degree or 1
    tp = tensor_model_parallel_degree or 1
    ep = expert_model_parallel_degree or 1
    degree_mult = sd * tp * ep
    world_size = (num_nodes or 1) * GPUS_PER_NODE

    if degree_mult > world_size:
        msg = "The multiplication of 'shard_degree', 'tensor_model_parallel_degree'"
        msg += " and 'expert_model_parallel_degree' is greater than the world size"
        raise ValueError(msg)


"""
BASE CLASSES
"""


class SageMakerParallelConfig(BaseModel):
    tensor_model_parallel_degree: int = Field(default=1, ge=1)
    expert_model_parallel_degree: int = Field(default=1, ge=1)
    # context_model_parallel_degree: int = Field(ge=1) # Rubik will support this soon

    @field_validator("tensor_model_parallel_degree", "expert_model_parallel_degree")
    @classmethod
    def validate_tensor_model_parallel_degree(cls, value: int, info: FieldValidationInfo):
        if not is_power_of_two(value):
            raise ValueError(f"{info.field_name} must be a power of 2")

        return value


class BaseModelOptimizerScheduler(BaseModel):
    name: Literal["CosineAnnealing"] = "CosineAnnealing"
    warmup_steps: int = Field(default=500, ge=0)
    constant_steps: int = Field(default=0, ge=0)
    min_lr: float = Field(default=2e-5, ge=0)


class BaseModelOptimizerConfig(BaseModel):
    # https://pytorch.org/docs/stable/optim.html
    name: Literal["adamw"] = "adamw"

    # ADAMW PARAMS
    lr: float = Field(default=2e-4, ge=0)
    weight_decay: float = Field(default=0.01, ge=0)
    betas: list[float] = [0.9, 0.98]

    # OTHER
    sched: BaseModelOptimizerScheduler = Field(default_factory=BaseModelOptimizerScheduler)


class BaseModelDataConfig(BaseModel):
    train_dir: Optional[list[str]] = None
    val_dir: Optional[list[str]] = None
    dataset_type: Literal["hf", "synthetic"] = "hf"
    use_synthetic_data: bool = False
    zipped_data: bool = False  # TODO: ideally we should have utils to check whether a data is zipped

    @model_validator(mode="after")
    def before_model_validations(self) -> "BaseModelDataConfig":
        if not self.use_synthetic_data:
            if not (self.train_dir and self.val_dir):
                raise ValueError("'train_dir' and 'val_dir' are required since model is not using Synthetic Data")

        return self


class BaseModelConfig(BaseModel):
    # needed to disallow protected namespace "model_"
    model_config: ConfigDict = ConfigDict(protected_namespaces=())

    model_type: str = Field(default=ModelType.LLAMA_V3.value)

    train_batch_size: int = Field(default=2, ge=1)
    fsdp: bool = True
    moe: bool = False
    sequence_parallel: bool = True
    activation_checkpointing: bool = True
    activation_loading_horizon: int = Field(default=2, ge=1)
    delayed_param: bool = True
    offload_activations: bool = False
    seed: int = 12345
    grad_clip: float = Field(default=1.0, ge=0)  # 0 == disabled

    # FSDP Configs
    sharding_strategy: Literal[
        "no_shard", "shard_grad_op", "hybrid_shard", "_hybrid_shard_zero2", "full_shard"
    ] = "hybrid_shard"
    forward_prefetch: bool = True
    shard_degree: Optional[int] = Field(default=None, ge=1)
    backward_fetch_policy: Literal["backward_post", "backward_pre"] = "backward_pre"
    auto_wrap_policy: Literal[
        "size_based_auto_wrap_policy", "transformer_auto_wrap_policy"
    ] = "transformer_auto_wrap_policy"
    limit_all_gathers: bool = True
    use_orig_param: bool = False

    # Model Architecture
    max_context_width: int = Field(default=4096, ge=1)
    max_position_embeddings: Optional[int] = Field(default=None, ge=1)
    num_layers: int = Field(default=32, ge=1)
    hidden_width: int = Field(default=4096, ge=1)
    num_heads: int = Field(default=32, ge=1)
    intermediate_size: int = Field(default=14336, ge=1)
    initializer_range: float = Field(default=0.02, ge=0)
    pad_token_id: int = 0
    layernorm_epsilon: float = Field(default=1e-5, ge=0)
    attention_bias: bool = False
    vocab_size: int = Field(default=32000, ge=1)
    activation: Literal["gelu"] = "gelu"  # TODO: https://www.tensorflow.org/api_docs/python/tf/keras/activations?
    num_key_value_heads: Optional[int] = Field(default=None, ge=1)
    use_flash_attention: bool = True

    # Transformer Engine
    transformer_engine: bool = True
    fp8: bool = True
    fp8_amax_history_len: int = Field(default=1024, ge=1)
    fp8_amax_compute_algo: Literal["max", "most_recent"] = "max"

    # Fine-Tuning
    # Rubik calls it `pretrained_model_weights` but we opted to follow the name used by HF
    pretrained_model_name_or_path: Optional[str] = None

    precision: Union[str, int, None] = None

    lr_decay_iters: int = Field(default=47683, ge=1)  # range? Optional?

    log_reduced_training_loss: bool = True  # Rubik has False

    # CHILD CONFIGS
    optim: BaseModelOptimizerConfig = Field(default_factory=BaseModelOptimizerConfig)
    data: BaseModelDataConfig = Field(default_factory=lambda: BaseModelDataConfig(use_synthetic_data=True))

    @model_validator(mode="before")
    def before_model_validations(cls, data: Any) -> Any:
        if data.get("max_position_embeddings") is None:
            data["max_position_embeddings"] = data.get("max_context_width")

        model_type = data.get("model_type")
        if model_type and model_type not in [e.value for e in ModelType]:
            raise ValueError(f"Invalid model_type '{model_type}'")

        return data

    @model_validator(mode="after")
    def after_model_validations(self) -> "BaseModelConfig":
        msg_fn = lambda field, val: f"'{field}' is suggested to be a power of 2. Current value is {val}"

        if not is_power_of_two(self.max_context_width):
            _logger.warning(msg_fn("max_context_width", self.max_context_width))

        if not is_power_of_two(self.hidden_width):
            _logger.warning(msg_fn("hidden_width", self.hidden_width))

        if not is_power_of_two(self.num_heads):
            _logger.warning(msg_fn("num_heads", self.num_heads))

        if not (self.num_key_value_heads is None or is_power_of_two(self.num_key_value_heads)):
            _logger.warning(msg_fn("num_key_value_heads", self.num_key_value_heads))

        return self


class BaseTrainerConfig(BaseModel):
    devices: int = Field(default=8, ge=1)
    num_nodes: int = Field(default=1, ge=1)
    # https://github.com/Lightning-AI/pytorch-lightning/blob/master/src/lightning/pytorch/trainer/trainer.py#L91
    accelerator: Literal["gpu", "auto"] = "gpu"
    #   https://github.com/Lightning-AI/pytorch-lightning/blob/828fd998961f6a60f92c35254bb94d6e049ad069/src/lightning/fabric/plugins/precision/precision.py#L36
    precision: Union[str, int] = "bf16"
    max_steps: int = Field(default=50, ge=1)
    log_every_n_steps: int = Field(default=10, ge=0)  # 0 == no logging

    accumulate_grad_batches: int = Field(default=1, ge=1)
    gradient_clip_val: float = Field(default=1.0, ge=0)  # TODO: Figure out how to set up in the Trainer

    @model_validator(mode="after")
    def after_model_validations(self) -> "BaseTrainerConfig":
        if self.devices % GPUS_PER_NODE != 0:
            raise ValueError(f"'devices' ({self.devices}) must be a multiple of {GPUS_PER_NODE}")

        return self


class BaseCheckpointCallbackConfig(BaseModel):
    save_top_k: int = Field(default=10, ge=0)  # 0 == no checkpointing
    every_n_train_steps: int = Field(default=0, ge=0)
    monitor: str = "step"
    mode: str = Field(default=SageMakerMonitorMode.MAX.value)


class BaseExportFullModelConfig(BaseModel):
    every_n_train_steps: int = Field(default=0, ge=0)


class BaseExpManager(BaseModel):
    exp_dir: str = "/fsx/users/rnadimp/exp/"
    name: str = "my_experiment"
    create_tensorboard_logger: bool = True
    create_checkpoint_callback: bool = True
    checkpoint_callback_params: BaseCheckpointCallbackConfig = Field(default_factory=BaseCheckpointCallbackConfig)
    export_full_model: BaseExportFullModelConfig = Field(default_factory=BaseExportFullModelConfig)
    checkpoint_dir: Optional[str] = None
    resume_from_checkpoint: Optional[str] = None
    auto_checkpoint: Optional[bool] = False


class BaseInternalConfig(BaseModel):
    config_verified: bool = True


class BaseRunConfig(BaseModel):
    name: str = "llama-8b"
    results_dir: Optional[str] = None
    time_limit: Optional[str] = None


class BaseConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: list[str] = ["hf_llama_8b"]
    use_smp: bool = True
    distributed_backend: Literal["smddp", "nccl"]
    restore_from_path: Optional[str] = None

    # CHILD CONFIGS - optional ones must be configured on the subclass
    model: Optional[Union[BaseModelConfig, type[BaseModelConfig]]] = None
    trainer: BaseTrainerConfig
    exp_manager: Optional[BaseExpManager] = None
    internal: BaseInternalConfig = Field(default_factory=BaseInternalConfig)
    run: Optional[BaseRunConfig] = None

    @model_validator(mode="before")
    def before_model_validations(cls, data: Any) -> Any:
        model = data.get("model")

        if model is None:
            raise ValueError("Field 'model' is required")

        if model.get("precision") is None:
            model["precision"] = data.get("trainer", {}).get("precision")

        return data

    @model_validator(mode="after")
    def after_model_validations(self) -> "BaseConfig":
        sd = getattr(self.model, "shard_degree", None)
        tp = getattr(self.model, "tensor_model_parallel_degree", None)
        ed = getattr(self.model, "expert_model_parallel_degree", None)
        num_nodes = getattr(self.trainer, "num_nodes", None)

        validate_degrees_le_world_size(sd, tp, ed, num_nodes)

        return self


"""
LLAMA V3 CONFIGS
"""


class LlamaV3ModelConfigWithSMP(BaseModelConfig, SageMakerParallelConfig):
    pass


class LlamaV3Config(BaseConfig):
    model: BaseModelConfig  # type: ignore comment;


class LlamaV3ConfigWithSMP(BaseConfig):
    model: LlamaV3ModelConfigWithSMP  # type: ignore comment;


"""
CONFIG SCHEMAS
"""

# Based on Hugging Face
HF_SCHEMAS: dict[str, type[BaseModel]] = {ModelType.LLAMA_V3.value: LlamaV3Config}

# With Rubik optimizations
SMP_SCHEMAS: dict[str, type[BaseModel]] = {ModelType.LLAMA_V3.value: LlamaV3ConfigWithSMP}
