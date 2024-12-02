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
from typing import Any, Literal, Optional, Sequence, Union

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from hyperpod_nemo_adapter.constants import (
    GPUS_PER_NODE,
    ModelType,
    SageMakerMonitorMode,
)
from hyperpod_nemo_adapter.utils.general_utils import (
    can_use_multimodal,
    is_power_of_two,
    is_slurm_run,
)
from hyperpod_nemo_adapter.utils.log_utils import Logger

_logger = Logger().get_logger()
smp = None


"""
HELPER FUNCTIONS
"""


def create_dynamic_model(base_model_cls, extra: str = "forbid"):
    """
    Dynamically creatig the model config based on the extra config
    extra="forbid": No extra configs will be allowed, will error out
    extra="allow": Extra configs will be merged into the model config without validate
    """
    config = ConfigDict(protected_namespaces=(), extra=extra)

    class DynamicModelConfig(base_model_cls):
        model_config = config

    return DynamicModelConfig


def get_model_validator(use_smp_model, extra="forbid") -> type[BaseModel]:
    global smp
    smp = use_smp_model
    if extra == "forbid":
        if use_smp_model:
            return ConfigWithSMPForbid
        else:
            return ConfigForbid
    elif extra == "allow":
        if use_smp_model:
            return ConfigWithSMPAllow
        else:
            return ConfigAllow
    else:
        raise ValueError(f"Unsupported extra type {extra}")


def validate_distributed_degrees(
    shard_degree: Optional[int],
    tensor_model_parallel_degree: Optional[int],
    expert_model_parallel_degree: Optional[int],
    context_parallel_degree: Optional[int],
    num_nodes: Optional[int],
) -> None:
    """
    Check that the degrees are legal.
    """
    # default param values to 1 if they are missing
    sd = shard_degree or 1
    tp = tensor_model_parallel_degree or 1
    ep = expert_model_parallel_degree or 1
    cp = context_parallel_degree or 1
    degree_mult = sd * tp * ep
    gpu_per_node = int(os.environ.get("LOCAL_WORLD_SIZE", GPUS_PER_NODE))
    world_size = (num_nodes or 1) * gpu_per_node

    # Validate the degree multiplication <= world_size
    if world_size % degree_mult > 0:
        raise ValueError(
            f"Please provide valid sizes: world size ({world_size}) is not a product of "
            f"(shard_degree, tensor_model_parallel_degree, expert_model_parallel_degree) ({sd}, {tp}, {ep})."
        )

    # Validate CP degree <= shard degree
    if cp > 1 and cp > sd:
        raise ValueError(
            f"Please provide valid sizes: context parallel degree ({cp}) shouldn't be greater than shard degree ({sd})"
        )


"""
BASE CLASSES
"""


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


class BaseRopeScalingConfig(BaseModel):
    # rope scaling
    rope_type: Literal["llama3", None] = None
    factor: float = Field(default=8.0)
    high_freq_factor: float = Field(default=4.0)
    low_freq_factor: float = Field(default=1.0)
    original_max_position_embeddings: int = Field(default=8192)


class BaseModelDataConfig(BaseModel):
    train_dir: Optional[Union[str, list[str]]] = None
    val_dir: Optional[Union[str, list[str]]] = None
    dataset_type: Literal["hf", "synthetic"] = "hf"
    use_synthetic_data: bool = False


class BaseModelNsysProfileConfig(BaseModel):
    enabled: bool = False
    start_step: int = 10
    end_step: int = 10
    ranks: list = [0]
    gen_shape: bool = False


class BaseGPUAffinityConfig(BaseModel):
    enabled: bool = True


# default values are from https://github.com/gaogaotiantian/viztracer/blob/master/src/viztracer/viztracer.py#L22
class BaseVizTracerConfig(BaseModel):
    enabled: bool = False
    ranks: list = [0]
    tracer_entries: int = 1000000
    verbose: int = 1
    max_stack_depth: int = -1
    ignore_c_function: bool = True
    ignore_frozen: bool = True
    log_func_retval: bool = False
    log_func_args: bool = False
    log_print: bool = False
    log_gc: bool = False
    log_sparse: bool = False
    log_async: bool = False
    log_audit: Optional[Sequence[str]] = None
    pid_suffix: bool = False
    file_info: bool = True
    register_global: bool = True
    trace_self: bool = False
    min_duration: float = 200
    minimize_memory: bool = False
    dump_raw: bool = False
    sanitize_function_name: bool = False
    output_file: Optional[str] = None


class BaseModelPeftConfig(BaseModel):
    peft_type: Optional[Literal["lora", "qlora_4bit"]] = None
    rank: int = Field(default=32, ge=1)
    alpha: float = Field(default=16.0, ge=0)
    dropout: float = Field(default=0.1, ge=0)
    target_modules: Optional[list[str]] = None


class BaseModelConfig(BaseModel):
    # needed to disallow protected namespace "model_"
    model_config: ConfigDict = ConfigDict(protected_namespaces=())

    model_type: str = Field(default=ModelType.LLAMA_V3.value)

    train_batch_size: int = Field(default=2, ge=1)
    val_batch_size: int = Field(default=1, ge=1)
    fsdp: bool = True
    moe: bool = False
    activation_checkpointing: bool = True
    activation_loading_horizon: int = Field(default=2, ge=1)
    delayed_param: bool = True
    offload_activations: bool = False
    seed: int = 12345
    grad_clip: float = Field(default=1.0, ge=0)  # 0 == disabled

    # FSDP Configs
    sharding_strategy: Literal["no_shard", "shard_grad_op", "hybrid_shard", "_hybrid_shard_zero2", "full_shard"] = (
        "hybrid_shard"
    )
    forward_prefetch: bool = True
    shard_degree: Optional[int] = Field(default=None, ge=1)
    backward_fetch_policy: Literal["backward_post", "backward_pre"] = "backward_pre"
    auto_wrap_policy: Literal["size_based_auto_wrap_policy", "transformer_auto_wrap_policy"] = (
        "transformer_auto_wrap_policy"
    )
    limit_all_gathers: bool = True
    use_orig_param: bool = True

    # Parallel degrees
    context_parallel_degree: int = Field(default=1, ge=1)
    tensor_model_parallel_degree: int = Field(default=1, ge=1)
    expert_model_parallel_degree: int = Field(default=1, ge=1)

    # Model Architecture
    max_context_width: int = Field(default=2048, ge=1)  # max_context_width is always required for data purposes
    max_position_embeddings: int | None = Field(default=None, ge=1)
    num_hidden_layers: int | None = Field(default=None, ge=1)
    hidden_size: int | None = Field(default=None, ge=1)
    num_attention_heads: int | None = Field(default=None, ge=1)
    intermediate_size: int | None = Field(default=None, ge=1)
    initializer_range: float | None = Field(default=None, ge=0)
    layernorm_epsilon: float | None = Field(default=None, ge=0)
    vocab_size: int | None = Field(default=None, ge=1)
    num_key_value_heads: int | None = Field(default=None, ge=1)
    use_flash_attention: bool | None = None
    mistral_sliding_window: int | None = Field(default=None, ge=1)
    rms_norm_eps: float | None = Field(default=None, ge=0)
    rope_theta: float = Field(default=10000.0)
    multi_modal: bool = False
    tie_word_embeddings: bool = False

    # Mixture of Experts
    mixtral_sliding_window: int | None = Field(default=None, ge=1)
    num_experts_per_tok: int | None = Field(default=None, ge=1)
    num_local_experts: int | None = Field(default=None, ge=1)
    moe_load_balancing: Literal["sinkhorn", "balanced", "aux_loss", "none"] = "sinkhorn"
    global_token_shuffle: bool | None = None
    moe_all_to_all_dispatcher: bool | None = None

    # fp8
    fp8: bool = True
    fp8_amax_history_len: int = Field(default=1024, ge=1)
    fp8_amax_compute_algo: Literal["max", "most_recent"] = "max"

    # Fine-Tuning
    do_finetune: bool = True
    hf_model_name_or_path: Optional[str] = None
    hf_access_token: Optional[str] = None

    precision: Union[str, int, None] = None

    lr_decay_iters: int = Field(default=47683, ge=1)

    log_reduced_training_loss: bool = True

    # CHILD CONFIGS
    optim: BaseModelOptimizerConfig = Field(default_factory=BaseModelOptimizerConfig)
    data: BaseModelDataConfig = Field(default_factory=lambda: BaseModelDataConfig(use_synthetic_data=True))
    gpu_affinity: BaseGPUAffinityConfig = Field(default_factory=BaseGPUAffinityConfig)
    nsys_profile: BaseModelNsysProfileConfig = Field(default_factory=BaseModelNsysProfileConfig)
    viztracer: BaseVizTracerConfig = Field(default_factory=BaseVizTracerConfig)
    peft: BaseModelPeftConfig = Field(default_factory=BaseModelPeftConfig)
    rope_scaling: BaseRopeScalingConfig = Field(default_factory=BaseRopeScalingConfig)

    # Transformer Engine
    nvte_attn_backend: Optional[Literal["fused", "flash"]] = None

    @model_validator(mode="before")
    def before_model_validations(cls, data: Any) -> Any:

        if data.get("max_position_embeddings") is None:
            data["max_position_embeddings"] = data.get("max_context_width")

        model_data_config = data.get("data", None)
        multi_modal = data.get("multi_modal")
        if multi_modal and not can_use_multimodal():
            raise ValueError("'multi_modal' requires transformers version of at least 4.45.2")

        if multi_modal and not data.get("model_type") == "llama_v3":
            raise ValueError("'multi_modal' only supported with 'model_type' llama_v3")

        if model_data_config and not model_data_config.get("use_synthetic_data", None):
            if not model_data_config.get("train_dir", None) and not multi_modal:
                raise ValueError("'train_dir' is required since model is not using Synthetic or multi-modal Data")

        if model_data_config and model_data_config.get("use_synthetic_data", None) and multi_modal:
            raise ValueError("'use_synthetic_data' not supported with 'multi_modal' training")

        return data

    @model_validator(mode="after")
    def after_model_validations(self) -> "BaseModelConfig":
        msg_fn = lambda field, val: f"'{field}' is suggested to be a power of 2. Current value is {val}"

        if getattr(self, "max_context_width", None) is not None and not is_power_of_two(self.max_context_width):
            _logger.warning(msg_fn("max_context_width", self.max_context_width))

        if getattr(self, "hidden_size", None) is not None and not is_power_of_two(self.hidden_size):
            _logger.warning(msg_fn("hidden_size", self.hidden_size))

        if getattr(self, "num_attention_heads", None) is not None and not is_power_of_two(self.num_attention_heads):
            _logger.warning(msg_fn("num_attention_heads", self.num_attention_heads))

        if getattr(self, "num_key_value_heads", None) is not None and not (
            self.num_key_value_heads is None or is_power_of_two(self.num_key_value_heads)
        ):
            _logger.warning(msg_fn("num_key_value_heads", self.num_key_value_heads))

        if self.do_finetune and self.hf_model_name_or_path is None:
            raise ValueError("Must provide 'hf_model_name_or_path' or set 'do_finetune' to False")

        if not smp and (self.tensor_model_parallel_degree > 1 or self.expert_model_parallel_degree > 1):
            raise ValueError(
                "Non SMP Model implementations do not support tensor_model_parallel_degree or expert_model_parallel_degree > 1"
            )

        if not self.activation_checkpointing and self.activation_loading_horizon > 1:
            _logger.warning(
                "Note: activation_loading_horizon will not be activated since activation_checkpointing is disabled"
            )

        return self


class BaseTrainerConfig(BaseModel):
    devices: Union[str, int] = "auto"
    num_nodes: int = Field(default=1, ge=1)
    # https://github.com/Lightning-AI/pytorch-lightning/blob/master/src/lightning/pytorch/trainer/trainer.py#L91
    accelerator: Literal["gpu", "auto"] = "gpu"
    #   https://github.com/Lightning-AI/pytorch-lightning/blob/828fd998961f6a60f92c35254bb94d6e049ad069/src/lightning/fabric/plugins/precision/precision.py#L36
    precision: Union[str, int] = "bf16"
    max_steps: int = Field(default=50, ge=1)
    log_every_n_steps: int = Field(default=10, ge=0)  # 0 == no logging
    val_check_interval: Union[float, int] = Field(
        default=-1
    )  # How often to check the validation set (float = fraction, int = num_batches)
    limit_val_batches: Union[float, int] = Field(
        default=0.0, ge=0
    )  # How much of validation dataset to check (float = fraction, int = num_batches)

    @model_validator(mode="after")
    def after_model_validations(self) -> "BaseTrainerConfig":
        if "LOCAL_WORLD_SIZE" in os.environ and "WORLD_SIZE" in os.environ:
            # read from torchrun environment variables
            actual_devices = int(os.environ["LOCAL_WORLD_SIZE"])
            actual_num_nodes = int(os.environ["WORLD_SIZE"]) // actual_devices
            if isinstance(self.devices, int) and self.devices != actual_devices:
                raise ValueError(
                    f"'devices' ({self.devices}) does not equal actual number of devices ({actual_devices})"
                )
            if self.num_nodes != actual_num_nodes:
                raise ValueError(
                    f"'num_nodes' ({self.num_nodes}) does not equal actual number of nodes ({actual_num_nodes})"
                )

        return self


class BaseCheckpointCallbackConfig(BaseModel):
    save_top_k: int = Field(default=10, ge=0)  # 0 == no checkpointing
    every_n_train_steps: int = Field(default=0, ge=0)
    monitor: str = "step"
    mode: str = Field(default=SageMakerMonitorMode.MAX.value)
    save_last: bool = True


class BaseAutoCheckpointConfig(BaseModel):
    enabled: Optional[bool] = False
    warmup_steps: int = Field(default=12, ge=3)
    drop_n_warmup_steps: int = Field(default=3, ge=2)

    @model_validator(mode="after")
    def after_model_validations(self) -> "BaseAutoCheckpoint":
        if self.warmup_steps < self.drop_n_warmup_steps:
            raise ValueError(f"warmup_steps < drop_n_warmup_steps ({self.warmup_steps} < {self.drop_n_warmup_steps})")
        return self


class BaseExportFullModelConfig(BaseModel):
    every_n_train_steps: int = Field(default=0, ge=0)
    save_last: bool = True
    final_export_dir: str = None


class BaseExpManager(BaseModel):
    exp_dir: str = "/fsx/exp/"
    name: str = "my_experiment"
    explicit_log_dir: Optional[str] = None
    create_tensorboard_logger: bool = True
    create_checkpoint_callback: bool = True
    checkpoint_callback_params: BaseCheckpointCallbackConfig = Field(default_factory=BaseCheckpointCallbackConfig)
    export_full_model: BaseExportFullModelConfig = Field(default_factory=BaseExportFullModelConfig)
    checkpoint_dir: Optional[str] = None
    resume_from_checkpoint: Optional[str] = None
    auto_checkpoint: BaseAutoCheckpointConfig = Field(default_factory=BaseAutoCheckpointConfig)


class BaseRunConfig(BaseModel):
    name: str = "llama-8b"
    results_dir: Optional[str] = None
    time_limit: Optional[str] = None


class BaseConfig(BaseModel):
    # Only disable extra args for slurm workflow
    model_config = ConfigDict(extra="forbid" if is_slurm_run() else "allow")
    name: list[str] = ["hf_llama_8b"]
    use_smp_model: bool = True
    distributed_backend: Literal["smddp", "nccl"]
    log_perf_metrics: bool = False

    model: BaseModel
    trainer: BaseTrainerConfig
    exp_manager: Optional[BaseExpManager] = None
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
        ep = getattr(self.model, "expert_model_parallel_degree", None)
        cp = getattr(self.model, "context_parallel_degree", None)
        num_nodes = getattr(self.trainer, "num_nodes", None)

        validate_distributed_degrees(sd, tp, ep, cp, num_nodes)

        return self


class ConfigForbid(BaseConfig):
    model: create_dynamic_model(BaseModelConfig, extra="forbid")


class ConfigAllow(BaseConfig):
    model: create_dynamic_model(BaseModelConfig, extra="allow")


class ConfigWithSMPForbid(BaseConfig):
    model: create_dynamic_model(BaseModelConfig, extra="forbid")


class ConfigWithSMPAllow(BaseConfig):
    model: create_dynamic_model(BaseModelConfig, extra="allow")
