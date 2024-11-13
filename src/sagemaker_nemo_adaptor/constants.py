from enum import Enum, auto, unique

GPUS_PER_NODE = 8
OPTIMIZER_KEY_PREFIX = "optimizer"

TRAIN_SEQUENCE_NUMBER = "train_sequence_num"
VAL_SEQUENCE_NUMBER = "val_sequence_num"
DEFAULT_SEED = 12345

CONFIG_MAPPING_HF_TO_RECIPE_ALIASES = {
    "vocab_size": ["vocab_size"],
    "hidden_size": ["hidden_size"],
    "intermediate_size": ["intermediate_size"],
    "num_hidden_layers": ["num_hidden_layers", "num_layers"],
    "num_attention_heads": ["num_attention_heads", "num_heads"],
    "max_position_embeddings": ["max_context_width"],
    "initializer_range": ["initializer_range"],
    "num_key_value_heads": ["num_key_value_heads"],
    "rms_norm_eps": ["layernorm_epsilon", "rms_norm_eps"],
    "rotary_pct": ["rotary_percentage", "rotary_pct"],
    "rotary_emb_base": ["rotary_emb_base"],
    "sliding_window": ["sliding_window"],
    "rope_theta": ["rope_theta"],
    "num_experts_per_tok": ["num_experts_per_tok"],
    "num_local_experts": ["num_local_experts"],
    "delayed_param": ["delayed_param"],
}


@unique
class ModelType(Enum):
    BERT = "bert"
    GPT2 = "gpt2"
    LLAMA_V2 = "llama_v2"
    LLAMA_V3 = "llama_v3"
    MISTRAL = "mistral"
    MIXTRAL = "mixtral"


class DataTypes:
    ARROW = ".arrow"
    JSON = ".json"
    JSONGZ = ".json.gz"


class SageMakerParallelParams(Enum):
    TENSOR_MODEL_PARALLEL_DEGREE = "tensor_model_parallel_degree"
    EXPERT_MODEL_PARALLEL_DEGREE = "expert_model_parallel_degree"
    CONTEXT_MODEL_PARALLEL_DEGREE = "context_model_parallel_degree"


class SageMakerCheckpointType(Enum):
    FULL = auto()
    SHARDED = auto()
    LOCAL = auto()
    PEFT_FULL = auto()
    PEFT_SHARDED = auto()


class SageMakerMonitorMode(Enum):
    MAX = "max"
    MIN = "min"
