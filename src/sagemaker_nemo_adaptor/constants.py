from enum import Enum, auto, unique

GPUS_PER_NODE = 8
OPTIMIZER_KEY_PREFIX = "optimizer"

TRAIN_SEQUENCE_NUMBER = "train_sequence_num"
VAL_SEQUENCE_NUMBER = "val_sequence_num"
DEFAULT_SEED = 12345


@unique
class ModelType(Enum):
    BERT = "bert"
    GPT_NEOX = "gpt_neox"
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
    PEFT = auto()


class SageMakerMonitorMode(Enum):
    MAX = "max"
    MIN = "min"
