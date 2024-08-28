from enum import Enum, auto, unique

GPUS_PER_NODE = 8


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
