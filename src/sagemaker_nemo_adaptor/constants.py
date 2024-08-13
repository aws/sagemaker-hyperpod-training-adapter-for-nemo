from enum import Enum, unique


@unique
class ModelType(Enum):
    LLAMA_V3 = "llama_v3"


class SageMakerParallelParams(Enum):
    TENSOR_MODEL_PARALLEL_DEGREE = "tensor_model_parallel_degree"
    EXPERT_MODEL_PARALLEL_DEGREE = "expert_model_parallel_degree"
    CONTEXT_MODEL_PARALLEL_DEGREE = "context_model_parallel_degree"
