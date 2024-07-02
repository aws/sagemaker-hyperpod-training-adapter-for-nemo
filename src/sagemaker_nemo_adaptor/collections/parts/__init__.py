from sagemaker_nemo_adaptor.utils.sm_env_utils import enable_dummy_sm_env

enable_dummy_sm_env()  # Need to be called before torch sagemaker is imported TODO: call in a more generic way

from .fsdp_strategy import SageMakerFSDPStrategy
