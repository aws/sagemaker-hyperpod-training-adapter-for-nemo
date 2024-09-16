import pytest
from omegaconf import OmegaConf
from pydantic import BaseModel

from sagemaker_nemo_adaptor.conf.config_schemas import HF_SCHEMAS, SMP_SCHEMAS
from sagemaker_nemo_adaptor.constants import ModelType
from sagemaker_nemo_adaptor.utils.config_utils import (
    _get_model_validator,
    _validate_custom_recipe_extra_params,
    _validate_model_type,
    _validate_schema,
)


@pytest.fixture
def sample_config():
    config_dict = {
        "model": {
            "model_type": "value",
        },
        "use_smp": True,
        "distributed_backend": "nccl",
        "trainer": {},
    }
    return OmegaConf.create(config_dict)


def test_get_model_validator(sample_config):
    # Test for invalid smp model type
    with pytest.raises(ValueError):
        _get_model_validator(sample_config)

    # Test for valid smp model type
    sample_config.model.model_type = ModelType.LLAMA_V3.value
    assert _get_model_validator(sample_config) == SMP_SCHEMAS[sample_config.model.model_type]

    # Test for invalid hf model type
    sample_config.use_smp = False
    sample_config.model.model_type = "value"
    with pytest.raises(ValueError):
        _get_model_validator(sample_config)

    # Test for valid hf model type
    sample_config.model.model_type = ModelType.LLAMA_V3.value
    assert _get_model_validator(sample_config) == HF_SCHEMAS[sample_config.model.model_type]


def test_validate_model_type(sample_config):
    # Test for invalid model type
    sample_config.model.model_type = "value"
    with pytest.raises(AttributeError):
        _validate_model_type(sample_config)

    # Test for missing model type
    sample_config.model.model_type = None
    with pytest.raises(AttributeError):
        _validate_model_type(sample_config.model.model_type)

    # Test for valid model type
    sample_config.model.model_type = ModelType.LLAMA_V3.value
    _validate_model_type(sample_config.model.model_type)


def test_validate_custom_recipe_extra_params(sample_config):
    # Test for extra fields
    class MockModel(BaseModel):
        pass

    with pytest.raises(AttributeError):
        _validate_custom_recipe_extra_params(MockModel)


def test_validate_schema(sample_config):
    # Test for invalid schema
    sample_config.model.model_type = "value"
    with pytest.raises(ValueError):
        _validate_schema(sample_config)

    # Test for valid schema
    sample_config.model.model_type = ModelType.LLAMA_V3.value
    _validate_schema(sample_config)
