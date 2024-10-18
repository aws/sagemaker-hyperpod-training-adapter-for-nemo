from unittest.mock import patch

import pytest
from omegaconf import OmegaConf
from pydantic import BaseModel

from sagemaker_nemo_adaptor.conf.config_schemas import (
    ConfigForbid,
    ConfigWithSMPForbid,
    get_model_validator,
)
from sagemaker_nemo_adaptor.constants import ModelType
from sagemaker_nemo_adaptor.utils.config_utils import (
    _validate_custom_recipe_extra_params,
    _validate_model_type,
    _validate_schema,
)


@pytest.fixture
def sample_config():
    config_dict = {
        "model": {
            "do_finetune": False,
            "model_type": "value",
        },
        "use_smp": True,
        "distributed_backend": "nccl",
        "trainer": {},
    }
    return OmegaConf.create(config_dict)


def test_get_model_validator(sample_config):

    # Test for valid smp model type
    assert get_model_validator(sample_config.use_smp) == ConfigWithSMPForbid

    # Test for valid hf model type
    sample_config.use_smp = False
    assert get_model_validator(sample_config.use_smp) == ConfigForbid


def test_validate_custom_recipe_extra_params(sample_config):
    # Test for extra fields
    class MockModel(BaseModel):
        pass

    with pytest.raises(AttributeError):
        with patch("os.environ['SLURM_JOB_ID']", return_value=True):
            _validate_custom_recipe_extra_params(MockModel)


def test_validate_schema(sample_config):

    # Test for valid schema
    sample_config.model.model_type = ModelType.LLAMA_V3.value
    _validate_schema(sample_config)
