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

from unittest.mock import patch

import pytest
from omegaconf import OmegaConf
from pydantic import BaseModel

from hyperpod_nemo_adapter.conf.config_schemas import (
    ConfigForbid,
    ConfigWithSMPForbid,
    get_model_validator,
    validate_distributed_degrees,
)
from hyperpod_nemo_adapter.constants import ModelType
from hyperpod_nemo_adapter.utils.config_utils import (
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
        "use_smp_model": True,
        "distributed_backend": "nccl",
        "trainer": {},
    }
    return OmegaConf.create(config_dict)


def test_get_model_validator(sample_config):

    # Test for valid smp model type
    assert get_model_validator(sample_config.use_smp_model) == ConfigWithSMPForbid

    # Test for valid hf model type
    sample_config.use_smp_model = False
    assert get_model_validator(sample_config.use_smp_model) == ConfigForbid

    with pytest.raises(ValueError):
        get_model_validator(sample_config.use_smp_model, extra="invalid")


def test_validate_distributed_degrees():

    # Test for invalid distributed degrees

    # Test world_size % degree_mult > 0
    with pytest.raises(ValueError):
        validate_distributed_degrees(
            shard_degree=2,
            tensor_model_parallel_degree=8,
            expert_model_parallel_degree=8,
            context_parallel_degree=8,
            num_nodes=1,
        )

    # CP degree > shard degree
    with pytest.raises(ValueError):
        validate_distributed_degrees(
            shard_degree=1,
            tensor_model_parallel_degree=8,
            expert_model_parallel_degree=1,
            context_parallel_degree=8,
            num_nodes=1,
        )


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


def test_validate_model_type(sample_config):

    # Test for invalid None values
    with pytest.raises(AttributeError):
        _validate_model_type(model_type=None, hf_model_name_or_path=None)

    # Test for invalid model_type (should pass with warning)
    _validate_model_type(model_type="fake_model_type", hf_model_name_or_path=None)
