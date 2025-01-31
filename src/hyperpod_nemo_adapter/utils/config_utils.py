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

from functools import wraps
from typing import Any, Callable, Optional, TypeVar, cast

from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, ValidationError
from transformers import AutoConfig

from hyperpod_nemo_adapter.conf.config_schemas import get_model_validator
from hyperpod_nemo_adapter.constants import ModelType
from hyperpod_nemo_adapter.utils.general_utils import is_slurm_run
from hyperpod_nemo_adapter.utils.log_utils import Logger

_logger = Logger().get_logger()

_T = TypeVar("_T", bound=Callable[..., Any])

MAX_RETRY_TIME = 5


def get_hf_config_from_name_or_path(config):
    """
    Create a HF pretrained config based on user specified name or path
    """
    access_token = config.get("hf_access_token", None)
    hf_config = None
    # When multiple ranks query for the same HF config, it might error out, adding some retry logic
    for i in range(MAX_RETRY_TIME):
        try:
            hf_config = AutoConfig.from_pretrained(config.hf_model_name_or_path, token=access_token)
        except FileNotFoundError:
            if i == MAX_RETRY_TIME - 1:
                raise RuntimeError(f"Max retry timeout when fetching config from {config.hf_model_name_or_path}")
            _logger.warning(f"Retrying to get the config of {config.hf_model_name_or_path}")
    return hf_config


def _validate_model_type(model_type: Optional[str], hf_model_name_or_path: Optional[str]) -> None:
    if model_type is None and hf_model_name_or_path is None:
        msg = "model_type and hf_model_name_or_path are missing but at least one is required"
        _logger.error(msg)
        raise AttributeError(msg)

    # Enums support the `in` operator starting with Python 3.12
    if model_type is not None and model_type not in [key.value for key in ModelType]:
        msg = (
            f'Model "{model_type}" is not supported by SageMaker Model Parallel. Please ensure `use_smp_model` is False'
        )
        _logger.warning(msg)


def _validate_custom_recipe_extra_params(model: type[BaseModel]) -> None:
    """
    Available only when the model has a config of extra=allow
    https://docs.pydantic.dev/2.1/usage/models/#extra-fields
    """
    extra_fields = model.__pydantic_extra__

    if extra_fields and is_slurm_run():
        msg = f"The recipe received defines the following keys that are not pre-defined for this model: {extra_fields}"
        _logger.error(msg)
        raise AttributeError(msg)


def _validate_schema(cfg: DictConfig, extra="forbid") -> tuple[DictConfig, type[BaseModel]]:
    SchemaValidator = get_model_validator(use_smp_model=cfg.use_smp_model, extra=extra)
    config_dict = OmegaConf.to_container(cfg, resolve=True)
    try:
        validated_model = SchemaValidator.model_validate(config_dict)
        validated_model_dict = validated_model.model_dump()
        validated_config: DictConfig = OmegaConf.create(validated_model_dict)
        return validated_config, validated_model
    except ValidationError as err:
        _logger.error(err)
        exit()
    except Exception as err:
        _logger.error(err)
        exit()


def validate_config(extra="forbid"):
    def _validate_config(fn: _T) -> _T:
        @wraps(fn)
        def validations_wrapper(cfg: DictConfig, *args, **kwargs) -> DictConfig:
            """
            Execute all validations in this function
            """
            _validate_model_type(cfg.model.get("model_type", None), cfg.model.get("hf_model_name_or_path", None))
            merged_config, validated_model = _validate_schema(cfg, extra=extra)
            _validate_custom_recipe_extra_params(validated_model)

            return fn(merged_config, *args, **kwargs)

        return cast(_T, validations_wrapper)

    return _validate_config
