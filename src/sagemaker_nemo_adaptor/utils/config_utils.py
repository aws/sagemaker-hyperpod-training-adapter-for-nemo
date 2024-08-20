from functools import wraps
from typing import Any, Callable, Optional, TypeVar, cast

from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, ValidationError

from sagemaker_nemo_adaptor.conf.config_schemas import HF_SCHEMAS, SMP_SCHEMAS
from sagemaker_nemo_adaptor.constants import ModelType
from sagemaker_nemo_adaptor.utils.log_utils import Logger

_logger = Logger().get_logger()

_T = TypeVar("_T", bound=Callable[..., Any])


def _get_model_validator(config: DictConfig) -> type[BaseModel]:
    SchemaValidator = (
        SMP_SCHEMAS.get(config.model.model_type) if config.use_smp else HF_SCHEMAS.get(config.model.model_type)
    )

    if not SchemaValidator:
        raise ValueError(f"Invalid model_type {config.model.model_type}")

    return SchemaValidator


def _save_merged_config(merged_config: DictConfig) -> None:
    # TODO: The dirname will point to a location in Hyperpod that the customer
    # will select. We will eventually have a pointer to that location.
    dirname = f"/fsx/users/julianhr/sagemaker-adaptor-nemo/scripts/conf"
    filename = "assembled_recipe.yaml"

    OmegaConf.save(merged_config, f"{dirname}/{filename}")
    _logger.info(f"Config saved to {dirname}/{filename}")


def _validate_model_type(model_type: Optional[str]) -> None:
    if model_type is None:
        msg = "model_type is missing but is required"
        _logger.error(msg)
        raise AttributeError(msg)

    # Enums support the `in` operator starting with Python 3.12
    if model_type not in [key.value for key in ModelType]:
        msg = f'Model "{model_type}" is not supported by SageMaker Model Parallel. Try setting `use_smp` to False'
        _logger.error(msg)
        raise AttributeError(msg)


def _validate_custom_recipe_extra_params(model: type[BaseModel]) -> None:
    """
    Available only when the model has a config of extra=allow
    https://docs.pydantic.dev/2.1/usage/models/#extra-fields
    """
    extra_fields = model.__pydantic_extra__

    if extra_fields:
        msg = f"The recipe received defines the following keys that are not pre-defined for this model: {extra_fields}"
        _logger.error(msg)
        raise AttributeError(msg)


def _validate_params_not_provided_by_custom_recipe(cfg: DictConfig, base_config) -> None:
    params_not_set = set(base_config.keys()) - set(cfg.keys())
    params_not_set.discard("internal")

    if params_not_set:
        msg = f"The following tunable parameters were not set on the custom recipe: {params_not_set}"
        _logger.info(msg)


def _validate_schema(cfg: DictConfig) -> tuple[DictConfig, type[BaseModel]]:
    SchemaValidator = _get_model_validator(cfg)
    config_dict = OmegaConf.to_container(cfg)

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


def validate_config(fn: _T) -> _T:
    @wraps(fn)
    def validations_wrapper(cfg: DictConfig, *args, **kwargs) -> DictConfig:
        """
        Execute all validations in this function
        """
        _validate_model_type(cfg.model.model_type)
        merged_config, validated_model = _validate_schema(cfg)
        _validate_custom_recipe_extra_params(validated_model)
        _validate_params_not_provided_by_custom_recipe(cfg, merged_config)
        # _save_merged_config(merged_config)

        return fn(merged_config, *args, **kwargs)

    return cast(_T, validations_wrapper)
