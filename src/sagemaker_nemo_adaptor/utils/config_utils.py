from functools import wraps
from typing import Any, Callable, Optional, TypeVar, cast

from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, ValidationError

from sagemaker_nemo_adaptor.conf.config_schemas import HF_SCHEMAS, SMP_SCHEMAS
from sagemaker_nemo_adaptor.constants import ModelType, SageMakerParallelParams
from sagemaker_nemo_adaptor.utils.log_utils import Logger

_logger = Logger().get_logger()

_T = TypeVar("_T", bound=Callable[..., Any])


def _get_merged_configs(cfg: DictConfig, base_config: DictConfig) -> DictConfig:
    """
    Merge the base config with the recipe config.
    The recipe config takes precedence over the base config.
    """
    merged_config = OmegaConf.create({**base_config, **cfg})
    _logger.info("Assembled recipe:", merged_config)
    return merged_config


def _get_base_config(cfg: DictConfig) -> DictConfig:
    # must re-initialize Hydra so the base conf file is read from src/conf,
    # otherwise Hydra raises exception
    GlobalHydra.instance().clear()

    with initialize(version_base=None, config_path="../conf"):
        config_filename = f"{cfg.model.model_type}_base_config"
        cfg = compose(config_name=config_filename)

        # remove keys that are tunable when SMP is enabled
        if not cfg.get("use_smp"):
            for key in SageMakerParallelParams:
                if key.value in cfg.model:
                    del cfg.model[key.value]

        return cfg


def _get_model_validator(config: DictConfig) -> BaseModel:
    schema = SMP_SCHEMAS.get(config.model.model_type) if config.use_smp else HF_SCHEMAS.get(config.model.model_type)

    if not schema:
        raise ValueError(f"Invalid model_type {config.model.model_type}")

    return schema


def _save_merged_config(merged_config: DictConfig) -> None:
    # TODO: This could change to give the customer a chance to select a different path and filename
    dirname = "/fsx/users/julianhr/sagemaker-adaptor-nemo/scripts/conf"
    filename = "assembled_recipe.yaml"

    OmegaConf.save(merged_config, f"{dirname}/{filename}")
    print(f"Config saved to {dirname}/{filename}")


def _validate_model_type(model_type: Optional[str]) -> None:
    if model_type is None:
        msg = "model_type is missing but is required"
        _logger.error(msg)
        raise AttributeError("model_type is missing but is required")

    # Enums support the `in` operator starting with Python 3.12
    if model_type not in [key.value for key in ModelType]:
        msg = f'Model "{model_type}" is not supported by SageMaker Model Parallel. Try setting `use_smp` to False'
        _logger.error(msg)
        raise AttributeError(msg)


def _validate_custom_recipe_params(cfg: DictConfig, base_config) -> None:
    unsupported_params = set(cfg.keys()) - set(base_config.keys())

    if unsupported_params:
        msg = (
            f"The recipe received defines the following keys that are not supported by this model: {unsupported_params}"
        )
        _logger.error(msg)
        raise AttributeError(msg)


def _validate_params_not_provided_by_custom_recipe(cfg: DictConfig, base_config) -> None:
    params_not_set = set(base_config.keys()) - set(cfg.keys())

    if params_not_set:
        msg = f"The following tunable parameters were not set in the custom recipe: {params_not_set}"
        _logger.info(msg)


def _validate_value_ranges(cfg: DictConfig):
    """
    Placeholder for future implementation.
    If we use Pydantic, this function would no longer be needed.
    """


def _validate_schema(config: DictConfig) -> None:
    dict_config = OmegaConf.to_container(config)
    model_validator = _get_model_validator(config)

    try:
        model_validator.model_validate(dict_config)
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
        base_config: DictConfig = _get_base_config(cfg)
        _validate_custom_recipe_params(cfg, base_config)
        _validate_params_not_provided_by_custom_recipe(cfg, base_config)
        _validate_value_ranges(cfg)
        merged_config = _get_merged_configs(cfg, base_config)

        _save_merged_config(merged_config)

        # config hasn't yet been verified
        merged_config.internal.cfg_verified = False

        _validate_schema(merged_config)

        # after all validations has passed
        merged_config.internal.cfg_verified = True

        return fn(merged_config, *args, **kwargs)

    return cast(_T, validations_wrapper)
