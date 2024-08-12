from functools import wraps
from typing import Any, Callable, TypeVar, cast

from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf

_T = TypeVar("_T", bound=Callable[..., Any])


def _get_base_config() -> DictConfig:
    # must re-initialize Hydra so the base conf file is read from src/conf,
    # otherwise Hydra raises exception
    GlobalHydra.instance().clear()

    with initialize(version_base=None, config_path="../conf"):
        cfg = compose(config_name="base_config")
        return cfg

def _merge_configs(recipe_config: DictConfig, base_config: DictConfig) -> DictConfig:
    oc = OmegaConf.create({**base_config, **recipe_config})
    return oc

def _validate_recipe(config: DictConfig) -> None:
    # Rubik optimizations
    if config.use_smp:
        # this will likely use a strategy pattern, using a switch statement
        # for illustrative purposes
        match config.model.model_type:
            case "llama_v3":
                # here will go all validations for Llama 3
                # For example, ensure no keys in the customized recipe are missing on the base config
                pass
            case _:
                raise ValueError("Invalid model type")

    # Directly from Hugging Face
    else:
        pass


def validate_config(fn: _T) -> _T:
    @wraps(fn)

    def validations_wrapper(cfg: DictConfig, *args, **kwargs) -> DictConfig:
        """
        Execute all validations in this function
        """
        base_config: DictConfig = _get_base_config()
        config: DictConfig = _merge_configs(cfg,  base_config)

        # config hasn't yet been verified
        config.internal.config_verified = False
        _validate_recipe(config)

        # after all validations has passed
        config.internal.config_verified = True

        return fn(config, *args, **kwargs)

    return cast(_T, validations_wrapper)
