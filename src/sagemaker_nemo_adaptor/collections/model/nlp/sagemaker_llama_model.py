import transformers
from omegaconf import OmegaConf
from packaging import version as pversion
from transformers import LlamaConfig

from sagemaker_nemo_adaptor.collections.model import SageMakerNLPBaseModel
from sagemaker_nemo_adaptor.constants import CONFIG_MAPPING_HF_TO_RECIPE_ALIASES
from sagemaker_nemo_adaptor.utils.config_utils import get_hf_config_from_name_or_path
from sagemaker_nemo_adaptor.utils.log_utils import Logger

_logger = Logger().get_logger()


class SageMakerLlamaModel(SageMakerNLPBaseModel):
    """
    Lightning Model class for Llama
    """

    predefined_model = True

    def set_config_mapping_hf_to_recipe_aliases(self):
        config_map = CONFIG_MAPPING_HF_TO_RECIPE_ALIASES
        if OmegaConf.select(self._cfg, "rope_scaling.rope_type") == "llama3":
            if pversion.parse(transformers.__version__) < pversion.parse("4.44.2"):
                _logger.warning(
                    f"Rope scaling type 'llama3' is only supported for transformers >= 4.44.2, the current version is {pversion.parse(transformers.__version__)}"
                )
            else:
                config_map["rope_scaling"] = ["rope_scaling"]
        self._config_mapping_hf_to_recipe_aliases = config_map

    def get_model_config(self):
        """
        Get model config for Llama
        """
        configurable_dict = self._get_model_configurable_dict()
        if self._cfg.get("hf_model_name_or_path", None) is not None:
            model_config = get_hf_config_from_name_or_path(self._cfg)
            assert isinstance(
                model_config, LlamaConfig
            ), f"model_type is set to llama but hf_model_name_or_path is not the same model, getting {type(model_config)}"
            # Update the config based on user's input
            model_config.update(configurable_dict)
        else:
            model_config = LlamaConfig(
                **configurable_dict,
                hidden_act="silu",
                use_cache=False,
                pretraining_tp=1,
                tie_word_embeddings=False,
            )
        return model_config
