import transformers
from packaging import version as pversion
from transformers import LlamaConfig

from sagemaker_nemo_adaptor.collections.model import SageMakerNLPBaseModel
from sagemaker_nemo_adaptor.utils.config_utils import get_hf_config_from_name_or_path
from sagemaker_nemo_adaptor.utils.log_utils import Logger

_logger = Logger().get_logger()


class SageMakerLlamaModel(SageMakerNLPBaseModel):
    """
    Lightning Model class for Llama
    """

    predefined_model = True

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
            model_config.rope_scaling = None  # TODO Add support once Rubik is ready
        else:
            rope_scaling = None
            if self._cfg.rope_scaling_type == "llama3":
                if pversion.parse(transformers.__version__) < pversion.parse("4.44.2"):
                    _logger.warning(
                        f"Rope scaling type 'llama3' is only supported for transformers >= 4.44.2, the current version is {pversion.parse(transformers.__version__)}"
                    )
                else:
                    rope_scaling = {
                        "rope_type": self._cfg.rope_scaling_type,
                        "factor": self._cfg.rope_scaling_factor,
                        "high_freq_factor": self._cfg.rope_scaling_high_freq_factor,
                        "low_freq_factor": self._cfg.rope_scaling_low_freq_factor,
                        "original_max_position_embeddings": self._cfg.rope_scaling_original_max_position_embeddings,
                    }
            model_config = LlamaConfig(
                **configurable_dict,
                hidden_act="silu",
                use_cache=False,
                pretraining_tp=1,
                tie_word_embeddings=False,
                rope_scaling=rope_scaling,
            )
        return model_config
