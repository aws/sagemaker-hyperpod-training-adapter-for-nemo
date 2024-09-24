from transformers import LlamaConfig

from sagemaker_nemo_adaptor.collections.model import SageMakerNLPBaseModel
from sagemaker_nemo_adaptor.utils.config_utils import get_hf_config_from_name_or_path


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
        else:
            model_config = LlamaConfig(
                **configurable_dict,
                hidden_act="silu",
                use_cache=False,
                pretraining_tp=1,
                tie_word_embeddings=False,
                rope_scaling=None,  # TODO Add support once Rubik is ready
            )
        return model_config
