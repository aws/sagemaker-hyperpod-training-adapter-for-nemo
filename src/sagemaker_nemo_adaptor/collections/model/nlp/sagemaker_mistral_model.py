from transformers import MistralConfig

from sagemaker_nemo_adaptor.collections.model import SageMakerNLPBaseModel
from sagemaker_nemo_adaptor.utils.config_utils import get_hf_config_from_name_or_path


class SageMakerMistralModel(SageMakerNLPBaseModel):
    """
    Lightning Model class for Mistral
    """

    predefined_model = True

    def get_model_config(self):
        """
        Get model config for Mistral
        """
        configurable_dict = self._get_model_configurable_dict()
        if self._cfg.get("hf_model_name_or_path", None) is not None:
            model_config = get_hf_config_from_name_or_path(self._cfg)
            assert isinstance(
                model_config, MistralConfig
            ), f"model_type is set to mistral but hf_model_name_or_path is not the same model, getting {type(model_config)}"
            # Update the config based on user's input
            model_config.update(configurable_dict)
        else:
            model_config = MistralConfig(
                **configurable_dict,
                hidden_act="silu",
                use_cache=False,
                pad_token_id=None,
                bos_token_id=1,
                eos_token_id=2,
                tie_word_embeddings=False,
                attention_dropout=0.0,
            )
        return model_config
