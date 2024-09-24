from transformers import GPTNeoXConfig

from sagemaker_nemo_adaptor.collections.model import SageMakerNLPBaseModel
from sagemaker_nemo_adaptor.utils.config_utils import get_hf_config_from_name_or_path


class SageMakerGPTNeoXModel(SageMakerNLPBaseModel):
    """
    Lightning Model class for Llama
    """

    predefined_model = True

    def get_model_config(self):
        """
        Get model config for Llama
        TODO: Implement Autoconfig in parent class, so Cx can init with only given a HF model name
        """
        configurable_dict = self._get_model_configurable_dict()
        # use intermediate_size as 4x hidden as default for gpt-like model
        if "hidden_size" in configurable_dict and "intermediate_size" not in configurable_dict:
            configurable_dict["intermediate_size"] = 4 * configurable_dict["hidden_size"]
        if self._cfg.get("hf_model_name_or_path", None) is not None:
            model_config = get_hf_config_from_name_or_path(self._cfg)
            assert isinstance(
                model_config, GPTNeoXConfig
            ), f"model_type is set to gpt-neox but hf_model_name_or_path is not the same model, getting {type(model_config)}"
            # Update the config based on user's input
            model_config.update(configurable_dict)
        else:
            model_config = GPTNeoXConfig(
                **configurable_dict,
                hidden_act="gelu",
                layer_norm_eps=1e-05,
                use_cache=False,
                tie_word_embeddings=False,
                use_parallel_residual=True,
                attention_dropout=0.0,
                hidden_dropout=0.0,
            )
        return model_config
