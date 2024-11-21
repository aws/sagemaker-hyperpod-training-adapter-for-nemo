from transformers import MixtralConfig

from hyperpod_nemo_adapter.collections.model import SageMakerNLPBaseModel
from hyperpod_nemo_adapter.utils.config_utils import get_hf_config_from_name_or_path


class SageMakerMixtralModel(SageMakerNLPBaseModel):
    """
    Lightning Model class for Mixtral
    """

    predefined_model = True

    def get_model_config(self):
        """
        Get model config for Mixtral
        """
        configurable_dict = self._get_model_configurable_dict()
        if self._cfg.get("hf_model_name_or_path", None) is not None:
            model_config = get_hf_config_from_name_or_path(self._cfg)
            assert isinstance(
                model_config, MixtralConfig
            ), f"model_type is set to mixtral but hf_model_name_or_path is not the same model, getting {type(model_config)}"
            # Update the config based on user's input
            model_config.update(configurable_dict)
        else:
            model_config = MixtralConfig(
                **configurable_dict,
                hidden_act="silu",
                use_cache=False,
                pad_token_id=None,
                bos_token_id=1,
                eos_token_id=2,
                tie_word_embeddings=False,
                attention_dropout=0.0,
                output_router_logits=False,
                router_aux_loss_coef=0.001,
            )
        return model_config
