from transformers import MistralConfig

from sagemaker_nemo_adaptor.collections.model import SageMakerNLPBaseModel


class SageMakerMistralModel(SageMakerNLPBaseModel):
    """
    Lightning Model class for Mistral
    """

    def get_model_config(self):
        """
        Get model config for Mistral
        TODO: Implement Autoconfig in parent class, so Cx can init with only given a HF model name
        """
        model_config = MistralConfig(
            vocab_size=self._cfg.vocab_size,
            hidden_size=self._cfg.hidden_width,
            intermediate_size=self._cfg.intermediate_size,
            num_hidden_layers=self._cfg.num_layers,
            num_attention_heads=self._cfg.num_heads,
            num_key_value_heads=self._cfg.num_key_value_heads,
            hidden_act="silu",
            max_position_embeddings=self._cfg.max_context_width,
            initializer_range=self._cfg.initializer_range,
            rms_norm_eps=1e-6,
            use_cache=False,
            pad_token_id=None,
            bos_token_id=1,
            eos_token_id=2,
            tie_word_embeddings=False,
            rope_theta=10000.0,
            sliding_window=self._cfg.mistral_sliding_window,
            attention_dropout=0.0,
        )
        return model_config
