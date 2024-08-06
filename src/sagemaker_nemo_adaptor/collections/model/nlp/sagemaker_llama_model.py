from transformers import LlamaConfig

from sagemaker_nemo_adaptor.collections.model import SageMakerNLPBaseModel


class SageMakerLlamaModel(SageMakerNLPBaseModel):
    """
    Lightning Model class for Llama
    """

    def get_model_config(self):
        """
        Get model config for Llama
        TODO: Implement Autoconfig in parent class, so Cx can init with only given a HF model name
        """
        model_config = LlamaConfig(
            vocab_size=self._cfg.vocab_size,
            hidden_size=self._cfg.hidden_width,
            intermediate_size=self._cfg.llama_intermediate_size,
            num_hidden_layers=self._cfg.num_layers,
            num_attention_heads=self._cfg.num_heads,
            num_key_value_heads=self._cfg.num_key_value_heads,
            hidden_act="silu",
            max_position_embeddings=self._cfg.max_context_width,
            initializer_range=self._cfg.initializer_range,
            rms_norm_eps=1e-5,
            use_cache=False,
            pretraining_tp=1,
            tie_word_embeddings=False,
            rope_scaling=None,
        )
        return model_config
