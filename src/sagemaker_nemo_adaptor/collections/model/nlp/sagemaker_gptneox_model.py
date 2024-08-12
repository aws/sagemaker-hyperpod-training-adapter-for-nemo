from transformers import GPTNeoXConfig

from sagemaker_nemo_adaptor.collections.model import SageMakerNLPBaseModel


class SageMakerGPTNeoXModel(SageMakerNLPBaseModel):
    """
    Lightning Model class for Llama
    """

    def get_model_config(self):
        """
        Get model config for Llama
        TODO: Implement Autoconfig in parent class, so Cx can init with only given a HF model name
        """
        model_config = GPTNeoXConfig(
            vocab_size=self._cfg.vocab_size,
            hidden_size=self._cfg.hidden_width,
            num_hidden_layers=self._cfg.num_layers,
            num_attention_heads=self._cfg.num_heads,
            hidden_act="gelu",
            intermediate_size=4 * self._cfg.hidden_width,
            rotary_pct=self._cfg.rotary_pct,
            rotary_emb_base=self._cfg.rotary_emb_base,
            max_position_embeddings=self._cfg.max_context_width,
            layer_norm_eps=1e-05,
            initializer_range=self._cfg.initializer_range,
            use_cache=False,
            tie_word_embeddings=False,
            use_parallel_residual=True,
            attention_dropout=0.0,
            hidden_dropout=0.0,
        )
        return model_config
