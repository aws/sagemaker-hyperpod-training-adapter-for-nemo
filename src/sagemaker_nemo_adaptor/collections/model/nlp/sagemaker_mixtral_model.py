from transformers import MixtralConfig

from sagemaker_nemo_adaptor.collections.model import SageMakerNLPBaseModel


class SageMakerMixtralModel(SageMakerNLPBaseModel):
    """
    Lightning Model class for Mixtral
    """

    def get_model_config(self):
        """
        Get model config for Mixtral
        TODO: Implement Autoconfig in parent class, so Cx can init with only given a HF model name
        """
        model_config = MixtralConfig(
            vocab_size=self._cfg.vocab_size,
            hidden_size=self._cfg.hidden_width,
            intermediate_size=self._cfg.intermediate_size,
            num_hidden_layers=self._cfg.num_layers,
            num_attention_heads=self._cfg.num_heads,
            num_key_value_heads=self._cfg.num_key_value_heads,
            hidden_act="silu",
            max_position_embeddings=self._cfg.max_context_width,
            initializer_range=self._cfg.initializer_range,
            rms_norm_eps=self._cfg.layernorm_epsilon,
            use_cache=False,
            pad_token_id=None,
            bos_token_id=1,
            eos_token_id=2,
            tie_word_embeddings=False,
            rope_theta=self._cfg.rope_theta,
            sliding_window=self._cfg.mixtral_sliding_window,
            attention_dropout=0.0,
            num_experts_per_tok=self._cfg.num_experts_per_tok,
            num_local_experts=self._cfg.num_local_experts,
            output_router_logits=False,
            router_aux_loss_coef=0.001,
            delayed_param=self._cfg.delayed_param,
        )
        return model_config
