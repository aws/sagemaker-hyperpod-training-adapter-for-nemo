from transformers import GPTNeoXConfig

from sagemaker_nemo_adaptor.collections.model import SageMakerBaseModel


class SageMakerLlamaModel(SageMakerBaseModel):
    """
    Lightning Model class for Llama
    """

    def get_model_config(self):
        """
        Get model config for Llama
        TODO: Implement Autoconfig in parent class, so Cx can init with only given a HF model name
        """
        model_config = GPTNeoXConfig(
            vocab_size=self.cfg.vocab_size,
            hidden_size=self.cfg.hidden_width,
            num_hidden_layers=self.cfg.num_layers,
            num_attention_heads=self.cfg.num_heads,
            hidden_act="gelu",
            intermediate_size=4 * self.cfg.hidden_width,
            rotary_pct=self.cfg.rotary_pct,
            rotary_emb_base=self.cfg.rotary_emb_base,
            max_position_embeddings=self.cfg.max_context_width,
            layer_norm_eps=1e-05,
            initializer_range=self.cfg.initializer_range,
            use_cache=False,
            tie_word_embeddings=False,
            use_parallel_residual=True,
            attention_dropout=0.0,
            hidden_dropout=0.0,
        )
        return model_config

    def configure_flash_attn(self):
        """
        Configure flash attention for GPT Neox
        """

        def new_attn(
            attn, q, k, v, attention_mask=None, head_mask=None
        ):  # TODO: check with rubik about the correctness of thi func
            del attention_mask
            del head_mask
            attn_weights = None
            return (
                attn.flashmod((q, k, v), causal=True, cast_dtype=dtype, layout=layout),
                attn_weights,
            )

        layout = "b s h d"
        layers = self.model.gpt_neox.layer
        attn_name = "attention"

        from torch.sagemaker.nn.attn import FlashSelfAttention

        for layer in layers:
            getattr(layer, attn_name).flashmod = FlashSelfAttention(attention_dropout_prob=0.0)
            getattr(layer, attn_name)._attn = functools.partial(new_attn, getattr(layer, attn_name))
