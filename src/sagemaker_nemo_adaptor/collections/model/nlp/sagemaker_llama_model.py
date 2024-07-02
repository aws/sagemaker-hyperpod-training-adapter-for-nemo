from torch.sagemaker.nn.huggingface.llama_flashattn import LlamaFlashAttention
from transformers import LlamaConfig

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
        model_config = LlamaConfig(
            vocab_size=self.cfg.vocab_size,
            hidden_size=self.cfg.hidden_width,
            intermediate_size=self.cfg.llama_intermediate_size,
            num_hidden_layers=self.cfg.num_layers,
            num_attention_heads=self.cfg.num_heads,
            num_key_value_heads=self.cfg.num_key_value_heads,
            hidden_act="silu",
            max_position_embeddings=self.cfg.max_context_width,
            initializer_range=self.cfg.initializer_range,
            rms_norm_eps=1e-5,
            use_cache=False,
            pretraining_tp=1,
            tie_word_embeddings=False,
            rope_scaling=None,
        )
        return model_config

    def configure_flash_attn(self):
        """
        Configure flash attention for Llama
        """
        layout = "b s h d"
        layers = self.model.model.layers
        attn_name = "self_attn"

        flash_attn_class = LlamaFlashAttention
        for layer in layers:
            prev_layer = getattr(layer, attn_name)
            setattr(layer, attn_name, flash_attn_class(model.config))
            attn_layer = getattr(layer, attn_name)
            attn_layer.pretraining_tp = model.config.pretraining_tp
            with torch.no_grad():
                attn_layer.q_proj.weight.copy_(prev_layer.q_proj.weight)
                attn_layer.k_proj.weight.copy_(prev_layer.k_proj.weight)
                attn_layer.v_proj.weight.copy_(prev_layer.v_proj.weight)
                attn_layer.o_proj.weight.copy_(prev_layer.o_proj.weight)
