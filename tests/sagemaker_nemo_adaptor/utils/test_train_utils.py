from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from sagemaker_nemo_adaptor.utils.train_utils import (
    apply_activation_checkpoint,
    apply_activation_checkpoint_moe,
    compute_tflops,
    get_batch_for_cp_rank,
)


class DummyTransformerLayer(nn.Module):
    pass


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([DummyTransformerLayer() for _ in range(3)])


@pytest.fixture
def dummy_model():
    return DummyModel()


def test_apply_activation_checkpoint(dummy_model):
    with patch("sagemaker_nemo_adaptor.utils.train_utils.get_transformer_layer") as mock_get_layer:
        mock_get_layer.return_value = DummyTransformerLayer
        with patch(
            "torch.distributed.algorithms._checkpoint.checkpoint_wrapper.apply_activation_checkpointing"
        ) as mock_apply:
            apply_activation_checkpoint(dummy_model, "gpt", use_smp_model=True)
            mock_apply.assert_called_once()


def test_apply_activation_checkpoint_moe(dummy_model):
    with patch(
        "torch.distributed.algorithms._checkpoint.checkpoint_wrapper.apply_activation_checkpointing"
    ) as mock_apply:
        apply_activation_checkpoint_moe(dummy_model)
        assert mock_apply.call_count == 2  # Called for both attention and MoE


@pytest.mark.parametrize(
    "cp_size,cp_rank,input_shape,expected_shape",
    [
        (1, 0, (8, 512, 1024), (8, 512, 1024)),
        (2, 0, (8, 512, 1024), (8, 256, 1024)),
        (2, 1, (8, 512, 1024), (8, 256, 1024)),
    ],
)
def test_get_batch_for_cp_rank(cp_size, cp_rank, input_shape, expected_shape):
    mock_state = MagicMock()
    mock_state.cp_size = cp_size
    mock_state.cp_rank = cp_rank

    with patch("sagemaker_nemo_adaptor.utils.train_utils.torch.sagemaker.state", mock_state):
        input_tensor = torch.randn(input_shape)
        result = get_batch_for_cp_rank((input_tensor,))
        assert result[0].shape == expected_shape


def test_compute_tflops():
    cfg = {"moe": 0, "num_experts_per_tok": 1, "max_context_width": 1024}
    model_config = MagicMock(
        hidden_size=1024,
        num_attention_heads=16,
        num_key_value_heads=16,
        num_hidden_layers=24,
        intermediate_size=4096,
        vocab_size=50000,
    )
    sample_processed = 1000
    step_time = 1.0
    world_size = 8

    tflops = compute_tflops(cfg, model_config, sample_processed, step_time, world_size)
    assert isinstance(tflops, float)
    assert tflops > 0
