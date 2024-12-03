# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.

from unittest.mock import patch

import pytest
import torch
from torch.distributed.fsdp import BackwardPrefetch, MixedPrecision, ShardingStrategy

from hyperpod_nemo_adapter.utils.fsdp_utils import (
    get_auto_wrap_policy,
    get_backward_fetch_policy,
    get_sharding_strategy,
    get_transformer_layer,
    set_mixed_precision_recipe,
)


def test_get_sharding_strategy():
    # Test for valid strategy
    assert get_sharding_strategy("FULL_SHARD") == ShardingStrategy.FULL_SHARD
    assert get_sharding_strategy("hybrid_shard") == ShardingStrategy.HYBRID_SHARD
    assert get_sharding_strategy("no_shard") == ShardingStrategy.NO_SHARD

    # Test for invalid strategy
    with pytest.raises(AttributeError):
        get_sharding_strategy("invalid")


def test_get_backward_fetch_policy():
    # Test for valid policy
    assert get_backward_fetch_policy("BACKWARD_PRE") == BackwardPrefetch.BACKWARD_PRE
    assert get_backward_fetch_policy("backward_post") == BackwardPrefetch.BACKWARD_POST

    # Test for invalid policy
    with pytest.raises(AttributeError):
        get_backward_fetch_policy("invalid")


def test_get_auto_wrap_policy():
    # Test for valid policies
    assert get_auto_wrap_policy("transformer_auto_wrap_policy") is not None
    assert get_auto_wrap_policy("size_based_auto_wrap_policy") is not None

    # Test for invalid policy
    with pytest.raises(NotImplementedError):
        get_auto_wrap_policy("invalid")


def test_get_transformer_layer():
    # Test for valid model types
    # Unit tests are currently run in smpv2 container hence below code paths cannot be tested
    assert get_transformer_layer("llama_v2").__name__ == "LlamaDecoderLayer"
    assert get_transformer_layer("llama_v3").__name__ == "LlamaDecoderLayer"
    assert get_transformer_layer("mistral").__name__ == "MistralDecoderLayer"
    assert get_transformer_layer("mixtral").__name__ == "MixtralDecoderLayer"
    # Test for SMP with PT 2.2.0 and above
    with patch("torch.__version__", "2.2.0"):
        assert get_transformer_layer("", use_smp_model=True) is not None

    # Test for SMP with PT 2.1.0 and below
    with patch("torch.__version__", "2.1.0"):
        assert get_transformer_layer("", use_smp_model=True) is not None

    # Test for invalid model type
    with pytest.raises(Exception):
        get_transformer_layer("invalid")


def test_set_mixed_precision_recipe():
    # Test for valid precision
    assert set_mixed_precision_recipe(16) == MixedPrecision(
        param_dtype=torch.float16, reduce_dtype=torch.float16, buffer_dtype=torch.float32
    )
    assert set_mixed_precision_recipe("bf16") == MixedPrecision(
        param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16, buffer_dtype=torch.float32
    )
    assert set_mixed_precision_recipe(32) == MixedPrecision(
        param_dtype=torch.float, reduce_dtype=torch.float, buffer_dtype=torch.float32
    )

    # Test for non smp cases
    assert set_mixed_precision_recipe(16, None, None, False) == MixedPrecision(
        param_dtype=torch.float16, reduce_dtype=torch.float16, buffer_dtype=torch.float16
    )
    assert set_mixed_precision_recipe("bf16", None, None, False) == MixedPrecision(
        param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16, buffer_dtype=torch.bfloat16
    )
    assert set_mixed_precision_recipe(32, None, None, False) == MixedPrecision(
        param_dtype=torch.float, reduce_dtype=torch.float, buffer_dtype=torch.float
    )

    # Test for custom reduce_dtype and buffer_dtype
    assert set_mixed_precision_recipe(16, 16, 16, False) == MixedPrecision(
        param_dtype=torch.float16, reduce_dtype=torch.float16, buffer_dtype=torch.float16
    )
    assert set_mixed_precision_recipe("bf16", 16, 16, False) == MixedPrecision(
        param_dtype=torch.bfloat16, reduce_dtype=torch.float16, buffer_dtype=torch.float16
    )
    assert set_mixed_precision_recipe(32, 16, 16, False) == MixedPrecision(
        param_dtype=torch.float, reduce_dtype=torch.float16, buffer_dtype=torch.float16
    )
