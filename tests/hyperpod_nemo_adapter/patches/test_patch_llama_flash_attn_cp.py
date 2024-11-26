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

import unittest
from unittest.mock import MagicMock, patch

from transformers import LlamaConfig

from hyperpod_nemo_adapter.patches.patch_llama_flash_attn_cp import (
    LlamaFlashAttention2,
    apply_patch,
    patched_get_extra_state,
    patched_LFA2__init__,
    patched_LFA2_forward,
    te,
    unapply_patch,
)
from tests.test_utils import skip_if_lt_x_gpu


class TestPatchingMechanism(unittest.TestCase):
    def setUp(self):
        self.original_get_extra_state = te.attention.DotProductAttention.get_extra_state
        self.original_LFA2__init__ = LlamaFlashAttention2.__init__
        self.original_LFA2_forward = LlamaFlashAttention2.forward

    def tearDown(self):
        te.attention.DotProductAttention.get_extra_state = self.original_get_extra_state
        LlamaFlashAttention2.__init__ = self.original_LFA2__init__
        LlamaFlashAttention2.forward = self.original_LFA2_forward

    def test_apply_patch(self):
        apply_patch()
        self.assertEqual(te.attention.DotProductAttention.get_extra_state, patched_get_extra_state)
        self.assertEqual(LlamaFlashAttention2.__init__, patched_LFA2__init__)
        self.assertEqual(LlamaFlashAttention2.forward, patched_LFA2_forward)

    def test_unapply_patch(self):
        apply_patch()
        unapply_patch()
        self.assertEqual(te.attention.DotProductAttention.get_extra_state, self.original_get_extra_state)
        self.assertEqual(LlamaFlashAttention2.__init__, self.original_LFA2__init__)
        self.assertEqual(LlamaFlashAttention2.forward, self.original_LFA2_forward)


class TestPatchedFunctions(unittest.TestCase):
    def test_patched_get_extra_state(self):
        apply_patch()

        class MockBase:
            def get_extra_state(self):
                return MagicMock()

        class MockDotProductAttention(MockBase):
            pass

        mock_self = MockDotProductAttention()

        result = patched_get_extra_state(mock_self)

        self.assertIsNone(result.device)

    @skip_if_lt_x_gpu(1)
    @patch("torch.sagemaker.state.get_rng_state_tracker")
    def test_patched_LFA2__init__(self, mock_get_rng_state_tracker):
        config = LlamaConfig(num_key_value_heads=4, num_attention_heads=8, hidden_size=32)
        lfa2 = LlamaFlashAttention2(config=config)
        patched_LFA2__init__(lfa2, config=config, layer_idx=0)
        self.assertTrue(hasattr(lfa2, "core_attention"))
        self.assertIsInstance(lfa2.core_attention, te.attention.DotProductAttention)
