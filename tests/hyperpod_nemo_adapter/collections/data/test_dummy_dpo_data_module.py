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

import pytest

from hyperpod_nemo_adapter.collections.data import DummyDPODataModule


class TestDummyDPODataModule:
    @pytest.fixture
    def dummy_dpo_data_module(self, mocker):
        cfg = mocker.MagicMock()
        cfg.model.vocab_size = 10
        cfg.model.max_context_width = 20
        trainer = mocker.MagicMock()

        dummy_dpo_data_module = DummyDPODataModule(trainer=trainer, cfg=cfg)
        return dummy_dpo_data_module

    def test_val_dataloader(self, dummy_dpo_data_module):
        val_dataloader = dummy_dpo_data_module.val_dataloader()
        assert val_dataloader is None

    def test_get_batch(self, dummy_dpo_data_module):
        data = {
            "prompt_input_ids": [1, 2],
            "prompt_attention_mask": [1, 1],
            "chosen_input_ids": [3, 4],
            "chosen_attention_mask": [1, 1],
            "rejected_input_ids": [5, 6],
            "rejected_attention_mask": [1, 1],
        }
        batch = dummy_dpo_data_module.get_batch(data)
        assert batch == (
            [1, 2],
            [1, 1],
            [3, 4],
            [1, 1],
            [5, 6],
            [1, 1],
        )

    def test_get_val_batch(self, dummy_dpo_data_module):
        data = {
            "prompt_input_ids": [1, 2],
            "prompt_attention_mask": [1, 1],
            "chosen_input_ids": [3, 4],
            "chosen_attention_mask": [1, 1],
            "rejected_input_ids": [5, 6],
            "rejected_attention_mask": [1, 1],
        }
        batch = dummy_dpo_data_module.get_val_batch(data)
        assert batch == (
            [1, 2],
            [1, 1],
            [3, 4],
            [1, 1],
            [5, 6],
            [1, 1],
        )
