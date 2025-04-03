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
import torch

from hyperpod_nemo_adapter.collections.data.datasets import DummyDPODataset


@pytest.fixture
def dummy_dpo_dataset():
    return DummyDPODataset()


def test_init(dummy_dpo_dataset):
    assert dummy_dpo_dataset.vocab_size == 1024
    assert dummy_dpo_dataset.seqlen == 2048
    assert dummy_dpo_dataset.length == 100000
    assert dummy_dpo_dataset.prompt_length == 1024
    assert dummy_dpo_dataset.completion_length == 512


def test_getitem(dummy_dpo_dataset):
    item = dummy_dpo_dataset[0]

    assert set(item.keys()) == {
        "prompt_input_ids",
        "chosen_input_ids",
        "rejected_input_ids",
        "prompt_attention_mask",
        "chosen_attention_mask",
        "rejected_attention_mask",
    }

    assert item["prompt_input_ids"].shape == (1024,)
    assert item["chosen_input_ids"].shape == (512,)
    assert item["rejected_input_ids"].shape == (512,)

    assert item["prompt_input_ids"].dtype == torch.long
    assert item["chosen_input_ids"].dtype == torch.long
    assert item["rejected_input_ids"].dtype == torch.long

    assert torch.all(item["prompt_attention_mask"] == 1)
    assert torch.all(item["chosen_attention_mask"] == 1)
    assert torch.all(item["rejected_attention_mask"] == 1)


def test_len(dummy_dpo_dataset):
    assert len(dummy_dpo_dataset) == 100000


def test_data_type_bert():
    with pytest.raises(NotImplementedError):
        DummyDPODataset(data_type="bert")
