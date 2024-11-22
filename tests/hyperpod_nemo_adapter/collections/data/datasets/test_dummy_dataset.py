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

from hyperpod_nemo_adapter.collections.data.datasets import DummyDataset


@pytest.fixture
def dummy_dataset():
    return DummyDataset()


def test_init(dummy_dataset):
    assert dummy_dataset.vocab_size == 1024
    assert dummy_dataset.seqlen == 2048
    assert dummy_dataset.length == 100000
    assert torch.all(dummy_dataset.mask == torch.ones((2048,)))


def test_getitem(dummy_dataset):
    item = dummy_dataset[0]
    assert len(item) == 2
    assert item[0].shape == (2048,)
    assert item[0].dtype == torch.long
    assert torch.all(item[1] == torch.ones((2048,)))


def test_len(dummy_dataset):
    assert len(dummy_dataset) == 100000


def test_data_type_bert():
    with pytest.raises(NotImplementedError):
        DummyDataset(data_type="bert")
