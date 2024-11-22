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

import logging
import os

import pytest
import torch
from datasets import Dataset

from hyperpod_nemo_adapter.collections.data.datasets import (
    HuggingFacePretrainingDataset,
)
from tests.test_utils import create_temp_directory

logger = logging.getLogger(__name__)


@pytest.fixture
def mock_dataset():
    data = {
        "input_ids": [[1, 2, 3], [4, 5, 6]],
        "attention_mask": [[1, 1, 1], [1, 1, 0]],
        "labels": [0, 1],
    }
    return Dataset.from_dict(data)


def assert_dataset(dataset):
    assert len(dataset) == 2
    iids, attns, labels = dataset[0]
    assert torch.equal(iids, torch.tensor([1, 2, 3]))
    assert torch.equal(attns, torch.tensor([1, 1, 1]))
    assert torch.equal(labels, torch.tensor(0))
    iids, attns, labels = dataset[1]
    assert torch.equal(iids, torch.tensor([4, 5, 6]))
    assert torch.equal(attns, torch.tensor([1, 1, 0]))
    assert torch.equal(labels, torch.tensor(1))


def test_hugging_face_pretraining_dataset(mock_dataset):
    temp_dir = create_temp_directory()
    logger.info("Running hugging face pretraining test by creating datasets in {} directory".format(temp_dir))

    # Test ARROW format
    arrow_dir = os.path.join(temp_dir, "arrow")
    os.makedirs(arrow_dir)
    mock_dataset.save_to_disk(arrow_dir)
    dataset = HuggingFacePretrainingDataset(arrow_dir)
    assert_dataset(dataset)

    # Test JSON format
    json_dir = os.path.join(temp_dir, "json")
    os.makedirs(json_dir)
    mock_dataset.to_json(os.path.join(json_dir, "data.json"))
    dataset = HuggingFacePretrainingDataset(json_dir)
    assert_dataset(dataset)

    # Test unsupported format
    unsupported_dir = os.path.join(temp_dir, "unsupported")
    os.makedirs(unsupported_dir)
    mock_dataset.to_json(os.path.join(unsupported_dir, "data.txt"))
    with pytest.raises(NotImplementedError):
        HuggingFacePretrainingDataset(unsupported_dir)
