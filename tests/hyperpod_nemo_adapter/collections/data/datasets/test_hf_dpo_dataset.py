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

from hyperpod_nemo_adapter.collections.data.datasets.hf_dpo_dataset import (
    HuggingFaceDPODataset,
)
from tests.test_utils import create_temp_directory

logger = logging.getLogger(__name__)


@pytest.fixture
def mock_dpo_dataset():
    data = {
        "prompt_input_ids": [[10, 11, 12], [20, 21, 22]],
        "chosen_input_ids": [[13, 14, 15], [23, 24, 25]],
        "rejected_input_ids": [[16, 17, 18], [26, 27, 28]],
    }
    return Dataset.from_dict(data)


def assert_dpo_dataset(dataset):
    assert len(dataset) == 2

    # Check the first sample.
    sample = dataset[0]
    assert torch.equal(sample["prompt_input_ids"], torch.tensor([10, 11, 12]))
    assert torch.equal(sample["chosen_input_ids"], torch.tensor([13, 14, 15]))
    assert torch.equal(sample["rejected_input_ids"], torch.tensor([16, 17, 18]))

    # Check the second sample.
    sample = dataset[1]
    assert torch.equal(sample["prompt_input_ids"], torch.tensor([20, 21, 22]))
    assert torch.equal(sample["chosen_input_ids"], torch.tensor([23, 24, 25]))
    assert torch.equal(sample["rejected_input_ids"], torch.tensor([26, 27, 28]))


def test_hugging_face_dpo_dataset(mock_dpo_dataset):
    temp_dir = create_temp_directory()
    logger.info("Running HuggingFace DPO dataset test by creating datasets in {} directory".format(temp_dir))

    # Test ARROW format.
    arrow_dir = os.path.join(temp_dir, "arrow")
    os.makedirs(arrow_dir)
    mock_dpo_dataset.save_to_disk(arrow_dir)
    dataset = HuggingFaceDPODataset(arrow_dir)
    assert_dpo_dataset(dataset)

    # Test JSON format.
    json_dir = os.path.join(temp_dir, "json")
    os.makedirs(json_dir)
    mock_dpo_dataset.to_json(os.path.join(json_dir, "data.json"))
    dataset = HuggingFaceDPODataset(json_dir)
    assert_dpo_dataset(dataset)

    # Test unsupported format.
    unsupported_dir = os.path.join(temp_dir, "unsupported")
    os.makedirs(unsupported_dir)
    mock_dpo_dataset.to_json(os.path.join(unsupported_dir, "data.txt"))
    with pytest.raises(NotImplementedError):
        HuggingFaceDPODataset(unsupported_dir)
