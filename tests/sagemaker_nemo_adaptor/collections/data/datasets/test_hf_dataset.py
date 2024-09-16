import logging
import os

import pytest
import torch
from datasets import Dataset

from sagemaker_nemo_adaptor.collections.data.datasets import (
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

    # TODO Test JSONGZ format
    # Below code is failing datasets.exceptions.DataFilesNotFoundError: No (supported) data files found
    # We need to enable this test after Fix PR https://github.com/aws/private-sagemaker-training-adapter-for-nemo-staging/pull/57 is merged
    # jsongz_dir = os.path.join(temp_dir, "jsongz")
    # os.makedirs(jsongz_dir)
    # file_path = os.path.join(jsongz_dir, "data.json.gz")
    # with gzip.open(file_path, 'wt') as f:
    #     json.dump(mock_dataset.to_dict(), f)
    # dataset = HuggingFacePretrainingDataset(jsongz_dir)
    # assert_dataset(dataset)
