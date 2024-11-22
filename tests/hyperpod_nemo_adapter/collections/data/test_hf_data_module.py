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

from unittest.mock import MagicMock

import pytest
from omegaconf import OmegaConf
from pytorch_lightning import Trainer

from hyperpod_nemo_adapter.collections.data import HuggingFaceDataModule


@pytest.fixture
def mock_cfg():
    return OmegaConf.create({"model": {"data": {"train_dir": "mock/train/dir", "val_dir": "mock/val/dir"}}})


@pytest.fixture
def mock_trainer():
    return MagicMock(Trainer)


@pytest.fixture
def data_module(mock_cfg, mock_trainer):
    return HuggingFaceDataModule(cfg=mock_cfg, trainer=mock_trainer)


def test_val_dataloader_no_val_dir(data_module):
    data_module.cfg.model.data.val_dir = None
    dataloader = data_module.val_dataloader()

    # Assertions
    assert dataloader is None


def test_get_batch(data_module):
    mock_data = {"input_ids": "mock_input_ids", "attention_mask": "mock_attention_mask", "labels": "mock_labels"}
    input_ids, attention_mask, labels = data_module.get_batch(mock_data)

    # Assertions
    assert input_ids == "mock_input_ids"
    assert attention_mask == "mock_attention_mask"
    assert labels == "mock_labels"


def test_get_val_batch(data_module):
    mock_data = {"input_ids": "mock_input_ids", "attention_mask": "mock_attention_mask", "labels": "mock_labels"}
    input_ids, attention_mask, labels = data_module.get_val_batch(mock_data)

    # Assertions
    assert input_ids == "mock_input_ids"
    assert attention_mask == "mock_attention_mask"
    assert labels == "mock_labels"
