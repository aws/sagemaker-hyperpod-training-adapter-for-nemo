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

from hyperpod_nemo_adapter.collections.data import HuggingFaceDPODataModule


@pytest.fixture
def mock_cfg():
    return OmegaConf.create(
        {
            "model": {
                "data": {"train_dir": "mock/train/dir", "val_dir": "mock/val/dir"},
                "train_batch_size": 1,
                "val_batch_size": 1,
                "seed": 42,
            }
        }
    )


@pytest.fixture
def mock_trainer():
    return MagicMock(spec=Trainer)


@pytest.fixture
def dpo_data_module(mock_cfg, mock_trainer):
    return HuggingFaceDPODataModule(cfg=mock_cfg, trainer=mock_trainer)


def test_val_dataloader_no_val_dir(dpo_data_module):
    dpo_data_module.cfg.model.data.val_dir = None
    dataloader = dpo_data_module.val_dataloader()
    # Assertions
    assert dataloader is None


def test_get_batch(dpo_data_module):
    mock_data = {
        "prompt_input_ids": "prompt_input_ids",
        "prompt_attention_mask": "prompt_attention_mask",
        "chosen_input_ids": "chosen_input_ids",
        "chosen_attention_mask": "chosen_attention_mask",
        "rejected_input_ids": "rejected_input_ids",
        "rejected_attention_mask": "rejected_attention_mask",
    }
    result = dpo_data_module.get_batch(mock_data)

    # Assertions
    assert result == (
        "prompt_input_ids",
        "prompt_attention_mask",
        "chosen_input_ids",
        "chosen_attention_mask",
        "rejected_input_ids",
        "rejected_attention_mask",
    )


def test_get_val_batch(dpo_data_module):
    mock_data = {
        "prompt_input_ids": "val_prompt_input_ids",
        "prompt_attention_mask": "val_prompt_attention_mask",
        "chosen_input_ids": "val_chosen_input_ids",
        "chosen_attention_mask": "val_chosen_attention_mask",
        "rejected_input_ids": "val_rejected_input_ids",
        "rejected_attention_mask": "val_rejected_attention_mask",
    }

    result = dpo_data_module.get_val_batch(mock_data)

    assert result == (
        "val_prompt_input_ids",
        "val_prompt_attention_mask",
        "val_chosen_input_ids",
        "val_chosen_attention_mask",
        "val_rejected_input_ids",
        "val_rejected_attention_mask",
    )
