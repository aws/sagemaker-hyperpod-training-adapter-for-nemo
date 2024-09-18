from unittest.mock import MagicMock

import pytest
from omegaconf import OmegaConf
from pytorch_lightning import Trainer

from sagemaker_nemo_adaptor.collections.data import HuggingFaceDataModule


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
