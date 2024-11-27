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

import os
from unittest.mock import patch

import pytest

from hyperpod_nemo_adapter.utils.exp_manager import exp_manager
from tests.test_utils import create_temp_directory


# Define a fixture for the trainer
@pytest.fixture
def mock_trainer(mocker):
    trainer = mocker.MagicMock()
    trainer.node_rank = 0
    trainer.num_devices = 1
    trainer.num_nodes = 1
    trainer.fast_dev_run = False
    trainer._default_root_dir = None
    trainer.logger = None
    return trainer


# Test case when cfg is None
def test_exp_manager_no_cfg(mock_trainer):
    with patch("nemo.utils.logging.error") as mock_error:
        result = exp_manager(mock_trainer, cfg=None)
        mock_error.assert_called_once_with("exp_manager did not receive a cfg argument. It will be disabled.")
        assert result is None


# Test case when fast_dev_run is True
def test_exp_manager_fast_dev_run(mock_trainer):
    mock_trainer.fast_dev_run = True
    with patch("nemo.utils.logging.info") as mock_info:
        result = exp_manager(mock_trainer, cfg={})
        mock_info.assert_called_once_with(
            "Trainer was called with fast_dev_run. exp_manager will return without any functionality."
        )
        assert result is None


# Test case for invalid cfg
def test_exp_manager_invalid_cfg(mock_trainer):
    mock_cfg = []
    with pytest.raises(
        ValueError, match="cfg was type: {}. Expected either a dict or a DictConfig".format(type(mock_cfg))
    ):
        exp_manager(mock_trainer, cfg=mock_cfg)


# Test case when cfg is a valid configuration
def test_exp_manager_valid_cfg(mock_trainer):
    exp_dir = create_temp_directory()
    test_name = "test_name"
    test_version = "test_version"
    mock_cfg = {
        "exp_dir": exp_dir,
        "name": test_name,
        "version": test_version,
        "use_datetime_version": False,
        "resume_if_exists": False,
        "ema": {"enable": True},
        "create_early_stopping_callback": True,
        "max_time_per_run": "00:03:55:00",
    }
    with patch("hyperpod_nemo_adapter.utils.get_rank.is_global_rank_zero", return_value=True):
        log_dir = exp_manager(mock_trainer, mock_cfg)

    expected_log_dir = str(exp_dir) + "/" + test_name + "/" + test_version
    assert str(log_dir) == str(expected_log_dir)
    assert os.path.exists(log_dir)
    assert os.path.exists(log_dir / "cmd-args.log")
    assert os.path.exists(log_dir / "nemo_error_log.txt")
    assert os.path.exists(log_dir / "lightning_logs.txt")
