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

from hyperpod_nemo_adapter.utils.sm_utils import _strip_mp_params_helper


class TestSMUtils:
    def test_command_line_args_no_mp(self):
        try:
            args = ["llama_pretrain.py", "--config ."]
            new_args = _strip_mp_params_helper(args)
            assert args == new_args
        except Exception as e:
            pytest.fail(f"_strip_mp_params_helper did not produce the correct args")

    def test_command_line_args_with_mp_single_middle(self):
        try:
            args = [
                "llama_pretrain.py",
                "--config",
                ".",
                "--mp_parameters",
                "placement=cluster",
                "--other_params",
                "abc",
            ]
            expected_args = ["llama_pretrain.py", "--config", ".", "--other_params", "abc"]
            new_args = _strip_mp_params_helper(args)
            assert new_args == expected_args
        except Exception as e:
            pytest.fail(f"_strip_mp_params_helper did not produce the correct args")

    def test_command_line_args_with_mp_multi_middle(self):
        try:
            args = [
                "llama_pretrain.py",
                "--config",
                ".",
                "--mp_parameters",
                "placement=cluster,auto-partition=True",
                "--other_params",
                "abc",
            ]
            expected_args = ["llama_pretrain.py", "--config", ".", "--other_params", "abc"]
            new_args = _strip_mp_params_helper(args)
            assert new_args == expected_args
        except Exception as e:
            pytest.fail(f"_strip_mp_params_helper did not produce the correct args")

    def test_command_line_args_with_mp_multi_with_space_middle(self):
        try:
            args = [
                "llama_pretrain.py",
                "--config",
                ".",
                "--mp_parameters",
                "placement=cluster",
                "auto-partition=True",
                "--other_params",
                "abc",
            ]
            expected_args = ["llama_pretrain.py", "--config", ".", "--other_params", "abc"]
            new_args = _strip_mp_params_helper(args)
            assert new_args == expected_args
        except Exception as e:
            pytest.fail(f"_strip_mp_params_helper did not produce the correct args")

    def test_command_line_args_with_mp_single_end(self):
        try:
            args = ["llama_pretrain.py", "--config", ".", "--mp_parameters", "placement=cluster"]
            expected_args = ["llama_pretrain.py", "--config", "."]
            new_args = _strip_mp_params_helper(args)
            assert new_args == expected_args
        except Exception as e:
            pytest.fail(f"_strip_mp_params_helper did not produce the correct args")

    def test_command_line_args_with_mp_multi_end(self):
        try:
            args = ["llama_pretrain.py", "--config", ".", "--mp_parameters", "placement=cluster,auto-partition=True"]
            expected_args = ["llama_pretrain.py", "--config", "."]
            new_args = _strip_mp_params_helper(args)
            assert new_args == expected_args
        except Exception as e:
            pytest.fail(f"_strip_mp_params_helper did not produce the correct args")

    def test_command_line_args_with_mp_multi_with_space_end(self):
        try:
            args = ["llama_pretrain.py", "--config", ".", "--mp_parameters", "placement=cluster", "auto-partition=True"]
            expected_args = ["llama_pretrain.py", "--config", "."]
            new_args = _strip_mp_params_helper(args)
            assert new_args == expected_args
        except Exception as e:
            pytest.fail(f"_strip_mp_params_helper did not produce the correct args")
