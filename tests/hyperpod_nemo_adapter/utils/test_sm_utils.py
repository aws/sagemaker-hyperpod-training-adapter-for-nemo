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
