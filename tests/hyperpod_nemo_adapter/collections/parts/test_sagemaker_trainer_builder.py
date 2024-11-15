from unittest.mock import patch

from omegaconf import OmegaConf

from hyperpod_nemo_adapter.collections.parts.sagemaker_trainer_builder import (
    _get_viztracer_profiler,
)


class TestGetViztracerProfiler:

    def test_viztracer_not_supported(self):
        with patch("hyperpod_nemo_adapter.collections.parts.sagemaker_trainer_builder.SUPPORT_VIZTRACER", False):
            cfg = OmegaConf.create({})
            assert _get_viztracer_profiler(cfg) is None

    def test_viztracer_not_enabled(self):
        cfg = OmegaConf.create({"model": {"viztracer": {"enabled": False}}})
        assert _get_viztracer_profiler(cfg) is None

    def test_viztracer_not_configured(self):
        cfg = OmegaConf.create({"model": {}})
        assert _get_viztracer_profiler(cfg) is None

    @patch("hyperpod_nemo_adapter.collections.parts.sagemaker_trainer_builder.VizTracerProfiler")
    def test_viztracer_enabled_with_default_output(self, mock_viztracer_profiler):
        cfg = OmegaConf.create({"model": {"viztracer": {"enabled": True}}, "exp_manager": {"exp_dir": "/exp/dir"}})
        _get_viztracer_profiler(cfg)
        mock_viztracer_profiler.assert_called_once_with(output_file="/exp/dir/result.json")
