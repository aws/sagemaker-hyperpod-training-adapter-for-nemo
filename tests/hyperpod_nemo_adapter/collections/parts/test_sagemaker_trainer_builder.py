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
