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

import json
import os
import shutil
import threading
import unittest
from unittest.mock import MagicMock, mock_open, patch

from hyperpod_nemo_adapter.utils.tracer_utils import VizTracerProfiler


class TestVizTracerProfiler(unittest.TestCase):

    def setUp(self):
        self.ranks = [0, 1]
        self.init_kwargs = {
            "verbose": 0,
            "output_file": "test_output.json",
            "pid_suffix": True,
            "minimize_memory": True,
        }
        with patch("os.environ.get", return_value="0"):
            self.profiler = VizTracerProfiler(self.ranks, **self.init_kwargs)

    def tearDown(self):
        if hasattr(self.profiler, "multiprocess_output_dir"):
            shutil.rmtree(self.profiler.multiprocess_output_dir, ignore_errors=True)

    def test_init(self):
        self.assertEqual(self.profiler.ranks, self.ranks)
        self.assertEqual(self.profiler.verbose, 0)
        self.assertEqual(self.profiler.output_file, "test_output.json")
        self.assertTrue(self.profiler.pid_suffix)
        self.assertFalse(self.profiler._exiting)
        self.assertEqual(self.profiler.rank, 0)
        self.assertFalse(hasattr(self.profiler, "multiprocess_output_dir"))

    @patch("os.environ.get")
    def test_init_with_rank(self, mock_get):
        mock_get.return_value = "2"
        profiler = VizTracerProfiler(self.ranks, **self.init_kwargs)
        self.assertEqual(profiler.rank, 2)

    @patch("tempfile.mkdtemp")
    @patch("hyperpod_nemo_adapter.utils.tracer_utils.VizTracer")
    @patch("hyperpod_nemo_adapter.utils.tracer_utils.install_all_hooks")
    @patch("signal.signal")
    @patch("multiprocessing.util.Finalize")
    def test_start(self, mock_finalize, mock_signal, mock_install_hooks, mock_viztracer, mock_mkdtemp):
        mock_mkdtemp.return_value = "/tmp/test_dir"
        self.profiler.rank = 0

        mock_tracer = MagicMock()
        mock_viztracer.return_value = mock_tracer

        self.profiler.start()

        mock_mkdtemp.assert_called_once()
        mock_viztracer.assert_called_once()
        mock_install_hooks.assert_called_once_with(mock_tracer, [], patch_multiprocess=True)
        mock_signal.assert_called_once()
        mock_finalize.assert_called_once_with(self.profiler, self.profiler.exit_routine, exitpriority=-1)
        mock_tracer.start.assert_called_once()
        self.assertEqual(self.profiler.multiprocess_output_dir, "/tmp/test_dir")

    @patch("os.path.splitext")
    @patch("os.listdir")
    @patch("shutil.rmtree")
    @patch("os.path.exists")
    @patch("hyperpod_nemo_adapter.utils.tracer_utils.ReportBuilder")
    def test_save(self, mock_builder, mock_exists, mock_rmtree, mock_listdir, mock_splitext):
        mock_splitext.return_value = ("prefix", ".json")
        mock_listdir.return_value = ["result1.json", "result2.json"]
        mock_exists.return_value = True
        self.profiler.multiprocess_output_dir = "/tmp/test_dir"

        mock_json_data = {"traceEvents": [], "viztracer_metadata": {"version": "0.0.1"}}

        m = mock_open(read_data=json.dumps(mock_json_data))
        with patch("builtins.open", m):
            self.profiler.save()

        # mock_builder.assert_called_once()
        mock_builder.return_value.save.assert_called_once_with(output_file=f"prefix_{os.getpid()}.json")
        mock_rmtree.assert_called_once_with("/tmp/test_dir", ignore_errors=True)

    @patch("threading._threading_atexits", [MagicMock(), MagicMock()])
    def test_exit_routine(self):
        self.profiler.tracer = MagicMock()
        self.profiler.multiprocess_output_dir = "/tmp/test_dir"
        with patch.object(self.profiler, "save") as mock_save:
            self.profiler.exit_routine()

            self.profiler.tracer.stop.assert_called_once()
            self.profiler.tracer.exit_routine.assert_called_once()
            self.assertEqual(len(threading._threading_atexits), 0)
            self.assertTrue(self.profiler._exiting)
            mock_save.assert_called_once()

    def test_exit_routine_no_tracer(self):
        self.profiler.tracer = None
        self.profiler.exit_routine()
        self.assertFalse(self.profiler._exiting)

    def test_exit_routine_already_exiting(self):
        self.profiler._exiting = True
        self.profiler.exit_routine()
        self.assertTrue(self.profiler._exiting)
