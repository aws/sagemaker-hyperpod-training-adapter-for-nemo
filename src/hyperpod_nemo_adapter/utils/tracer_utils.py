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

import copy
import multiprocessing.util
import os
import shutil
import signal
import sys
import tempfile
import threading

from nemo.utils import logging
from viztracer import VizTracer
from viztracer.patch import install_all_hooks
from viztracer.report_builder import ReportBuilder


class VizTracerProfiler:
    def __init__(self, ranks, **kw):
        self.ranks = ranks
        self.init_kwargs = copy.deepcopy(kw)
        self.init_kwargs["pid_suffix"] = True
        self.init_kwargs["file_info"] = False
        self.init_kwargs["register_global"] = True
        self.init_kwargs["dump_raw"] = True
        self.init_kwargs["process_name"] = None
        self.init_kwargs["verbose"] = 0

        self._exiting = False
        self.verbose = kw["verbose"]
        self.output_file = kw["output_file"]
        self.pid_suffix = kw["pid_suffix"]
        self.tracer = None

        # We cannot use dist.get_rank() because `init_process_group` may not
        # be called when the tracer is created.
        self.rank = int(os.environ.get("RANK", -1))

    def start(self, *a, **kw):
        if self.rank < 0:
            logging.warning("Global rank not found. Disable Viztracer")
            return
        if self.rank not in self.ranks:
            return
        self.multiprocess_output_dir = tempfile.mkdtemp()
        init_kwargs = copy.deepcopy(self.init_kwargs)
        init_kwargs["output_file"] = os.path.join(self.multiprocess_output_dir, "result.json")

        # trace all fork processes
        self.tracer = VizTracer(**init_kwargs)
        install_all_hooks(self.tracer, [], patch_multiprocess=True)

        def term_handler(signalnum, frame):
            sys.exit(0)

        signal.signal(signal.SIGTERM, term_handler)
        multiprocessing.util.Finalize(self, self.exit_routine, exitpriority=-1)
        self.tracer.start()

    def save(self):
        try:
            if self.pid_suffix:
                prefix, suffix = os.path.splitext(self.output_file)
                prefix = f"{prefix}_{os.getpid()}"
                self.output_file = prefix + suffix

            out = self.multiprocess_output_dir
            builder = ReportBuilder(
                [os.path.join(out, f) for f in os.listdir(out) if f.endswith(".json")],
                minimize_memory=self.init_kwargs["minimize_memory"],
                verbose=self.verbose,
            )
            builder.save(output_file=self.output_file)
        finally:
            shutil.rmtree(self.multiprocess_output_dir, ignore_errors=True)

    def exit_routine(self):
        if not self.tracer:
            return

        if self._exiting:
            return

        self._exiting = True
        self.tracer.stop()
        # concurrent.future requires a proper release by executing
        # threading._threading_atexits or it will deadlock if not explicitly
        # release the resource in the code
        # Python 3.9+ has this issue
        if threading._threading_atexits:
            for atexit_call in threading._threading_atexits:
                atexit_call()
            threading._threading_atexits = []

        self.tracer.exit_routine()
        self.save()
