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
import stat
import tempfile
from functools import wraps

import pytest
import torch


def create_temp_directory():
    """Create a temporary directory and Set full permissions for the directory"""
    temp_dir = tempfile.mkdtemp()
    os.chmod(temp_dir, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
    return temp_dir


def skip_if_lt_x_gpu(x):
    """
    Skip the test if less than x GPUs are available.
    """

    def decorator(func):
        @pytest.mark.skipif(
            not torch.cuda.is_available() or torch.cuda.device_count() < x,
            reason=f"This test requires at least {x} GPU(s)",
        )
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        return wrapper

    return decorator
