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
