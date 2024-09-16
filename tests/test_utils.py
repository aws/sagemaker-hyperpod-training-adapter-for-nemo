import os
import stat
import tempfile


def create_temp_directory():
    """Create a temporary directory and Set full permissions for the directory"""
    temp_dir = tempfile.mkdtemp()
    os.chmod(temp_dir, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)
    return temp_dir
