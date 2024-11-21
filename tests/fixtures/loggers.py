import logging

import pytest


@pytest.fixture
def sagemaker_logger():
    logger = logging.getLogger("smp")
    return logger
