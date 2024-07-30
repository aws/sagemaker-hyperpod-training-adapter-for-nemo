import logging

import pytest


@pytest.fixture
def sagemaker_logger():
    # name comes from
    # https://code.amazon.com/packages/RubikPytorch/blobs/967e6920d3cdb5deaeb1b7bb3327f5a81892dcf3/--/torch/sagemaker/logger.py#L106
    logger = logging.getLogger("smp")
    return logger
