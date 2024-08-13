import logging as logger
from typing import cast

try:
    from torch.sagemaker.logger import get_logger

    use_smp = True
except:
    use_smp = False

class Logger:
    """
    Simple wrapper class for logging with and without SMP enabled environments.
    NOTE: Python Logging facility does not have .off() or .fatal() log levels like torch.sagemaker.logger
    """

    def __init__(self):
        self.logger = None

        if use_smp:
            self.logger = get_logger()
        else:
            _logger = logger.getLogger("smp")
            _logger.setLevel(logger.DEBUG)

            sh = logger.StreamHandler()
            sh.setLevel(logger.DEBUG)

            formatter = logger.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            sh.setFormatter(formatter)

            _logger.addHandler(sh)
            self.logger = _logger

    def get_logger(self) -> logger.Logger:
        return cast(logger.Logger, self.logger)
