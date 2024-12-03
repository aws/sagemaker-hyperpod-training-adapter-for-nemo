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

import logging as logger
from typing import cast

try:
    from torch.sagemaker.logger import get_logger

    use_smp_model = True
except:
    use_smp_model = False


class Logger:
    """
    Simple wrapper class for logging with and without SMP enabled environments.
    NOTE: Python Logging facility does not have .off() or .fatal() log levels like torch.sagemaker.logger
    """

    def __init__(self):
        self.logger = None

        if use_smp_model:
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
