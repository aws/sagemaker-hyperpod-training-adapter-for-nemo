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

from nemo.utils import logging

logging.info("Penrose before SageMakerFSDPStrategy on parts.__init__.py")
from .fsdp_strategy import SageMakerFSDPStrategy
logging.info("Penrose before SageMakerTrainerBuilder on parts.__init__.py")
from .sagemaker_trainer_builder import SageMakerTrainerBuilder

logging.info("Penrose after imports on parts.__init__.py")
