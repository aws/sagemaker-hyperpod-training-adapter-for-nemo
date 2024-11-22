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

try:
    import gpu_affinity as ga

    SUPPORT_GPU_AFFINITY = True
except ImportError:
    SUPPORT_GPU_AFFINITY = False

from hyperpod_nemo_adapter.utils.log_utils import Logger

logger = Logger().get_logger()


def initialize_gpu_affinity(gpu_id, nproc_per_node):
    if not SUPPORT_GPU_AFFINITY:
        return

    try:
        affinity = ga.set_affinity(gpu_id, nproc_per_node)
        logger.debug(f"[GPU:{gpu_id}] affinity: {affinity}")
    except Exception as e:
        logger.warning(f"set_affinity fail. error: {e}")
