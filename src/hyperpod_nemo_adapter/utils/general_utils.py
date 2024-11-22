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

from packaging import version as pversion

from hyperpod_nemo_adapter.constants import MULTI_MODAL_HF_VERSION


def is_power_of_two(n: int) -> bool:
    "Brian Kernighan's Algorithm"
    return n > 0 and n & (n - 1) == 0


def is_slurm_run():
    """Check if the script is running under SLURM."""
    return "SLURM_JOB_ID" in os.environ


def can_use_multimodal() -> bool:
    import transformers

    current_version = pversion.parse(transformers.__version__)
    required_version = pversion.parse(MULTI_MODAL_HF_VERSION)

    return current_version >= required_version
