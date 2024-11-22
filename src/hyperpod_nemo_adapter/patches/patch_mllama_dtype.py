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

import torch
from transformers.models.mllama import MllamaVisionModel

is_patched = False

orginal_dtype = MllamaVisionModel.dtype


def unapply_patch():
    global is_patched
    MllamaVisionModel.dtype = orginal_dtype
    is_patched = False


def apply_patch(dtype=torch.bfloat16):
    global is_patched
    MllamaVisionModel.dtype = dtype
    is_patched = True
