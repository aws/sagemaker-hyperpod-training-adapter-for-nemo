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

from hyperpod_nemo_adapter.collections.data.datasets.hf_dataset import (
    HuggingFacePretrainingDataset,
)


class HuggingFaceDPODataset(HuggingFacePretrainingDataset):
    """
    Dataset for binary preference in DPO jobs. Inherits from HuggingFacePretrainingDataset.
    """

    def __getitem__(self, index: int) -> dict:
        obj = self._dataset[index]
        return {
            "prompt_input_ids": torch.tensor(obj["prompt_input_ids"], dtype=torch.long),
            "chosen_input_ids": torch.tensor(obj["chosen_input_ids"], dtype=torch.long),
            "rejected_input_ids": torch.tensor(obj["rejected_input_ids"], dtype=torch.long),
        }
