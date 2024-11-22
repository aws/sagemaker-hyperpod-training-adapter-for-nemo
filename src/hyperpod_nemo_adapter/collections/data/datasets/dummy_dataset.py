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


class DummyDataset(torch.utils.data.dataset.Dataset):
    """Dummy Dataset."""

    def __init__(self, vocab_size=1024, seqlen=2048, length=100000, data_type="gpt"):
        self.vocab_size = vocab_size
        self.seqlen = seqlen
        if data_type == "gpt":
            self.mask = torch.ones((seqlen,))
        elif data_type == "bert":
            raise NotImplementedError
        self.length = length
        self.input_paths = None

    def __getitem__(self, index):
        return torch.randint(self.vocab_size, (self.seqlen,), dtype=torch.long), self.mask

    def __len__(self):
        return self.length
