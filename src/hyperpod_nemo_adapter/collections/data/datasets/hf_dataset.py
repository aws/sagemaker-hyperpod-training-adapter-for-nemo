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
from pathlib import Path
from typing import List, Tuple, Union

import torch
from datasets import interleave_datasets, load_dataset, load_from_disk

from hyperpod_nemo_adapter.constants import DEFAULT_SEED, DataTypes
from hyperpod_nemo_adapter.utils.log_utils import Logger

_logger = Logger().get_logger()


class HuggingFacePretrainingDataset:
    def __init__(self, input_path: Union[str, List[str]], partition: str = "train"):
        self.input_path = input_path
        self.partition = partition
        self.data_format = self._get_data_format(self.input_path)
        self._dataset = None
        datasets = []
        if isinstance(self.input_path, str):
            datasets.append(self.fetch_dataset(self.input_path))
        else:
            for p in self.input_path:
                datasets.append(self.fetch_dataset(p))
        # Combine all datasets into single one.
        self._dataset = interleave_datasets(datasets, seed=DEFAULT_SEED)

    def fetch_dataset(self, path):
        match self.data_format:
            case DataTypes.ARROW:
                dataset = load_from_disk(path)
            case DataTypes.JSONGZ:
                dataset = load_dataset(
                    self.input_path,
                    data_files=[os.path.join(path, f"*{DataTypes.JSONGZ}")],
                    split=self.partition,
                )
            case DataTypes.JSON:
                dataset = load_dataset(
                    self.input_path,
                    data_files=[os.path.join(path, f"*{DataTypes.JSON}")],
                    split=self.partition,
                )
            case _:
                raise NotImplementedError(f"{self.data_format} is not supported.")
        return dataset

    def _get_data_format(self, path):
        if isinstance(path, str):
            path = [path]
        files = []
        for p in path:
            files += [f for f in Path(p).iterdir() if f.is_file()]
        suffixes_list = list(set(["".join(Path(f).suffixes) for f in files]))
        if any(suffix == DataTypes.ARROW for suffix in suffixes_list):
            return DataTypes.ARROW

        elif any(suffix == DataTypes.JSONGZ for suffix in suffixes_list):
            return DataTypes.JSONGZ

        elif any(suffix == DataTypes.JSON for suffix in suffixes_list):
            return DataTypes.JSON

        else:
            raise NotImplementedError(
                f"Unsupported file format in dataset directory. Expecting files of type '.arrow' '.json.gz' or '.json' but instead found {','.join(suffixes_list)}."
            )

    @property
    def dataset(self):
        return self._dataset

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        obj = self._dataset[index]
        iids = torch.tensor(obj["input_ids"], dtype=torch.long)
        attns = torch.tensor(obj["attention_mask"], dtype=torch.long)
        labels = torch.tensor(obj["labels"], dtype=torch.long)
        return iids, attns, labels

    def __len__(self) -> int:
        return len(self._dataset)


class HuggingFacePretrainingVisionDataset(HuggingFacePretrainingDataset):
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        obj = self._dataset[index]
        return obj
