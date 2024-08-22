import json
import os
from pathlib import Path
from typing import List, Tuple

import torch
from datasets import load_dataset, load_from_disk

from sagemaker_nemo_adaptor.constants import DataTypes
from sagemaker_nemo_adaptor.utils.log_utils import Logger

_logger = Logger().get_logger()

class HuggingFacePretrainingDataset():
    def __init__(self, input_path: str, partition: str = "train"):
        self.input_path = input_path
        self.partition = partition
        self.data_format = self._get_data_format(self.input_path)
        self._dataset = None
        match self.data_format:
            case DataTypes.ARROW:
                self._dataset = load_from_disk(self.input_path)
            case DataTypes.JSONGZ:
                self._dataset = load_dataset(self.input_path, data_files=[os.path.join(self.input_path, f'*{DataTypes.JSONGZ}')], split=self.partition)
            case DataTypes.JSON:
                self._dataset = load_dataset(self.input_path, data_files=[os.path.join(self.input_path, f'*{DataTypes.JSON}')], split=self.partition)

    def _get_data_format(self, path):
        files = list(Path(path).iterdir())
        files = [f for f in Path(path).iterdir() if f.is_file()]
        suffixes_list = list(set([Path(f).suffixes[0] for f in files]))
        if any(suffix == DataTypes.ARROW for suffix in suffixes_list):
            return DataTypes.ARROW

        elif any(suffix == DataTypes.JSONGZ for suffix in suffixes_list):
            return DataTypes.JSONGZ

        elif any(suffix == DataTypes.JSON for suffix in suffixes_list):
            return DataTypes.JSON

        else:
            raise NotImplementedError(f"Unsupported file format in dataset directory. Expecting files of type '.arrow' '.json.gz' or '.json' but instead found {','.join(suffixes_list)}.")

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
