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

from hyperpod_nemo_adapter.collections.data.base import BaseDataModule
from hyperpod_nemo_adapter.collections.data.datasets import DummyDataset
from hyperpod_nemo_adapter.utils.config_utils import get_hf_config_from_name_or_path

_DEFAULT_VOCAB_SIZE = 1024


class DummyDataModule(BaseDataModule):
    """
    Lightning DataModule for synthetic data pipelining
    """

    def train_dataloader(self):
        vocab_size = self.get_vocab_size()
        self._train_ds = DummyDataset(vocab_size=vocab_size, seqlen=self.cfg.model.max_context_width)
        return self._build_dataloader(self._train_ds, batch_size=self.cfg.model.train_batch_size)

    def val_dataloader(self):
        """We're not doing validation for synthetic data"""
        return None

    def get_vocab_size(self):
        """
        Respect predefined model vocab size from recipe. Otherwise use
        default vocab size unless hf config vocab size if provided.
        """
        vocab_size = _DEFAULT_VOCAB_SIZE
        if self.cfg.model.get("vocab_size", None) and self.trainer.model.predefined_model:
            return self.cfg.model.vocab_size

        hf_model_name_or_path = self.cfg.model.get("hf_model_name_or_path", None)
        if hf_model_name_or_path:
            hf_config = get_hf_config_from_name_or_path(self.cfg.model)
            if hf_config:
                vocab_size = hf_config.vocab_size

        return vocab_size

    def get_batch(self, data):
        return data[0], data[1], data[0]

    def get_val_batch(self, data):
        return self.get_batch(data)
