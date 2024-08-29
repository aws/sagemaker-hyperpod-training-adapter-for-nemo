# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import sys
from typing import Union

from lightning_fabric.plugins import CheckpointIO
from omegaconf import DictConfig
from pytorch_lightning import Trainer

from sagemaker_nemo_adaptor.collections.data import (
    DummyDataModule,
    HuggingFaceDataModule,
    MegatronDataModule,
)
from sagemaker_nemo_adaptor.collections.parts import (
    SageMakerDDPStrategy,
    SageMakerFSDPStrategy,
)
from sagemaker_nemo_adaptor.utils.callbacks.checkpoint import (
    SageMakerCheckpoint,
    SageMakerCheckpointIO,
)


def _disable_flash_attn_info_log():
    """Disable flash attn logs from transformer_engin.

    Note that this is a workaround solution bc the issue was from Megatron 0.7
    and tranformer_engine v1.8 by setting logging.basicConfig. The function can
    be removed when Nvidia fix the issue.
    """
    logger = logging.getLogger("FusedAttention")
    logger.setLevel(logging.WARNING)
    logger = logging.getLogger("DotProductAttention")
    logger.setLevel(logging.WARNING)


class SageMakerTrainerBuilder:
    """
    Builder type to hide complex configuration of PTL Trainers for SMP/HF models.
    Can be extended to change behavior for a specific model.
    """

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg
        _disable_flash_attn_info_log()

    def _training_strategy(self) -> Union[SageMakerDDPStrategy, SageMakerFSDPStrategy]:
        """
        Returns a DDP or a FSDP strategy passed to Trainer.strategy.
        """
        # check interactive environment TODO: Currently not supporting interactive mode
        _IS_INTERACTIVE = hasattr(sys, "ps1") or bool(sys.flags.interactive)

        if _IS_INTERACTIVE and self.cfg.trainer.devices == 1:
            raise NotImplementedError(f"Currently we don't support interactive mode in SM adaptor")

        if self.cfg.use_smp or self.cfg.model.get("fsdp", True):
            # We're using FSDPStrategy for all SMP usecase for now
            return SageMakerFSDPStrategy(self.cfg)
        else:
            return SageMakerDDPStrategy(self.cfg)

    def _create_checkpoin_io(self) -> CheckpointIO:
        return SageMakerCheckpointIO()

    def _create_plugins(self) -> list:
        plugins = [self._create_checkpoin_io()]
        return plugins

    def _create_callbacks(self, callbacks=None) -> list:
        assert callbacks is None or isinstance(callbacks, list)
        callbacks = callbacks if callbacks else []
        callbacks.append(SageMakerCheckpoint(self.cfg.model))
        return callbacks

    def _create_data_module(self, trainer):
        if self.cfg.model.data.use_synthetic_data:
            return DummyDataModule(self.cfg, trainer)
        if self.cfg.model.data.dataset_type == "hf":
            return HuggingFaceDataModule(self.cfg, trainer)
        return MegatronDataModule(self.cfg, trainer)

    def create_trainer(self, callbacks=None) -> Trainer:
        strategy = self._training_strategy()
        plugins = self._create_plugins()
        callbacks = self._create_callbacks(callbacks)

        # TODO: could be configurable with cfg.trainer
        trainer = Trainer(
            strategy=strategy,
            max_steps=self.cfg.trainer.max_steps,
            logger=False,  # Logger will be configured in exp_manager, set to false here to prevent conflict
            plugins=plugins,
            callbacks=callbacks,
        )

        data_module = self._create_data_module(trainer)
        return trainer, data_module
