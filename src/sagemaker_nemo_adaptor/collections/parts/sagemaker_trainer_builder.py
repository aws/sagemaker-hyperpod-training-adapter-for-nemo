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

import sys
from typing import Union

from omegaconf import DictConfig
from pytorch_lightning import Trainer
from torch.sagemaker.logger import get_logger

from sagemaker_nemo_adaptor.collections.data import DummyDataModule, GPTDataModule
from sagemaker_nemo_adaptor.collections.parts import (
    SageMakerDDPStrategy,
    SageMakerFSDPStrategy,
)

logger = get_logger()


class SageMakerTrainerBuilder:
    """
    Builder type to hide complex configuration of PTL Trainers for SMP/HF models.
    Can be extended to change behavior for a specific model.
    """

    def __init__(self, cfg: DictConfig) -> None:
        self.cfg = cfg

    def _training_strategy(self) -> Union[SageMakerDDPStrategy, SageMakerFSDPStrategy]:
        """
        Returns a DDP or a FSDP strategy passed to Trainer.strategy.
        """

        # check interactive environment TODO: Currently not supporting interactive mode
        _IS_INTERACTIVE = hasattr(sys, "ps1") or bool(sys.flags.interactive)

        if _IS_INTERACTIVE and self.cfg.trainer.devices == 1:
            raise NotImplementedError(f"Currently we don't support interactive mode in SM adaptor")

        if self.cfg.use_smp:  # TODO: we need recipe checker to check when use_smp is not provided
            smp_config_dict = {
                "activation_loading_horizon": self.cfg.model.activation_loading_horizon,
                "sm_activation_offloading": self.cfg.model.offload_activations > 0,
            }
            if self.cfg.model.shard_degree is not None:
                smp_config_dict["hybrid_shard_degree"] = self.cfg.model.shard_degree
            smp_config_dict["tensor_parallel_degree"] = self.cfg.model.tensor_model_parallel_degree
            smp_config_dict["expert_parallel_degree"] = self.cfg.model.expert_model_parallel_degree
            smp_config_dict["random_seed"] = self.cfg.model.seed

        if self.cfg.use_smp or self.cfg.model.get("fsdp", False):
            # We're using FSDPStrategy for all SMP usecase for now
            return SageMakerFSDPStrategy(use_smp=self.cfg.use_smp, cfg=self.cfg, smp_config_dict=smp_config_dict)
        else:
            return NLPDDPStrategy(cfg=self.cfg)

    def create_trainer(self, callbacks=None) -> Trainer:
        strategy = self._training_strategy()
        # TODO: Add callbacks for checkpoints

        trainer = Trainer(
            strategy=strategy,
            max_steps=self.cfg.trainer.max_steps,
            logger=False,  # Logger will be configured in exp_manager, set to false here to prevent conflict
        )  # TODO: could be configurable with cfg.trainer

        data_module = (
            DummyDataModule(self.cfg, trainer)
            if self.cfg.model.data.use_synthetic_data
            else GPTDataModule(self.cfg, trainer)
        )

        return trainer, data_module
