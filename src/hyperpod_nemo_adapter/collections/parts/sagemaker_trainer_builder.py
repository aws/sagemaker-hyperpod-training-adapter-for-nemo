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

import logging
import os
import sys

from nemo.lightning.pytorch.callbacks import NsysCallback
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer

from hyperpod_nemo_adapter.collections.data import (
    DummyDataModule,
    HuggingFaceDataModule,
    HuggingFaceVisionDataModule,
)
from hyperpod_nemo_adapter.collections.parts import SageMakerFSDPStrategy

try:
    from hyperpod_nemo_adapter.utils.callbacks.checkpoint import (
        SageMakerCheckpoint,
        SageMakerCheckpointIO,
        SageMakerCheckpointPeft,
        SageMakerModelCheckpointResilience,
    )

    SUPPORT_CHECKPOINT = True
except ImportError:
    SUPPORT_CHECKPOINT = False

try:
    from hyperpod_nemo_adapter.utils.tracer_utils import VizTracerProfiler

    SUPPORT_VIZTRACER = True
except ImportError:
    SUPPORT_VIZTRACER = False


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


def _disable_aiobotocore_credential_log():
    from aiobotocore.credentials import logger

    logger.setLevel(logging.WARNING)


def _disable_warning_message(message):
    import warnings

    warnings.filterwarnings("ignore", message)


def _get_viztracer_profiler(cfg):
    if not SUPPORT_VIZTRACER:
        return
    viztracer = cfg.model.get("viztracer", None)
    if not viztracer:
        return
    enabled = viztracer.get("enabled", False)
    if not enabled:
        return
    init_kwargs = OmegaConf.to_container(viztracer, resolve=True)
    init_kwargs.pop("enabled", None)
    if not init_kwargs.get("output_file", None):
        path = os.path.join(cfg.exp_manager.exp_dir, "result.json")
        init_kwargs["output_file"] = path
    return VizTracerProfiler(**init_kwargs)


class SageMakerTrainerBuilder:
    """
    Builder type to hide complex configuration of PTL Trainers for SMP/HF models.
    Can be extended to change behavior for a specific model.
    """

    def __init__(self, cfg: DictConfig) -> None:
        self._profile(cfg)
        self.cfg = cfg
        _disable_flash_attn_info_log()
        _disable_aiobotocore_credential_log()
        # We don't save the state_dict of the dataloader. Instead, we save/load
        # the state_dict of the datamodule.
        _disable_warning_message(".*your dataloader is not resumable*")

    def _profile(self, cfg):
        self.tracer = _get_viztracer_profiler(cfg)
        if self.tracer:
            self.tracer.start()

    def _training_strategy(self) -> SageMakerFSDPStrategy:
        """
        Returns a FSDP strategy passed to Trainer.strategy.
        """
        _IS_INTERACTIVE = hasattr(sys, "ps1") or bool(sys.flags.interactive)

        if _IS_INTERACTIVE and self.cfg.trainer.devices == 1:
            raise NotImplementedError(f"Currently we don't support interactive mode in SM adapter")

        if self.cfg.use_smp_model or self.cfg.model.get("fsdp", True):
            # We're using FSDPStrategy for all SMP usecase for now
            return SageMakerFSDPStrategy(self.cfg)
        else:
            raise NotImplementedError(f"Currently we only support FSDPStrategy")

    @property
    def use_generic_checkpoint(self):
        # Sharded checkpoint.
        sharded_save_any = self.cfg.exp_manager.checkpoint_callback_params.get("save_top_k", 0) != 0
        sharded_save_last = self.cfg.exp_manager.checkpoint_callback_params.get("save_last", True)
        export_sharded = sharded_save_any or sharded_save_last
        assert not (
            self.use_resilience_checkpoint and export_sharded
        ), "Turning on auto_checkpoint and checkpoint in checkpoint_callback_params are mutually exclusive"

        # Full checkpoint
        full_save_any = self.cfg.exp_manager.export_full_model.get("every_n_train_steps", 0) != 0
        full_save_last = self.cfg.exp_manager.export_full_model.get("save_last", True)
        export_full = full_save_any or full_save_last
        return export_sharded or export_full

    @property
    def use_resilience_checkpoint(self):
        auto_checkpoint = self.cfg.exp_manager.auto_checkpoint
        return auto_checkpoint.get("enabled", False)

    def _create_checkpoint_callbacks(self):
        callbacks = []
        if not SUPPORT_CHECKPOINT:
            return callbacks

        exp_manager = self.cfg.exp_manager
        # PEFT checkpointing callback.
        if self.cfg.model.peft.peft_type is not None:
            if self.use_generic_checkpoint:
                callbacks.append(SageMakerCheckpointPeft(self.cfg))
            # If using PEFT, do not use regular checkpoint callbacks as they may fail
            return callbacks

        # Resilience checkpointing callback.
        if self.use_resilience_checkpoint:
            # If user specify a path to resume, disable auto resume.
            enabled_auto_reload = exp_manager.resume_from_checkpoint == None
            warmup_steps = exp_manager.auto_checkpoint.warmup_steps
            drop_n_warmup_steps = exp_manager.auto_checkpoint.drop_n_warmup_steps
            callbacks.append(
                SageMakerModelCheckpointResilience(
                    enable_auto_reload=enabled_auto_reload,
                    checkpoint_dir=exp_manager.get("checkpoint_dir", None),
                    warmup_steps=warmup_steps,
                    drop_n_warmup_steps=drop_n_warmup_steps,
                )
            )
        # Generic checkpointing callback.
        if self.use_generic_checkpoint:
            callbacks.append(SageMakerCheckpoint(self.cfg))
        return callbacks

    def _create_nsys_callbacks(self):
        nsys_profile = self.cfg.model.get("nsys_profile", None)
        if not nsys_profile:
            return []

        enabled = nsys_profile.get("enabled", False)
        if not enabled:
            return []

        return [
            NsysCallback(
                start_step=nsys_profile.get("start_step", 10),
                end_step=nsys_profile.get("end_step", 10),
                ranks=nsys_profile.get("ranks", [0]),
                gen_shape=nsys_profile.get("gen_shape", False),
            )
        ]

    def _create_plugins(self) -> list:
        plugins = []

        if SUPPORT_CHECKPOINT and (self.use_resilience_checkpoint or self.use_generic_checkpoint):
            plugins.append(SageMakerCheckpointIO())

        return plugins

    def _create_callbacks(self, callbacks=None) -> list:
        assert callbacks is None or isinstance(callbacks, list)
        callbacks = callbacks if callbacks else []
        callbacks += self._create_nsys_callbacks()
        callbacks += self._create_checkpoint_callbacks()
        return callbacks

    def _create_data_module(self, trainer):
        if self.cfg.model.multi_modal:
            return HuggingFaceVisionDataModule(self.cfg, trainer)
        if self.cfg.model.data.use_synthetic_data:
            return DummyDataModule(self.cfg, trainer)
        if self.cfg.model.data.dataset_type == "hf":
            return HuggingFaceDataModule(self.cfg, trainer)

    def create_trainer(self, callbacks=None) -> Trainer:
        strategy = self._training_strategy()
        plugins = self._create_plugins()
        callbacks = self._create_callbacks(callbacks)

        trainer = Trainer(
            strategy=strategy,
            max_steps=self.cfg.trainer.max_steps,
            logger=False,  # Logger will be configured in exp_manager, set to false here to prevent conflict
            plugins=plugins,
            callbacks=callbacks,
            log_every_n_steps=self.cfg.trainer.log_every_n_steps,
            # Disable deafult lightning ModelCheckpoint if none of them are used.
            enable_checkpointing=self.use_generic_checkpoint or self.use_resilience_checkpoint,
            val_check_interval=self.cfg.trainer.val_check_interval,
            limit_val_batches=self.cfg.trainer.limit_val_batches,
            devices=self.cfg.trainer.get("devices", "auto"),
        )

        data_module = self._create_data_module(trainer)
        return trainer, data_module
