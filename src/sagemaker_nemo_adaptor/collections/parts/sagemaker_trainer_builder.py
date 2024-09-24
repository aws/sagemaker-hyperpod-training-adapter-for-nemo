import logging
import sys
from typing import Union

from omegaconf import DictConfig
from pytorch_lightning import Trainer

from sagemaker_nemo_adaptor.collections.data import (
    DummyDataModule,
    HuggingFaceDataModule,
)
from sagemaker_nemo_adaptor.collections.parts import (
    SageMakerDDPStrategy,
    SageMakerFSDPStrategy,
)

try:
    from sagemaker_nemo_adaptor.utils.callbacks.checkpoint import (
        SageMakerCheckpoint,
        SageMakerCheckpointIO,
        SageMakerCheckpointPeft,
        SageMakerModelCheckpointResilience,
    )

    SUPPORT_CHECKPOINT = True
except:
    SUPPORT_CHECKPOINT = False


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

    @property
    def use_generic_checkpoint(self):
        # Sharded checkpoint.
        sharded_save_any = self.cfg.exp_manager.checkpoint_callback_params.get("save_top_k", 0) != 0
        sharded_save_last = self.cfg.exp_manager.checkpoint_callback_params.get("save_last", True)
        export_sharded = sharded_save_any or sharded_save_last

        # Full checkpoint
        full_save_any = self.cfg.exp_manager.export_full_model.get("every_n_train_steps", 0) != 0
        full_save_last = self.cfg.exp_manager.export_full_model.get("save_last", True)
        export_full = full_save_any or full_save_last
        return export_sharded or export_full

    @property
    def use_resilience_checkpoint(self):
        auto_checkpoint = self.cfg.exp_manager.auto_checkpoint
        return auto_checkpoint.get("enabled", False)

    def _create_plugins(self) -> list:
        plugins = []

        if SUPPORT_CHECKPOINT and (self.use_resilience_checkpoint or self.use_generic_checkpoint):
            plugins.append(SageMakerCheckpointIO())

        return plugins

    def _create_callbacks(self, callbacks=None) -> list:
        assert callbacks is None or isinstance(callbacks, list)
        callbacks = callbacks if callbacks else []

        if SUPPORT_CHECKPOINT:
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
                interval_guard = exp_manager.auto_checkpoint.interval_guard
                callbacks.append(
                    SageMakerModelCheckpointResilience(
                        enable_auto_reload=enabled_auto_reload,
                        checkpoint_dir=exp_manager.get("checkpoint_dir", None),
                        warmup_steps=warmup_steps,
                        drop_n_warmup_steps=drop_n_warmup_steps,
                        interval_guard=interval_guard,
                    )
                )
            # Generic checkpointing callback.
            if self.use_generic_checkpoint:
                callbacks.append(SageMakerCheckpoint(self.cfg))
        return callbacks

    def _create_data_module(self, trainer):
        if self.cfg.model.data.use_synthetic_data:
            return DummyDataModule(self.cfg, trainer)
        if self.cfg.model.data.dataset_type == "hf":
            return HuggingFaceDataModule(self.cfg, trainer)

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
            # Disable deafult lightning ModelCheckpoint if none of them are used.
            enable_checkpointing=self.use_generic_checkpoint or self.use_resilience_checkpoint,
        )

        data_module = self._create_data_module(trainer)
        return trainer, data_module
