import logging
import os

import pytorch_lightning as pl
import torch.distributed as dist
from nemo.utils import logging
from pytorch_lightning import Callback

from sagemaker_nemo_adaptor.constants import SageMakerCheckpointType
from sagemaker_nemo_adaptor.utils.callbacks.ckpt_io import SageMakerCheckpointIO


class SageMakerCheckpoint(Callback):
    def __init__(
        self,
        cfg,
        *a,
        **kw,
    ):
        super().__init__(*a, **kw)
        self._checkpoint_dir = cfg.exp_manager.get("checkpoint_dir", None)
        self._save_full_every_n_steps = cfg.exp_manager.export_full_model.get("every_n_train_steps", None)
        self._resume_from_checkpoint = cfg.get("resume_from_checkpoint", None)

    def on_train_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if self._resume_from_checkpoint:
            logging.info(f"load_checkpoint: {self._resume_from_checkpoint}")
            self._load_checkpoint(trainer)

    def on_train_batch_end(
        self,
        trainer: "pl.Trainer",
        *args,
        **kwargs,
    ) -> None:
        self._save_checkpoints(trainer)

    def _get_checkpoint_dir(self, trainer):
        if not self._checkpoint_dir:
            path = os.path.join(trainer.default_root_dir, "checkpoints")
            self._checkpoint_dir = path
        return self._checkpoint_dir

    def _should_save_full(self, trainer: "pl.Trainer"):
        """
        Check if the full checkpoint should be saved if:
        1. hit every n steps defined by save_full_every_n_steps.
        2. reach max steps. ie: traininig finishes.
        """
        if not self._save_full_every_n_steps:
            return False
        is_every_n = trainer.global_step % self._save_full_every_n_steps == 0
        is_last_step = trainer.max_steps == trainer.global_step
        return is_every_n or is_last_step

    def _save(
        self,
        trainer: "pl.Trainer",
        checkpoint_io: SageMakerCheckpointIO,
        checkpoint_type: SageMakerCheckpointType,
        checkpoint_dir: str,
    ):
        """Save one checkpoint using corresponding checkpoint type."""
        checkpoint_io.checkpoint_type = checkpoint_type
        trainer.save_checkpoint(checkpoint_dir)
        if dist.get_rank() == 0:
            logging.info(f"Saving {checkpoint_type} Checkpoint to {checkpoint_dir}")

    def _save_checkpoints(self, trainer: "pl.Trainer"):
        """
        At each step, we check if we should save one checkpoint type.
        """
        checkpoint_io = trainer.strategy.checkpoint_io
        assert isinstance(checkpoint_io, SageMakerCheckpointIO)
        checkpoint_io.wait()

        checkpoint_info = []
        if self._should_save_full(trainer):
            checkpoint_info.append(
                (
                    SageMakerCheckpointType.FULL,
                    os.path.join(self._get_checkpoint_dir(trainer), f"full/steps_{trainer.global_step}/"),
                )
            )

        for checkpoint_type, checkpoint_dir in checkpoint_info:
            self._save(trainer, checkpoint_io, checkpoint_type, checkpoint_dir)

    def _load_checkpoint(self, trainer):
        checkpoint_io = trainer.strategy.checkpoint_io
        assert isinstance(checkpoint_io, SageMakerCheckpointIO)
        path = self._resume_from_checkpoint
        state_dict = trainer.strategy.load_checkpoint(path, trainer)
        logging.debug(state_dict)
