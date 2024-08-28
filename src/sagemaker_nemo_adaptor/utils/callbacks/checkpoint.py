import os

import pytorch_lightning as pl
from pytorch_lightning import Callback

from sagemaker_nemo_adaptor.utils.callbacks.ckpt_io import SageMakerCheckpointIO


class SageMakerCheckpoint(Callback):
    def __init__(
        self,
        cfg,
        *a,
        **kw,
    ):
        super().__init__(*a, **kw)
        self._checkpoint_dir = cfg.get("checkpoint_dir", None)

    def on_train_batch_end(
        self,
        trainer: "pl.Trainer",
        *args,
        **kwargs,
    ) -> None:
        self._save_checkpoint(trainer)

    def on_train_epoch_end(
        self,
        trainer: "pl.Trainer",
        *args,
        **kwargs,
    ) -> None:
        self._save_checkpoint(trainer)

    def on_train_end(
        self,
        trainer: "pl.Trainer",
        *args,
        **kwargs,
    ) -> None:
        checkpoint_io = trainer.strategy.checkpoint_io
        assert isinstance(checkpoint_io, SageMakerCheckpointIO)
        checkpoint_io.wait()

    def _get_checkpoint_dir(self, trainer):
        if not self._checkpoint_dir:
            path = os.path.join(trainer.default_root_dir, "checkpoints")
            self._checkpoint_dir = path
        return self._checkpoint_dir

    def _save_checkpoint(self, trainer) -> SageMakerCheckpointIO:
        checkpoint_io = trainer.strategy.checkpoint_io
        assert isinstance(checkpoint_io, SageMakerCheckpointIO)
        checkpoint_io.wait()
        checkpoint_dir = self._get_checkpoint_dir(trainer)
        trainer.save_checkpoint(checkpoint_dir)
