import os
from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, Optional, Union

import pytorch_lightning as pl
import torch
from nemo.utils import logging
from pytorch_lightning.callbacks import Checkpoint
from torch import Tensor
from torch.sagemaker import state

from sagemaker_nemo_adaptor.constants import (
    SageMakerCheckpointType,
    SageMakerMonitorMode,
)
from sagemaker_nemo_adaptor.utils.app_state import SageMakerAppState
from sagemaker_nemo_adaptor.utils.callbacks.ckpt_io import SageMakerCheckpointIO


@dataclass
class TopkCheckPoint:
    score: Union[int, float]
    checkpoint_path: str = ""
    monitor: str = "step"
    step_at_save: int = 0
    epoch_at_save: int = 0


class SageMakerModelCheckpointBase(Checkpoint):
    """
    Base class for SageMakerModelCheckpoint callback.
    """

    def __init__(self, checkpoint_dir: Optional[str] = None, *args, **kw):
        super().__init__(*args, **kw)
        self._checkpoint_dir = checkpoint_dir
        self._app_state = SageMakerAppState()

    def on_train_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        checkpoint_io = trainer.strategy.checkpoint_io
        assert isinstance(checkpoint_io, SageMakerCheckpointIO)
        checkpoint_io.teardown(trainer)

    def on_train_batch_end(
        self,
        trainer: "pl.Trainer",
        *args,
        **kwargs,
    ) -> None:
        self._save_checkpoints(trainer)

    @property
    def checkpoint_dir(self):
        if not self._checkpoint_dir:
            path = os.path.join(self._app_state.exp_dir, "checkpoints")
            self._checkpoint_dir = path
        return self._checkpoint_dir

    def _load_checkpoint(self, trainer, path, typ):
        """
        Load checkpoint from a given path with the given checkpoint type.
        """
        checkpoint_io = trainer.strategy.checkpoint_io
        assert isinstance(checkpoint_io, SageMakerCheckpointIO)
        checkpoint_io.checkpoint_type = typ
        state_dict = trainer.strategy.load_checkpoint(path, trainer)
        logging.debug(state_dict)
        trainer.strategy.load_model_state_dict(state_dict)
        trainer.strategy.load_optimizer_state_dict(trainer, state_dict, path)
        # TODO(htzhong): add the data module loading here.

    def _save(
        self,
        trainer: "pl.Trainer",
        checkpoint_io: SageMakerCheckpointIO,
        checkpoint_type: SageMakerCheckpointType,
        checkpoint_dir: str,
    ):
        """Save one checkpoint using corresponding checkpoint type."""
        checkpoint_io.checkpoint_type = checkpoint_type
        weights_only = checkpoint_type == SageMakerCheckpointType.FULL
        trainer.save_checkpoint(checkpoint_dir, weights_only, trainer)


class SageMakerModelCheckpointResilience(SageMakerModelCheckpointBase):
    """
    This callback is used to enable resilience feature which automatically save local checkpoint
    asynchronously with a background process.

    Note: The saved checkpiont is a local checkpoint type which ONLY contains the local model/optimizer
    weights in a given shard. Therefore, it requires the same hybrid shard degree if enable_auto_reload
    is set to True.

    every_n_train_steps attributes will be automatically updated.

    A checkpoint will be saved to: /checkpoint_dir/local/steps_{i}/tp{j}_ep{k}_fsdp{l}/

    Usage:
    trainer = Trainer(
        strategy,
        max_steps,
        plugins=[SageMakerCheckpointIO()],
        callbacks=[SageMakerModelCheckpointResilience(checkpoint_dir, True)],
    )
    """

    def __init__(self, enable_auto_reload: bool, checkpoint_dir: Optional[str] = None, *args, **kw):
        super().__init__(checkpoint_dir, *args, **kw)
        self._enable_auto_reload = enable_auto_reload
        # TODO: Update the "every_n_train_steps"
        self._every_n_train_steps = 1

    def on_train_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """
        Resuming from local checkpoints until it succeeds.
        """
        if self._enable_auto_reload:
            for saved_checkpoint_dir in self.saved_checkpoint_dirs:
                try:
                    logging.info(f"loading local checkpoint: {saved_checkpoint_dir}")
                    typ = SageMakerCheckpointType.LOCAL
                    path = os.path.join(saved_checkpoint_dir, self.local_checkpoint_sub_dir(trainer))
                    self._load_checkpoint(trainer, path, typ)
                    break
                except:
                    pass

    def local_checkpoint_sub_dir(self, trainer):
        tp_rank = f"tp{state.tp_rank}"
        ep_rank = f"ep{state.ep_rank}"
        fsdp_rank = f"fsdp{trainer.strategy.model.rank}"
        return "_".join([tp_rank, ep_rank, fsdp_rank])

    @property
    def saved_checkpoint_dirs(self):
        """
        Retreive all saved local checkpoint directories.
        """
        local_checkpoint_dir = os.path.join(self.checkpoint_dir, "local")
        saved_checkpoints = []
        if os.path.isdir(local_checkpoint_dir):
            saved_checkpoints = os.scandir(local_checkpoint_dir)
        return saved_checkpoints

    def _should_save_local(self, trainer: "pl.Trainer"):
        # TODO: check when should save local checkpoints for resilience
        is_last_step = trainer.max_steps == trainer.global_step
        is_every_n = trainer.global_step % self._every_n_train_steps == 0
        return is_last_step or is_every_n

    def _save_checkpoints(self, trainer: "pl.Trainer"):
        """
        Save local checkpiont if it should.
        """
        checkpoint_io = trainer.strategy.checkpoint_io
        assert isinstance(checkpoint_io, SageMakerCheckpointIO)

        if not self._should_save_local(trainer):
            return

        local_sub_dir = self.local_checkpoint_sub_dir(trainer)
        local_checkpoint_dir = os.path.join(self.checkpoint_dir, "local", f"steps_{trainer.global_step}", local_sub_dir)
        self._save(
            trainer, checkpoint_io, checkpoint_type=SageMakerCheckpointType.LOCAL, checkpoint_dir=local_checkpoint_dir
        )


class SageMakerCheckpoint(SageMakerModelCheckpointBase):
    """
    SageMakerCheckpoint Supports three types of checkpointing at each train batch. Potentially together:
    1. Full:
        This is configured to be saved every n train steps into exp_manager.export_full_model.
        ONLY model weights will be saved on rank 0.
        Note that selecting Full could be slow.
    2. Sharded:
        Save every n train steps in distributed checkpointing manner if save_top_k, monitor, mode and
        every_n_train_steps are provided in exp_manager.checkpoint_callback_params.
    """

    def __init__(self, cfg, *args, **kw):
        checkpoint_dir = cfg.exp_manager.get("checkpoint_dir", None)
        super().__init__(checkpoint_dir, *args, **kw)
        self._resume_from_checkpoint = cfg.exp_manager.get("resume_from_checkpoint", None)
        # Full checkpoint
        self._save_full_every_n_steps = None
        if "export_full_model" in cfg.exp_manager:
            self._save_full_every_n_steps = cfg.exp_manager.export_full_model.get("every_n_train_steps", None)
        # Sharded checkpoint
        checkpoint_callback_params = {}
        if "checkpoint_callback_params" in cfg.exp_manager:
            checkpoint_callback_params = cfg.exp_manager.checkpoint_callback_params
        self._save_sharded_every_n_steps = checkpoint_callback_params.get("every_n_train_steps", None)
        self._save_top_k = checkpoint_callback_params.get("save_top_k", None)
        self._monitor = checkpoint_callback_params.get("monitor", "step")
        mode = checkpoint_callback_params.get("mode", "max")
        assert (
            mode in [member.value for member in SageMakerMonitorMode],
            f"{mode} is not a valid value for {SageMakerMonitorMode.__name__}",
        )
        self._mode = (
            SageMakerMonitorMode.MAX if mode == SageMakerMonitorMode.MAX.value.lower() else SageMakerMonitorMode.MIN
        )
        self._best_k_models = []

    def on_train_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if self._resume_from_checkpoint:
            logging.info(f"loading sharded checkpoint: {self._resume_from_checkpoint}")
            typ = SageMakerCheckpointType.SHARDED
            path = self._resume_from_checkpoint
            sub_dir = f"tp{state.tp_rank}_ep{state.ep_rank}"
            sharded_checkpoint_dir = os.path.join(path, sub_dir)
            self._load_checkpoint(trainer, sharded_checkpoint_dir, typ)

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

    def _should_save_sharded(self, trainer: "pl.Trainer", monitor_candidates):
        """
        Make sure we need to save if all criterias are met:
        1. Hit every n steps
        2. Have the value in metric logged
        3. The new score is better.
        """

        if not self._save_sharded_every_n_steps or self._save_top_k < 1:
            return False

        is_every_n = trainer.global_step % self._save_sharded_every_n_steps == 0
        if not is_every_n:
            return False
        has_value = self._monitor in monitor_candidates
        if not has_value:
            m = (
                f"`SageMakerModelCheckpoint(monitor={self.monitor!r})` could not find the monitored key in the returned"
                f" metrics: {list(monitor_candidates)}."
                f" HINT: Did you call `log({self.monitor!r}, value)` in the `LightningModule`?"
            )
            logging.warn(m)
            return False

        # Check if it hits topk capacity. if hits, check if it is one of the topk.
        is_top_k = len(self._best_k_models) < self._save_top_k
        if len(self._best_k_models) == self._save_top_k and has_value:
            lowest = self._best_k_models[-1].score
            is_top_k = (
                lowest < monitor_candidates[self._monitor]
                if self._mode == SageMakerMonitorMode.MAX
                else lowest > monitor_candidates[self._monitor]
            )
        if not is_top_k:
            return False

        return True

    def _monitor_candidates(self, trainer: "pl.Trainer") -> Dict[str, Tensor]:
        """
        Retrieve the callback_metrics from trainer.
        """
        monitor_candidates = deepcopy(trainer.callback_metrics)
        # cast to int if necessary because `self.log("epoch", 123)` will convert it to float. if it's not a tensor
        # or does not exist we overwrite it as it's likely an error
        epoch = monitor_candidates.get("epoch")
        monitor_candidates["epoch"] = epoch.int() if isinstance(epoch, Tensor) else torch.tensor(trainer.current_epoch)
        step = monitor_candidates.get("step")
        monitor_candidates["step"] = step.int() if isinstance(step, Tensor) else torch.tensor(trainer.global_step)
        return monitor_candidates

    def _update_topk(self, new_checkpoint, checkpoint_io):
        """
        Update the topk models base on the metric value. Remove if needed.
        """
        reverse = self._mode == SageMakerMonitorMode.MAX
        self._best_k_models.append(new_checkpoint)
        # Sort the saved models so that the worst model is the end.
        self._best_k_models = sorted(
            self._best_k_models, key=lambda x: (x.score, (1 if reverse else -1) * x.step_at_save), reverse=reverse
        )

        if len(self._best_k_models) > self._save_top_k:
            path_to_remove = self._best_k_models[-1].checkpoint_path
            self._best_k_models.pop()
            checkpoint_io.get_checkpoint_io(SageMakerCheckpointType.SHARDED).remove_checkpoint(path_to_remove)

    def format_sharded_checkpoint_path(self, score):
        """
        Format the checkpoint saving dir with the format: {metric_name}_{score}.
        """
        value_format = "{value:d}"
        if isinstance(score, float) or (isinstance(score, torch.Tensor) and torch.is_floating_point(score)):
            value_format = "{value:.5f}"

        save_dir = "{monitor}_" + value_format

        save_dir = save_dir.format(monitor=self._monitor, value=score)
        return save_dir

    def _save_full(self, trainer: "pl.Trainer", path):
        path = os.path.join(path, "full", f"steps_{trainer.global_step}")
        return SageMakerCheckpointType.FULL, path

    def _save_sharded(self, trainer: "pl.Trainer", path, monitor_candidates):
        score = monitor_candidates.get(self._monitor)
        name = self.format_sharded_checkpoint_path(score)
        sub_dir = os.path.join(name, f"tp{state.tp_rank}_ep{state.ep_rank}")
        sharded_checkpoint_dir = os.path.join(path, "sharded", sub_dir)
        new_checkpoint = TopkCheckPoint(
            monitor=self._monitor,
            score=score,
            checkpoint_path=sharded_checkpoint_dir,
            step_at_save=trainer.global_step,
            epoch_at_save=trainer.current_epoch,
        )
        checkpoint_io = trainer.strategy.checkpoint_io
        self._update_topk(new_checkpoint, checkpoint_io)
        return SageMakerCheckpointType.SHARDED, sharded_checkpoint_dir

    def _save_checkpoints(self, trainer: "pl.Trainer"):
        """
        At each step, we check if we should save one checkpoint type.
        """
        checkpoint_io = trainer.strategy.checkpoint_io
        assert isinstance(checkpoint_io, SageMakerCheckpointIO)
        checkpoint_info = []
        checkpoint_dir = self.checkpoint_dir
        monitor_candidates = self._monitor_candidates(trainer)

        if self._should_save_sharded(trainer, monitor_candidates):
            checkpoint_info.append(self._save_sharded(trainer, checkpoint_dir, monitor_candidates))
        if self._should_save_full(trainer):
            checkpoint_info.append(self._save_full(trainer, checkpoint_dir))

        for checkpoint_type, checkpoint_dir in checkpoint_info:
            self._save(trainer, checkpoint_io, checkpoint_type, checkpoint_dir)
