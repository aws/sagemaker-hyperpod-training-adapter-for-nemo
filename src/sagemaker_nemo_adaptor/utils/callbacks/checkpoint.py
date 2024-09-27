import math
import os
import time
from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, Optional, Union

import pytorch_lightning as pl
import torch
import torch.distributed as dist
from nemo.utils import logging
from pytorch_lightning.callbacks import Checkpoint
from torch import Tensor
from torch.sagemaker import state
from torch.sagemaker.distributed.checkpoint.s3_filesystem import is_s3_uri

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
        trainer.strategy.load_model_state_dict(state_dict)
        trainer.strategy.load_optimizer_state_dict(trainer, state_dict, path)
        self.load_state_dict(state_dict)
        trainer._checkpoint_connector._loaded_checkpoint = state_dict
        trainer._checkpoint_connector.restore_loops()
        trainer._checkpoint_connector._loaded_checkpoint = None

    def _save(
        self,
        trainer: "pl.Trainer",
        checkpoint_io: SageMakerCheckpointIO,
        checkpoint_type: SageMakerCheckpointType,
        checkpoint_dir: str,
    ):
        """Save one checkpoint using corresponding checkpoint type."""
        checkpoint_io.checkpoint_type = checkpoint_type
        weights_only = (
            checkpoint_type == SageMakerCheckpointType.FULL or checkpoint_type == SageMakerCheckpointType.PEFT_FULL
        )
        # trainer.save_checkpoint will call barrier which lead to have extra cost.
        # https://github.com/Lightning-AI/pytorch-lightning/blob/builds/2.3.3/src/lightning/pytorch/trainer/trainer.py#L1371
        checkpoint = trainer._checkpoint_connector.dump_checkpoint(weights_only)
        trainer.strategy.save_checkpoint(checkpoint, checkpoint_dir, trainer)


class _IntervalDecisionMaker:
    def __init__(self, warmup_steps=12, drop_n_warmup_steps=3, interval_guard=1.25):
        self.warmup_steps = warmup_steps
        self.drop_n_warmup_steps = drop_n_warmup_steps
        self.interval_guard = interval_guard
        self.reset()

    def reset(self):
        self.step = 0
        self.min_step_duration = float("inf")
        self.max_ckpt_duration = float("-inf")
        self._start = -1
        self._end = -1
        self._interval = 0

    def start(self):
        self._start = time.perf_counter()

    def end(self):
        self._end = time.perf_counter()
        self.step += 1

    def get_interval(self, max_ckpt_duration):
        if self._interval > 0:
            return self._interval
        if self.step < self.drop_n_warmup_steps:
            return 1

        step_duration = self._end - self._start
        self.min_step_duration = min(self.min_step_duration, step_duration)
        self.max_ckpt_duration = max(self.max_ckpt_duration, max_ckpt_duration)
        if self.step <= self.warmup_steps:
            return 1

        # update interval once. In order to save communication time, we compute
        # min training step time and max ckeckpoint time locally during warmup.
        # Once final warmup step is hit, we update interval by using all_gather
        # one time to collect global max I/O and min training duration across
        # all ranks.
        device = torch.cuda.current_device()
        world_size = dist.get_world_size()
        dtype = torch.float16
        tensor_list = [torch.zeros(2, dtype=dtype, device=device) for _ in range(world_size)]
        tensor = torch.tensor([self.max_ckpt_duration, self.min_step_duration], dtype=dtype, device=device)
        dist.all_gather(tensor_list, tensor)
        max_ckpt_duration = max(tensor[0].item() for tensor in tensor_list)
        min_step_duration = min(tensor[1].item() for tensor in tensor_list)

        self.min_step_duration = min_step_duration
        self.max_ckpt_duration = max_ckpt_duration
        interval = int(math.ceil(max_ckpt_duration / min_step_duration * self.interval_guard))
        self._interval = int(max(interval, 1))
        logging.info(f"Warmup hit. {self}")
        return self._interval

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f"step_duration: {self.min_step_duration}, ckpt_duration: {self.max_ckpt_duration}, interval = {self._interval}"


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

    def __init__(
        self,
        enable_auto_reload: bool,
        checkpoint_dir: Optional[str] = None,
        warmup_steps: int = 12,
        drop_n_warmup_steps: int = 3,
        interval_guard: float = 1.25,
        *args,
        **kw,
    ):
        # If user specify a path to resume, disable auto resume.
        super().__init__(checkpoint_dir, *args, **kw)
        self._enable_auto_reload = enable_auto_reload
        self._every_n_train_steps = 1
        self._num_checkpoint = 0
        self._interval_decision_maker = _IntervalDecisionMaker(warmup_steps, drop_n_warmup_steps, interval_guard)

    @property
    def save_top_k(self):
        # Set save_top_k to three is safer. The reason is that the worst case
        # is new checkpoint start but old checkpoint is still writing. If both
        # checkpoint corrupt, we don't have any checkpoint.
        return 3

    def on_train_batch_start(
        self,
        trainer: "pl.Trainer",
        *args,
        **kwargs,
    ) -> None:
        super().on_train_batch_start(trainer, *args, **kwargs)
        self._interval_decision_maker.start()

    def on_train_batch_end(
        self,
        trainer: "pl.Trainer",
        *args,
        **kwargs,
    ) -> None:
        self._interval_decision_maker.end()
        super().on_train_batch_end(trainer, *args, **kwargs)

        # update every_n_train_steps
        checkpoint_io = trainer.strategy.checkpoint_io
        assert isinstance(checkpoint_io, SageMakerCheckpointIO)
        typ = SageMakerCheckpointType.LOCAL
        checkpoint_io = trainer.strategy.checkpoint_io[typ]
        max_ckpt_duration = checkpoint_io.profiler.max_duration
        interval = self._interval_decision_maker.get_interval(max_ckpt_duration)
        self._every_n_train_steps = interval

    def on_train_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """Resuming from local checkpoints until it succeeds."""
        if self._enable_auto_reload:
            try:
                path = os.path.join(self.checkpoint_dir, "local")
                self._load_checkpoint(trainer, path, SageMakerCheckpointType.LOCAL)
            except FileNotFoundError:
                logging.warning("checkpoint not found.")

    def _should_save_local(self, trainer: "pl.Trainer"):
        is_last_step = trainer.max_steps == trainer.global_step
        is_every_n = trainer.global_step % self._every_n_train_steps == 0
        return is_last_step or is_every_n

    def _save_checkpoints(self, trainer: "pl.Trainer"):
        """Save local checkpiont if it should."""
        checkpoint_io = trainer.strategy.checkpoint_io
        assert isinstance(checkpoint_io, SageMakerCheckpointIO)

        if not self._should_save_local(trainer):
            return

        dir_name = f"{self._num_checkpoint % self.save_top_k}"
        local_checkpoint_dir = os.path.join(self.checkpoint_dir, "local", dir_name)
        self._save(
            trainer,
            checkpoint_io,
            checkpoint_type=SageMakerCheckpointType.LOCAL,
            checkpoint_dir=local_checkpoint_dir,
        )
        self._num_checkpoint += 1

    def load_state_dict(self, state_dict):
        state_dict = state_dict.get("callbacks", {})
        if self.state_key in state_dict:
            state_dict = state_dict[self.state_key]
            min_step_duration = self._interval_decision_maker.min_step_duration
            max_ckpt_duration = self._interval_decision_maker.max_ckpt_duration
            self._interval_decision_maker._interval = state_dict.get("interval", 0)
            self._interval_decision_maker.min_step_duration = state_dict.get("min_step_duration", min_step_duration)
            self._interval_decision_maker.max_ckpt_duration = state_dict.get("max_ckpt_duration", max_ckpt_duration)
            self._interval_decision_maker.step = state_dict.get("interval_decision_maker_step", 0)
            self._every_n_train_steps = self._interval_decision_maker.get_interval(max_ckpt_duration)
            logging.info(f"load interval_decision_maker: {self._interval_decision_maker}")

    def state_dict(self):
        max_ckpt_duration = self._interval_decision_maker.max_ckpt_duration
        min_step_duration = self._interval_decision_maker.min_step_duration
        interval = self._interval_decision_maker._interval
        step = self._interval_decision_maker.step
        return {
            "interval": interval,
            "max_ckpt_duration": max_ckpt_duration,
            "min_step_duration": min_step_duration,
            "interval_decision_maker_step": step,
        }


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
        self._save_last_full = None
        if "export_full_model" in cfg.exp_manager:
            self._save_full_every_n_steps = cfg.exp_manager.export_full_model.get("every_n_train_steps", None)
            self._save_last_full = cfg.exp_manager.export_full_model.get("save_last", True)
        # Sharded checkpoint
        checkpoint_callback_params = {}
        if "checkpoint_callback_params" in cfg.exp_manager:
            checkpoint_callback_params = cfg.exp_manager.checkpoint_callback_params
        self._save_sharded_every_n_steps = checkpoint_callback_params.get("every_n_train_steps", None)
        self._save_last_sharded = checkpoint_callback_params.get("save_last", True)
        self._save_top_k = checkpoint_callback_params.get("save_top_k", None)
        self._monitor = checkpoint_callback_params.get("monitor", "step")
        mode = checkpoint_callback_params.get("mode", "max")
        assert mode in [
            member.value for member in SageMakerMonitorMode
        ], f"{mode} is not a valid value for {SageMakerMonitorMode.__name__}"
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
        2. reach max steps if export_full_model.save_last = True.
        """
        is_last_step = trainer.max_steps == trainer.global_step
        if self._save_last_full and is_last_step:
            return True
        if not self._save_full_every_n_steps:
            return False
        return trainer.global_step % self._save_full_every_n_steps == 0

    def _should_save_sharded(self, trainer: "pl.Trainer", monitor_candidates):
        """
        Make sure we need to save if all criterias are met:
        1. Hit every n steps
        2. Have the value in metric logged
        3. The new score is better.
        """
        save_last_step = trainer.max_steps == trainer.global_step and self._save_last_sharded
        is_sharded_on = self._save_sharded_every_n_steps and self._save_top_k >= 1
        is_every_n = trainer.global_step % self._save_sharded_every_n_steps == 0
        # Neither saving last step nor every n step is needed.
        if not save_last_step and not (is_sharded_on and is_every_n):
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
        lowest = -math.inf if self._mode == SageMakerMonitorMode.MAX else math.inf
        if len(self._best_k_models) == self._save_top_k and has_value:
            if len(self._best_k_models):
                lowest = self._best_k_models[-1].score
            else:
                lowest = -math.inf if self._mode == SageMakerMonitorMode.MAX else math.inf
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

    def _update_topk(self, new_checkpoint, checkpoint_io, checkpoint_type):
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
            checkpoint_io[checkpoint_type].remove_checkpoint(path_to_remove)

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
        if trainer.max_steps != trainer.global_step:
            self._update_topk(new_checkpoint, checkpoint_io, SageMakerCheckpointType.SHARDED)
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


class SageMakerCheckpointPeft(SageMakerCheckpoint):
    """
    Class to Support checkpointing for PEFT models.
    """

    def __init__(self, cfg, *args, **kw):
        super().__init__(cfg, *args, **kw)
        self._is_peft = hasattr(cfg.model, "peft") and cfg.model.peft.peft_type is not None
        assert self._is_peft, "SageMakerCheckpointPeft should only be used for PEFT models"
        assert not is_s3_uri(self._checkpoint_dir), "PEFT checkpointing does not support saving to S3"

    def _load_checkpoint(self, trainer, path, typ):
        """
        Load checkpoint from a given path with the given checkpoint type.
        """
        checkpoint_io = trainer.strategy.checkpoint_io
        assert isinstance(checkpoint_io, SageMakerCheckpointIO)
        checkpoint_io.checkpoint_type = typ
        state_dict = trainer.strategy.load_checkpoint(path, trainer)
        # Skip loading model state_dict as this gets loaded at model creation
        trainer.strategy.load_optimizer_state_dict(trainer, state_dict, path)

    def on_train_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if self._resume_from_checkpoint:
            logging.info(f"loading peft_sharded checkpoint: {self._resume_from_checkpoint}")
            typ = SageMakerCheckpointType.PEFT_SHARDED
            path = self._resume_from_checkpoint
            sub_dir = f"tp{state.tp_rank}_ep{state.ep_rank}"
            sharded_checkpoint_dir = os.path.join(path, sub_dir)
            self._load_checkpoint(trainer, sharded_checkpoint_dir, typ)

    def _save_peft_sharded(self, trainer: "pl.Trainer", path, monitor_candidates):
        score = monitor_candidates.get(self._monitor)
        name = self.format_sharded_checkpoint_path(score)
        sub_dir = os.path.join(name, f"tp{state.tp_rank}_ep{state.ep_rank}")
        sharded_checkpoint_dir = os.path.join(path, "peft_sharded", sub_dir)
        # Get parent_dir so pruning will remove the entire checkpoint directory at step_x
        parent_dir = os.path.dirname(sharded_checkpoint_dir)
        new_checkpoint = TopkCheckPoint(
            monitor=self._monitor,
            score=score,
            checkpoint_path=parent_dir,
            step_at_save=trainer.global_step,
            epoch_at_save=trainer.current_epoch,
        )
        checkpoint_io = trainer.strategy.checkpoint_io
        self._update_topk(new_checkpoint, checkpoint_io, SageMakerCheckpointType.PEFT_SHARDED)
        return SageMakerCheckpointType.PEFT_SHARDED, sharded_checkpoint_dir

    def _save_peft_full(self, trainer: "pl.Trainer", path):
        path = os.path.join(path, "peft_full", f"steps_{trainer.global_step}")
        return SageMakerCheckpointType.PEFT_FULL, path

    def _save_checkpoints(self, trainer: "pl.Trainer"):
        """
        At each step, we check if we should save Peft checkpoint.
        """
        checkpoint_io = trainer.strategy.checkpoint_io
        assert isinstance(checkpoint_io, SageMakerCheckpointIO)
        checkpoint_dir = self.checkpoint_dir
        checkpoint_info = []
        monitor_candidates = self._monitor_candidates(trainer)

        if self._should_save_sharded(trainer, monitor_candidates):
            checkpoint_info.append(self._save_peft_sharded(trainer, checkpoint_dir, monitor_candidates))
        # Note that PEFT FULL checkpoints reuse the configs for saving regular FULL checkpoints
        if self._should_save_full(trainer):
            checkpoint_info.append(self._save_peft_full(trainer, checkpoint_dir))

        for checkpoint_type, checkpoint_dir in checkpoint_info:
            self._save(trainer, checkpoint_io, checkpoint_type, checkpoint_dir)
