# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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
# Modifications (c) Amazon.com, Inc

import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from shutil import copy
from typing import Dict, Optional, Union

from nemo.collections.common.callbacks import EMA
from nemo.utils import logging
from nemo.utils.exp_manager import CallbackParams, EarlyStopping
from nemo.utils.exp_manager import ExpManagerConfig as NeMoExpManagerConfig
from nemo.utils.exp_manager import (
    StatelessTimer,
    Timer,
    TimingCallback,
    configure_loggers,
    configure_no_restart_validation_training_loop,
    get_git_diff,
    get_git_hash,
    get_log_dir,
)
from nemo.utils.lightning_logger_patch import add_filehandlers_to_pl_logger
from omegaconf import DictConfig, OmegaConf

from hyperpod_nemo_adapter.utils.app_state import SageMakerAppState
from hyperpod_nemo_adapter.utils.get_rank import is_global_rank_zero


@dataclass
class SageMakerExportFullModel(CallbackParams):
    every_n_train_steps = 0
    save_last = True
    final_export_dir: Optional[str] = None


@dataclass
class SageMakerAutoCheckpoint:
    enabled: bool = False
    warmup_steps: int = 12
    drop_n_warmup_steps: int = 3


@dataclass
class ExpManagerConfig(NeMoExpManagerConfig):
    log_reduced_training_loss: Optional[bool] = True
    export_full_model: Optional[SageMakerExportFullModel] = field(default_factory=lambda: SageMakerExportFullModel())
    checkpoint_dir: Optional[str] = None
    auto_checkpoint: Optional[SageMakerAutoCheckpoint] = field(default_factory=SageMakerAutoCheckpoint)
    log_step_timing: Optional[bool] = False


def exp_manager(trainer: "pytorch_lightning.Trainer", cfg: Optional[Union[DictConfig, Dict]] = None) -> Optional[Path]:
    """
    exp_manager is a helper function used to manage folders for experiments. It follows the pytorch lightning paradigm
    of exp_dir/model_or_experiment_name/version. If the lightning trainer has a logger, exp_manager will get exp_dir,
    name, and version from the logger. Otherwise it will use the exp_dir and name arguments to create the logging
    directory. exp_manager also allows for explicit folder creation via explicit_log_dir.

    The version can be a datetime string or an integer. Datestime version can be disabled if use_datetime_version is set
    to False. It optionally creates TensorBoardLogger, WandBLogger, DLLogger, MLFlowLogger, ClearMLLogger,
    ModelCheckpoint objects from pytorch lightning.
    It copies sys.argv, and git information if available to the logging directory. It creates a log file for each
    process to log their output into.

    exp_manager additionally has a resume feature (resume_if_exists) which can be used to continuing training from
    the constructed log_dir. When you need to continue the training repeatedly (like on a cluster which you need
    multiple consecutive jobs), you need to avoid creating the version folders. Therefore from v1.0.0, when
    resume_if_exists is set to True, creating the version folders is ignored.
    """

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    global_rank = trainer.node_rank * trainer.num_devices + local_rank
    logging.rank = global_rank

    if cfg is None:
        logging.error("exp_manager did not receive a cfg argument. It will be disabled.")
        return
    if trainer.fast_dev_run:
        logging.info("Trainer was called with fast_dev_run. exp_manager will return without any functionality.")
        return

    # Ensure passed cfg is compliant with ExpManagerConfig
    schema = OmegaConf.structured(ExpManagerConfig)
    if isinstance(cfg, dict):
        cfg = OmegaConf.create(cfg)
    elif not isinstance(cfg, DictConfig):
        raise ValueError(f"cfg was type: {type(cfg)}. Expected either a dict or a DictConfig")
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    cfg = OmegaConf.merge(schema, cfg)

    log_dir, exp_dir, name, version = get_log_dir(
        trainer=trainer,
        exp_dir=cfg.exp_dir,
        name=cfg.name,
        version=cfg.version,
        explicit_log_dir=cfg.explicit_log_dir,
        use_datetime_version=cfg.use_datetime_version,
        resume_if_exists=cfg.resume_if_exists,
    )

    checkpoint_name = name
    # If name returned from get_log_dir is "", use cfg.name for checkpointing
    if checkpoint_name is None or checkpoint_name == "":
        checkpoint_name = cfg.name or "default"

    # Set mlflow name if it's not set, before the main name is erased
    if cfg.create_mlflow_logger and (not cfg.mlflow_logger_kwargs.get("experiment_name", None)):
        cfg.mlflow_logger_kwargs.experiment_name = cfg.name
        logging.warning(
            "mlflow logger specified but no experiment name set. Using the same as Tensorboard: %s",
            cfg.mlflow_logger_kwargs.experiment_name,
        )

    cfg.name = name  # Used for configure_loggers so that the log_dir is properly set even if name is ""
    cfg.version = version

    # update app_state with log_dir, exp_dir, etc
    app_state = SageMakerAppState()
    app_state.log_dir = log_dir
    app_state.exp_dir = exp_dir
    app_state.name = name
    app_state.version = version
    app_state.checkpoint_name = checkpoint_name
    app_state.create_checkpoint_callback = cfg.create_checkpoint_callback
    app_state.checkpoint_callback_params = cfg.checkpoint_callback_params

    # Create the logging directory if it does not exist
    os.makedirs(log_dir, exist_ok=True)  # Cannot limit creation to global zero as all ranks write to own log file
    logging.info(f"Experiments will be logged at {log_dir}")
    trainer._default_root_dir = log_dir

    if cfg.log_local_rank_0_only is True and cfg.log_global_rank_0_only is True:
        raise ValueError(
            f"Cannot set both log_local_rank_0_only and log_global_rank_0_only to True. Please set either one or neither."
        )

    # Handle logging to file, overriding log dir and remove Nemo Testing flags
    log_file = log_dir / f"sagemaker_log_globalrank-{global_rank}_localrank-{local_rank}.txt"
    if cfg.log_local_rank_0_only is True:
        if local_rank == 0:
            logging.add_file_handler(log_file)
    elif cfg.log_global_rank_0_only is True:
        if global_rank == 0:
            logging.add_file_handler(log_file)
    else:
        # Logs on all ranks.
        logging.add_file_handler(log_file)

    # For some reason, LearningRateLogger requires trainer to have a logger. Safer to create logger on all ranks
    # not just global rank 0.
    if (
        cfg.create_tensorboard_logger
        or cfg.create_wandb_logger
        or cfg.create_mlflow_logger
        or cfg.create_dllogger_logger
        or cfg.create_clearml_logger
        or cfg.create_neptune_logger
    ):
        configure_loggers(
            trainer,
            exp_dir,
            log_dir,
            cfg.name,
            cfg.version,
            cfg.checkpoint_callback_params,
            cfg.create_tensorboard_logger,
            cfg.summary_writer_kwargs,
            cfg.create_wandb_logger,
            cfg.wandb_logger_kwargs,
            cfg.create_mlflow_logger,
            cfg.mlflow_logger_kwargs,
            cfg.create_dllogger_logger,
            cfg.dllogger_logger_kwargs,
            cfg.create_clearml_logger,
            cfg.clearml_logger_kwargs,
            cfg.create_neptune_logger,
            cfg.neptune_logger_kwargs,
        )

    # add loggers timing callbacks
    if cfg.log_step_timing:
        timing_callback = TimingCallback(timer_kwargs=cfg.step_timing_kwargs or {})
        trainer.callbacks.insert(0, timing_callback)

    if cfg.ema.enable:
        ema_callback = EMA(
            decay=cfg.ema.decay,
            validate_original_weights=cfg.ema.validate_original_weights,
            cpu_offload=cfg.ema.cpu_offload,
            every_n_steps=cfg.ema.every_n_steps,
        )
        trainer.callbacks.append(ema_callback)

    if cfg.create_early_stopping_callback:
        early_stop_callback = EarlyStopping(**cfg.early_stopping_callback_params)
        trainer.callbacks.append(early_stop_callback)

    if cfg.disable_validation_on_resume:
        # extend training loop to skip initial validation when resuming from checkpoint
        configure_no_restart_validation_training_loop(trainer)
    # Setup a stateless timer for use on clusters.
    if cfg.max_time_per_run is not None:
        found_ptl_timer = False
        for idx, callback in enumerate(trainer.callbacks):
            if isinstance(callback, Timer):
                # NOTE: PTL does not expose a `trainer.max_time`. By the time we are in this function, PTL has already setup a timer if the user specifies `trainer.max_time` so best we can do is replace that.
                # Working: If only `trainer.max_time` is set - it behaves as a normal PTL timer. If only `exp_manager.max_time_per_run` is set - it behaves as a StateLessTimer. If both are set, it also behaves as a StateLessTimer.
                logging.warning(
                    f"Found a PTL Timer callback, replacing with a StatelessTimer callback. This will happen if you set trainer.max_time as well as exp_manager.max_time_per_run."
                )
                trainer.callbacks[idx] = StatelessTimer(cfg.max_time_per_run)
                found_ptl_timer = True
                break

        if not found_ptl_timer:
            trainer.max_time = cfg.max_time_per_run
            trainer.callbacks.append(StatelessTimer(cfg.max_time_per_run))

    if is_global_rank_zero():
        # Move files_to_copy to folder and add git information if present
        if cfg.files_to_copy:
            for _file in cfg.files_to_copy:
                copy(Path(_file), log_dir)

        # Create files for cmd args and git info
        with open(log_dir / "cmd-args.log", "w", encoding="utf-8") as _file:
            _file.write(" ".join(sys.argv))

        # Try to get git hash
        git_repo, git_hash = get_git_hash()
        if git_repo:
            with open(log_dir / "git-info.log", "w", encoding="utf-8") as _file:
                _file.write(f"commit hash: {git_hash}")
                _file.write(get_git_diff())

        # Add err_file logging to global_rank zero
        logging.add_err_file_handler(log_dir / "nemo_error_log.txt")

        # Add lightning file logging to global_rank zero
        add_filehandlers_to_pl_logger(log_dir / "lightning_logs.txt", log_dir / "nemo_error_log.txt")

    elif trainer.num_nodes * trainer.num_devices > 1:
        # sleep other ranks so rank 0 can finish
        # doing the initialization such as moving files
        time.sleep(cfg.seconds_to_sleep)

    return log_dir
