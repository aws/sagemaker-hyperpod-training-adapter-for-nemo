import shutil
from typing import Any, Dict, Optional

import pytorch_lightning as pl
import torch.sagemaker.distributed.checkpoint.state_dict_loader as loader
import torch.sagemaker.distributed.checkpoint.state_dict_saver as saver
from lightning_fabric.plugins import CheckpointIO
from lightning_fabric.utilities.types import _PATH

from sagemaker_nemo_adaptor.utils.get_rank import get_coordinator_rank, is_action_rank


class SageMakerShardedCheckpointIO(CheckpointIO):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)

    def save_checkpoint(
        self,
        checkpoint: Dict[str, Any],
        path: _PATH,
        storage_options: Optional[Any] = None,
    ) -> None:
        trainer = storage_options
        assert isinstance(trainer, pl.Trainer)
        strategy = trainer.strategy
        group = strategy.model.process_group
        saver.maybe_finalize_async_calls(blocking=True, process_group=group)
        coordinator_rank = get_coordinator_rank(group)
        if is_action_rank(strategy.global_rank):
            saver.async_save(
                checkpoint,
                checkpoint_id=path,
                process_group=group,
                coordinator_rank=coordinator_rank,
            )

    def load_checkpoint(
        self,
        path: _PATH,
        trainer: "pl.Trainer",
        map_location: Optional[Any] = None,
    ) -> Dict[str, Any]:
        assert trainer, "Bad parameter, trainer is empty"
        state_dict = trainer._checkpoint_connector.dump_checkpoint(False)
        state_dict.pop("optimizer_states")
        loader.load(state_dict, checkpoint_id=path)
        return state_dict

    def remove_checkpoint(self, path: _PATH) -> None:
        shutil.rmtree(path, ignore_errors=True)

    def teardown(self, trainer):
        strategy = trainer.strategy
        group = strategy.model.process_group
        saver.maybe_finalize_async_calls(blocking=True, process_group=group)
