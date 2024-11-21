from lightning_fabric.plugins import CheckpointIO
from torch.sagemaker.distributed.checkpoint.s3_filesystem import (
    format_s3_path,
    get_s3_region_from_uri,
    is_s3_uri,
    parse_s3_uri,
)

from hyperpod_nemo_adapter.utils.app_state import SageMakerAppState


class SageMakerBaseCheckpointIO(CheckpointIO):
    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.app_state = SageMakerAppState()

    def _load_data_module(self, trainer, state_dict):
        """
        datamodule.load_state_dict will loop over dataloader and load from state_dict.
        """
        trainer.datamodule.load_state_dict(state_dict)

    def _load_lr_schedulers(self, trainer, state_dict):
        """
        Loop over lr_schedulers and load state_dict.
        """
        for i, config in enumerate(trainer.lr_scheduler_configs):
            config.scheduler.load_state_dict(state_dict["lr_schedulers"][i])

    def load_data_module_and_lr_schedulers(self, trainer, state_dict):
        """
        Load both data_module and lr_schedulers.
        """
        self._load_data_module(trainer, state_dict)
        self._load_lr_schedulers(trainer, state_dict)
