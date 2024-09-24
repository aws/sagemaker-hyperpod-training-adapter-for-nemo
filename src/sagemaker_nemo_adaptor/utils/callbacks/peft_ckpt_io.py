import os
from typing import Any, Dict, Optional

import pytorch_lightning as pl
import torch.distributed as dist
from lightning_fabric.utilities.types import _PATH
from nemo.utils import logging
from peft import PeftModel
from transformers import AutoModelForCausalLM

from sagemaker_nemo_adaptor.utils.callbacks.base_ckpt_io import (
    SageMakerBaseCheckpointIO,
)


class SageMakerPeftCheckpointIO(SageMakerBaseCheckpointIO):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)

    def save_checkpoint(self, checkpoint: Dict[str, Any], path: _PATH, storage_options: Optional[Any] = None) -> None:
        trainer = storage_options
        assert isinstance(trainer, pl.Trainer)
        # Save the adapter weights
        trainer.strategy.save_peft_model(path)
        # Save the fully merged model on rank 0
        if dist.get_rank() == 0:
            self._merge_and_upload_peft_model(trainer, path)

    def load_checkpoint(
        self,
        path: _PATH,
        trainer: "pl.Trainer",
        map_location: Optional[Any] = None,
    ) -> Dict[str, Any]:
        pass

    def remove_checkpoint(self, path: _PATH) -> None:
        pass

    def teardown(self):
        pass

    def _merge_and_upload_peft_model(self, trainer: "pl.Trainer", checkpoint_dir: str):
        """Merge adapter weights with base model and upload final model"""
        hf_model_name_or_path = trainer.strategy.cfg.model.get("hf_model_name_or_path", None)
        if hf_model_name_or_path is None:
            logging.warning("No pretrained model name or path found, could not upload final model.")
            return

        final_model_dir = os.path.join(checkpoint_dir, "final-model")

        logging.info(f"Loading Base model from : {hf_model_name_or_path}")
        base_model = AutoModelForCausalLM.from_pretrained(
            hf_model_name_or_path,
            attn_implementation="flash_attention_2",
            torch_dtype="auto",
            use_cache=False,
            device_map="cpu",
        )
        logging.debug(f"Base model: {base_model}")

        peft_model = PeftModel.from_pretrained(base_model, checkpoint_dir)
        logging.debug(f"Peft model after loading weights: {peft_model}")
        logging.info("Merging the adapter, this might take a while......")

        merged_model = peft_model.merge_and_unload()
        logging.debug(f"Model after merging: {merged_model}")
        logging.info(f"Checkpointing to {final_model_dir}......")
        merged_model.save_pretrained(final_model_dir)
        logging.info("Successfully save the merged model checkpoint.")
