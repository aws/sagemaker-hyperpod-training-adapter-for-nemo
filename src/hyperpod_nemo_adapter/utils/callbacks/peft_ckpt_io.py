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

import os
from datetime import timedelta
from typing import Any, Dict, Optional

import pytorch_lightning as pl
import torch.distributed as dist
import torch.sagemaker.distributed.checkpoint.state_dict_loader as loader
from lightning_fabric.utilities.types import _PATH
from nemo.utils import logging
from peft import PeftModel
from transformers import AutoModelForCausalLM

from hyperpod_nemo_adapter.patches import patch_llama_flash_attn_cp
from hyperpod_nemo_adapter.utils.callbacks.base_ckpt_io import SageMakerBaseCheckpointIO
from hyperpod_nemo_adapter.utils.callbacks.sharded_ckpt_io import (
    SageMakerShardedCheckpointIO,
)


class SageMakerPeftFullCheckpointIO(SageMakerBaseCheckpointIO):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)

    def save_checkpoint(self, checkpoint: Dict[str, Any], path: _PATH, storage_options: Optional[Any] = None) -> None:
        # this function may take a long time for large models. the save on rank 0 takes the vast majority of the time.
        # so if we have a nccl op (barrier) after this function, it will likely timeout while waiting for rank 0, causing a crash.
        # so create a new process group with a long timeout, and run barrier with this pg before returning.
        custom_timeout = timedelta(hours=6)
        pg = dist.new_group(backend="gloo", timeout=custom_timeout)

        trainer = storage_options
        assert isinstance(trainer, pl.Trainer)
        # Save the adapter weights
        trainer.strategy.save_peft_model(path)
        dist.barrier(group=pg)  # Wait for adapter save to finish
        # Save the fully merged model on rank 0 only
        if dist.get_rank() == 0:
            self._merge_and_upload_peft_model(trainer, path)
        dist.barrier(group=pg)  # Wait for merged save to finish
        dist.destroy_process_group(group=pg)

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

    def _merge_and_upload_peft_model(self, trainer: "pl.Trainer", checkpoint_dir: str, upload_to_storage=True):
        """Merge adapter weights with base model and upload final model"""
        hf_model_name_or_path = trainer.strategy.cfg.model.get("hf_model_name_or_path", None)
        access_token = trainer.strategy.cfg.model.get("hf_access_token", None)
        if hf_model_name_or_path is None:
            logging.warning("No pretrained model name or path found, could not upload final model.")
            return

        final_model_dir = os.path.join(checkpoint_dir, "final-model")

        logging.info(f"Loading Base model from : {hf_model_name_or_path}")
        # the patch for llama attention context parallel causes a crash here.
        # we don't need the patch since the model we are loading here is only used for merging weights.
        # therefore, we disable the patch before calling from_pretrained.
        is_patched = patch_llama_flash_attn_cp.is_patched
        if is_patched:
            patch_llama_flash_attn_cp.unapply_patch()
        base_model = AutoModelForCausalLM.from_pretrained(
            hf_model_name_or_path,
            attn_implementation="flash_attention_2",
            torch_dtype="auto",
            use_cache=False,
            device_map="cpu",
            token=access_token,
        )
        if is_patched:
            patch_llama_flash_attn_cp.apply_patch()
        logging.debug(f"Base model: {base_model}")

        peft_model = PeftModel.from_pretrained(base_model, checkpoint_dir, torch_device="cpu")
        logging.debug(f"Peft model after loading weights: {peft_model}")
        logging.info("Merging the adapter, this might take a while......")

        merged_model = peft_model.merge_and_unload(progressbar=True)
        logging.debug(f"Model after merging: {merged_model}")

        if upload_to_storage:
            logging.info(f"Checkpointing to {final_model_dir}......")
            merged_model.save_pretrained(final_model_dir)
            logging.info("Successfully save the merged model checkpoint.")

        return merged_model


class SageMakerPeftShardedCheckpointIO(SageMakerShardedCheckpointIO):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)

    def save_checkpoint(
        self,
        checkpoint: Dict[str, Any],
        path: _PATH,
        storage_options: Optional[Any] = None,
    ) -> None:
        # Save everything else but the model state_dict in sharded mode
        checkpoint.pop("state_dict")
        super().save_checkpoint(checkpoint, path, storage_options)
        # Save adapter weights separately
        trainer = storage_options
        assert isinstance(trainer, pl.Trainer)
        # Save adapter weights to parent dir
        parent_dir = os.path.dirname(path)
        trainer.strategy.save_peft_model(parent_dir)

    def load_checkpoint(
        self,
        path: _PATH,
        trainer: "pl.Trainer",
        map_location: Optional[Any] = None,
    ) -> Dict[str, Any]:
        assert trainer, "Bad parameter, trainer is empty"
        state_dict = trainer._checkpoint_connector.dump_checkpoint(False)
        state_dict.pop("optimizer_states")
        state_dict.pop("state_dict")
        loader.load(
            state_dict,
            checkpoint_id=path,
            process_group=self.app_state.fsdp_process_group,
            coordinator_rank=self.app_state.fsdp_coordinator_rank,
        )
        self.load_data_module_and_lr_schedulers(trainer, state_dict)
        logging.info(f"Loaded Sharded checkpoint for PEFT")
        return state_dict
