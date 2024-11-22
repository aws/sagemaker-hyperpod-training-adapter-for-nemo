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
import shutil
import socket
from dataclasses import dataclass
from functools import wraps
from typing import Dict, Union

import hydra
import pytest
import torch
import torch.distributed as dist
import torch.sagemaker as tsm
from hydra import initialize
from nemo.utils import logging
from omegaconf import DictConfig
from omegaconf.omegaconf import OmegaConf
from torch.distributed._sharded_tensor import ShardedTensor
from torch.testing._internal.common_distributed import TEST_SKIPS

from hyperpod_nemo_adapter.collections.model.nlp import (
    SageMakerLlamaModel,
    SageMakerMistralModel,
    SageMakerMixtralModel,
)
from hyperpod_nemo_adapter.collections.parts import SageMakerTrainerBuilder
from hyperpod_nemo_adapter.constants import SageMakerCheckpointType
from hyperpod_nemo_adapter.utils.config_utils import validate_config
from hyperpod_nemo_adapter.utils.exp_manager import exp_manager
from hyperpod_nemo_adapter.utils.log_utils import Logger

_WORLD_SIZE = 8
_DEFAULT_TYPES = [
    SageMakerCheckpointType.LOCAL,
    SageMakerCheckpointType.SHARDED,
    SageMakerCheckpointType.FULL,
]


@dataclass
class SageMakerModel:
    model_type: str = ""
    model: Union[SageMakerLlamaModel, SageMakerMistralModel, SageMakerMixtralModel] = None
    model_config_name: str = ""


SAGEMAKER_TEST_MODEL_FACTORY = {
    "llama": SageMakerModel(
        model_type="llama",
        model=SageMakerLlamaModel,
        model_config_name="smp_llama_config",
    ),
    "llama_lora": SageMakerModel(
        model_type="llama",
        model=SageMakerLlamaModel,
        model_config_name="smp_llama_config_lora",
    ),
    "mistral": SageMakerModel(
        model_type="mistral",
        model=SageMakerMistralModel,
        model_config_name="smp_mistral_config",
    ),
    "mixtral": SageMakerModel(
        model_type="mixtral",
        model=SageMakerMixtralModel,
        model_config_name="smp_mixtral_config",
    ),
}


def skip_if_lt_x_gpu(x):
    """
    This is from torch/testing/_internal/common_distributed.py
    Instead of using sys.exit() which marks as FAIL, we use os.exit() to exit
    the process gracefully.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if torch.cuda.is_available() and torch.cuda.device_count() >= x:
                return func(*args, **kwargs)
            os.exit(TEST_SKIPS[f"multi-gpu-{x}"].exit_code)

        return wrapper

    return decorator


@validate_config(extra="allow")
def setup_test_cfg(cfg: DictConfig):
    """Set the update the cfg so that it can be used for testing in one node.

    Only the model related cfg should be set. Other test config params should override
    somewhere else.

    Args:
        cfg (DictConfig): The original config

    Returns:
        DictConfig: The updated config
    """

    # exp_manager
    cfg.exp_manager.resume_from_checkpoint = None
    # trainer
    cfg.trainer.max_steps = 2
    cfg.trainer.num_nodes = 1
    cfg.trainer.devices = 8
    cfg.trainer.limit_val_batches = 0

    # Model
    cfg.model.train_batch_size = 1
    cfg.model.max_context_width = 8
    cfg.model.max_position_embeddings = 8
    cfg.model.num_hidden_layers = 2
    cfg.model.vocab_size = 32
    cfg.model.hidden_size = 8
    cfg.model.num_attention_heads = 2
    cfg.model.intermediate_size = 8
    cfg.model.num_heads = 2

    if not cfg.model.get("rope_scaling", None):
        cfg.model.rope_scaling.original_max_position_embeddings = 8
        if cfg.model.rope_scaling.get("rope_type", None):
            cfg.model.rope_scaling.rope_type = "default"

    if cfg.model.model_type == "mistral":
        cfg.model.num_attention_heads = 1
        cfg.model.sliding_window = 8
        cfg.model.num_key_value_heads = 1

    cfg.model.shard_degree = _WORLD_SIZE
    if cfg.model.model_type == "mixtral":
        cfg.model.num_key_value_heads = 1
        cfg.model.rope_scaling.rope_type = "default"

    if cfg.use_smp_model:
        cfg.model.tensor_model_parallel_degree = 1
        cfg.model.expert_model_parallel_degree = 1
    cfg.model.do_finetune = False

    # data
    cfg.model.data.use_synthetic_data = True

    return cfg


def assert_state_dict_equal(
    state_dict_1: Dict[str, torch.Tensor],
    state_dict_2: Dict[str, torch.Tensor],
) -> bool:
    """Check if two state_dicts are the same."""
    if not isinstance(state_dict_1, dict):
        assert_values(state_dict_1, state_dict_2, "")
        return True
    assert len(state_dict_1) == len(state_dict_2), "state_dict must be the same size"

    assert set(state_dict_1.keys()) == set(state_dict_2.keys()), "state_dict keys do not match"

    for key, value_1 in state_dict_1.items():
        value_2 = state_dict_2[key]
        assert_values(value_1, value_2, key)
    return True


def assert_values(value_1, value_2, key=None):
    if key is None:
        key = ""
    if isinstance(value_1, ShardedTensor):
        for local_shard_1, local_shard_2 in zip(value_1.local_shards(), value_2.local_shards()):
            # Remove nan
            t1 = torch.nan_to_num(local_shard_1.tensor, nan=0.0)
            t2 = torch.nan_to_num(local_shard_2.tensor, nan=0.0)
            assert torch.equal(t1, t2), f"Key {key}'s shard does not match"
    elif isinstance(value_1, torch.Tensor):
        # Remove nan
        t1 = torch.nan_to_num(value_1, nan=0.0)
        t2 = torch.nan_to_num(value_2, nan=0.0)
        assert torch.equal(t1, t2), f"Key {key}'s tensor does not match"
    elif isinstance(value_1, int) or isinstance(value_1, float):
        assert value_1 == value_2, f"Key {key}'s value does not match"
    elif isinstance(value_1, dict):
        assert assert_state_dict_equal(value_1, value_2), f"Key {key}'s dict does not match"
    elif isinstance(value_1, list):
        for i in range(len(value_1)):
            assert assert_state_dict_equal(value_1[i], value_2[i]), f"Key {key}'s list does not match"


class TestCheckpoint:

    def config(self, model_type="llama"):
        model_base_type = SAGEMAKER_TEST_MODEL_FACTORY[model_type].model_type
        with initialize(version_base="1.2", config_path=f"../../../../../examples/{model_base_type}/conf"):
            cfg = hydra.compose(config_name=SAGEMAKER_TEST_MODEL_FACTORY[model_type].model_config_name)
            logging.debug("\n\n************** Experiment configuration ***********")
            logging.debug(f"\n{OmegaConf.to_yaml(cfg)}")
            cfg.exp_manager.exp_dir = "tmp"
            return setup_test_cfg(cfg)

    def create_and_fit(self, config, model_type="llama", callbacks=None, sample=None):
        """Create a trainer, model and datamoudle then run fit."""
        trainer, data_module = SageMakerTrainerBuilder(config).create_trainer()
        exp_manager(trainer, config.exp_manager)
        sagemaker_model = SAGEMAKER_TEST_MODEL_FACTORY[model_type].model
        model_module = sagemaker_model(config.model, trainer, use_smp_model=config.use_smp_model)
        if callbacks:
            if not isinstance(callbacks, list):
                callbacks = [callbacks]
            trainer.callbacks.extend(callbacks)
        # train
        trainer.fit(model_module, datamodule=data_module)

        # Run one manual step.
        outputs = {}
        if sample:
            outputs = self.run_step(trainer, model_module, sample)
        return trainer, data_module, model_module, outputs

    def check_correctness(self, state_dict1, state_dict2, data_module_key, is_full=False):
        """Check if the two state_dicts are the same.

        In particular, we check the following:

        In Full mode:
            1. model state_dict.

        Other modes:
            1. global_step.
            2. model state_dict.
            3. optimizer state_dict.
            4. lr_schedulers.
            5. data_module.

        Other loop related state_dicts are ommited, as we recreate the trainers.
        """
        assert_state_dict_equal(state_dict1["state_dict"], state_dict2["state_dict"])

        if not is_full:
            assert_state_dict_equal(state_dict1["global_step"], state_dict2["global_step"])

            for opt1, opt2 in zip(state_dict1["optimizer_states"], state_dict2["optimizer_states"]):
                assert_state_dict_equal(opt1, opt2)

            for lr1, lr2 in zip(state_dict1["lr_schedulers"], state_dict2["lr_schedulers"]):
                assert_state_dict_equal(lr1, lr2)

            for data_module1, data_module2 in zip(state_dict1[data_module_key], state_dict2[data_module_key]):
                assert_state_dict_equal(data_module1, data_module2)

    def update_checkpoint_config(self, config, checkpoint_param):
        """Update the checkpoint config to use the same model config as the training config."""
        save_top_k, sharded_save_last, auto_checkpoint, every_n_train_steps, save_full_last, peft_type = (
            checkpoint_param
        )
        # sharded
        config.exp_manager.checkpoint_callback_params.save_top_k = save_top_k
        config.exp_manager.checkpoint_callback_params.save_last = sharded_save_last
        config.exp_manager.checkpoint_callback_params.every_n_train_steps = 5

        # resilience
        config.exp_manager.auto_checkpoint.enabled = auto_checkpoint
        config.exp_manager.auto_checkpoint.warmup_steps = 0
        config.exp_manager.auto_checkpoint.drop_n_warmup_steps = 0

        # full
        config.exp_manager.export_full_model.every_n_train_steps = every_n_train_steps
        config.exp_manager.export_full_model.save_last = save_full_last

        # peft
        config.model.peft.peft_type = peft_type

        return config

    def update_checkpoint_config_with_type(self, config, type=None):
        # Default to turn off all checkpointing.
        # Each entry in checkpoint_param maps to
        # (save_top_k, sharded_save_last, auto_checkpoint,
        #  every_n_train_steps, save_full_last, peft_type) correspondingly.
        checkpoint_param = (0, False, False, 0, False, None)
        if type == SageMakerCheckpointType.FULL:
            checkpoint_param = (0, False, False, 5, True, None)
        elif type == SageMakerCheckpointType.SHARDED:
            checkpoint_param = (3, True, False, 0, False, None)
        elif type == SageMakerCheckpointType.LOCAL:
            checkpoint_param = (0, False, True, 0, False, None)
        self.update_checkpoint_config(config, checkpoint_param)

    def config_for_setup(self, temp_path):
        config = self.config()
        config.exp_manager.exp_dir = temp_path
        # Turn off all the checkpointing.
        return self.update_checkpoint_config(config, (0, False, False, 0, False, None))

    @pytest.fixture
    def cuda_available(self):
        """Check if cuda is available."""
        return torch.cuda.is_available() and torch.cuda.device_count() >= _WORLD_SIZE

    def generate_sample(self, config):
        """Generate a random sample."""
        vocab_size = config.model.vocab_size
        seqlen = config.model.max_context_width
        global_rank = dist.get_rank()
        sample_inputs = (
            torch.randint(
                vocab_size - 2 * global_rank,
                (
                    1,
                    seqlen,
                ),
                dtype=torch.long,
            )
            + global_rank
        )
        labels = sample_inputs + global_rank
        return sample_inputs, labels

    def run_step(self, trainer, model_module, sample):
        """Manually run a forward + backward.

        Then retrieve the loss, logits, and gradients.

        steps:
        1. forward
        2. zero grad
        3. backward
        Note: We don't run optimizer.step() here, as we want to keep the
        model + optimizer state_dict intact for comparison.
        """
        inputs, labels = sample
        outputs = model_module.forward(input_ids=inputs, labels=labels)
        loss = outputs["loss"]
        logits = outputs["logits"]

        trainer.optimizers[0].zero_grad()

        model_module.backward(loss)

        grads = []
        for param_group in trainer.optimizers[0].param_groups:
            for param in param_group["params"]:
                if param.grad is not None:
                    grads.append(param.grad.view(-1))
        return {"loss": loss, "logits": logits, "grads": torch.cat(grads).clone()}

    @pytest.fixture
    def temp_dir(self, tmp_path, cuda_available):
        """Create a temporary directory for the test.

        Here we use the rank0's pytest tmp_path as the tmp_dir.
        We do it by:
        1. setup the lighhtnigh trainer.
        2. we don't run anything but setup the process_group.
            Note that if we don't use pytorch lightning trainer to
            setup the process_group, Some of the ports will be occupied
            and caused hang.
        3. Logging is disabled during the process and re-enabled afterwards.
        4. We broadcast the temp_dir to all the ranks.
        """
        if not cuda_available:
            pytest.skip("CUDA is not available, skipping the test")

        # Disable logging during setup to avoid confusion.
        Logger().get_logger().disabled = True
        logging._logger.disabled = True
        config = self.config_for_setup(tmp_path)
        config.exp_manager.exp_dir = tmp_path
        config.trainer.max_steps = 0
        # Setting warmup_steps to be nonzeros to avoid division zero erro.
        config.model.optim.sched.warmup_steps = 1

        # Set up the process group in pytorch lightning, but no training happens.
        trainer, data_module, model_module, _ = self.create_and_fit(config)
        del trainer, data_module, model_module
        torch.cuda.empty_cache()

        if dist.get_rank() == 0:
            temp_dir = tmp_path
            print(f"Using temp directory: {temp_dir}")
        else:
            temp_dir = ""
        object_list = [temp_dir]

        # Broadcast temp_dir to all the other ranks
        dist.broadcast_object_list(object_list)
        temp_dir = object_list[0]

        # Re-enable logging after setup.
        Logger().get_logger().disabled = False
        logging._logger.disabled = False

        yield temp_dir
        if not dist.is_initialized():
            return
        if dist.get_rank() == 0:
            shutil.rmtree(temp_dir, ignore_errors=True)
        torch.cuda.empty_cache()
        dist.barrier()

    def retrieve_state_dicts(self, trainer, checkpoint_types=_DEFAULT_TYPES):
        state_dicts = []
        for checkpoint_type in checkpoint_types:
            trainer.strategy.checkpoint_io.checkpoint_type = checkpoint_type
            state_dicts.append(trainer._checkpoint_connector.dump_checkpoint())
        return state_dicts

    def reset_state_and_groups(self, port):
        """ "Reset the tsm state since shard degrees and world_size will be changing."""
        # Use a different address and port, since resuing the same
        # address and port will cause the NCCL to hang
        old_address = os.environ.get("MASTER_ADDR", "127.0.0.1")

        old_addresses = old_address.split(".")
        old_addresses[-1] = str(int(old_addresses[-1]) + 1)
        os.environ["MASTER_ADDR"] = ".".join(old_addresses)
        os.environ["MASTER_PORT"] = str(port)

        if not tsm.is_initialized() or not dist.is_initialized():
            return
        tsm.state.reset()
        dist.barrier()
        dist.destroy_process_group()

    def find_free_network_ports(self):
        """Finds two free port on localhost.

        It is useful in single-node training when we don't want to connect to a real main node but have to set the
        `MASTER_PORT` environment variable.

        """

        def find_port():
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.bind(("", 0))
            port = s.getsockname()[1]
            s.close()
            return port

        # we will need to find two ports.
        ports = set()
        while len(ports) < 2:
            ports.add(find_port())
        return list(ports)

    def broadcast_ports(self, ports):
        ports = self.find_free_network_ports()
        # broadcast to the other ranks from rank 0
        if dist.get_rank() == 0:
            p = ports
        else:
            p = []
        object_list = [p]

        # Broadcast ports from rank 0 to all the other ranks
        dist.broadcast_object_list(object_list)
        ports = object_list[0]
        return ports
