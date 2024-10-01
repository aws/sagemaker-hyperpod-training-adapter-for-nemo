import os
import shutil
from typing import Dict

import hydra
import pytest
import torch
import torch.distributed as dist
from hydra import initialize
from nemo.utils import logging
from omegaconf import DictConfig
from omegaconf.omegaconf import OmegaConf
from torch.distributed._sharded_tensor import ShardedTensor

from sagemaker_nemo_adaptor.collections.model.nlp import SageMakerLlamaModel
from sagemaker_nemo_adaptor.collections.parts import SageMakerTrainerBuilder
from sagemaker_nemo_adaptor.utils.config_utils import validate_config
from sagemaker_nemo_adaptor.utils.exp_manager import exp_manager


@validate_config()
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

    # Model
    cfg.model.train_batch_size = 1
    cfg.model.max_context_width = 256
    cfg.model.max_position_embeddings = 256
    cfg.model.num_layers = 2
    cfg.model.hidden_width = 256
    cfg.model.num_heads = 2
    cfg.model.intermediate_size = 256

    # data
    cfg.model.data.use_synthetic_data = True

    cfg.model.do_finetune = False

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


def assert_values(value_1, value_2, key):
    if isinstance(value_1, ShardedTensor):
        for local_shard_1, local_shard_2 in zip(value_1.local_shards(), value_2.local_shards()):
            assert torch.equal(local_shard_1.tensor, local_shard_2.tensor), f"Key {key}'s shard does not match"
    elif isinstance(value_1, torch.Tensor):
        assert torch.equal(value_1, value_2), f"Key {key}'s tensor does not match"
    elif isinstance(value_1, int) or isinstance(value_1, float):
        assert value_1 == value_2, f"Key {key}'s value does not match"
    elif isinstance(value_1, dict):
        assert assert_state_dict_equal(value_1, value_2), f"Key {key}'s dict does not match"
    elif isinstance(value_1, list):
        for i in range(len(value_1)):
            assert assert_state_dict_equal(value_1[i], value_2[i]), f"Key {key}'s list does not match"


class TestCheckpoint:

    def config(self):
        with initialize(version_base="1.2", config_path="../../../../../examples/llama/conf"):
            cfg = hydra.compose(config_name="smp_llama_config")
            logging.debug("\n\n************** Experiment configuration ***********")
            logging.debug(f"\n{OmegaConf.to_yaml(cfg)}")
            cfg.exp_manager.exp_dir = "tmp"
            return setup_test_cfg(cfg)

    @pytest.fixture
    def temp_dir(self, request):
        # TODO(htzhong): Find a better way to create the tmp and broadcast to the other ranks.
        # Currently the name of the tests are used to create the exp_dir as we don't have
        # the shared file system.
        path = request.param
        if os.path.exists(path):
            shutil.rmtree(path)
        os.mkdir(path)
        yield path
        if not dist.is_initialized():
            return
        logging.info("removing")
        if dist.get_rank() == 0:
            shutil.rmtree(path)
        dist.barrier()

    def create_and_fit(self, config):
        """Create a trainer, model and datamoudle then run fit."""
        trainer, data_module = SageMakerTrainerBuilder(config).create_trainer()
        exp_manager(trainer, config.exp_manager)
        model_module = SageMakerLlamaModel(config.model, trainer, use_smp=config.use_smp)
        # train
        trainer.fit(model_module, datamodule=data_module)
        return trainer, data_module, model_module

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
