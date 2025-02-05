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
from nemo.utils import logging
logging.info('Penrose before imports on deepseek_pretrain.py')

import hydra
from omegaconf import DictConfig
from omegaconf.omegaconf import OmegaConf

logging.info('Penrose before hyperpod_nemo_adapter.collections.model.nlp on deepseek_pretrain.py')
from hyperpod_nemo_adapter.collections.model.nlp import (
    SageMakerDeepSeekDistilledLlamaModel,
    SageMakerDeepSeekDistilledQwenModel,
)
logging.info('Penrose before SageMakerTrainerBuilder on deepseek_pretrain.py')
from hyperpod_nemo_adapter.collections.parts import SageMakerTrainerBuilder
logging.info('Penrose before validate_config on deepseek_pretrain.py')
from hyperpod_nemo_adapter.utils.config_utils import validate_config
logging.info('Penrose before exp_manager on deepseek_pretrain.py')
from hyperpod_nemo_adapter.utils.exp_manager import exp_manager
logging.info('Penrose before setup_args_for_sm on deepseek_pretrain.py')
from hyperpod_nemo_adapter.utils.sm_utils import setup_args_for_sm

logging.info('Penrose after imports on deepseek_pretrain.py')


def train(cfg: DictConfig) -> None:
    logging.info("\n\n************** Experiment configuration ***********")
    logging.info(f"\n{OmegaConf.to_yaml(cfg)}")
    trainer, data_module = SageMakerTrainerBuilder(cfg).create_trainer()
    exp_manager(trainer, cfg.exp_manager)

    if "llama" in cfg.model.model_type:
        model_module = SageMakerDeepSeekDistilledLlamaModel(cfg.model, trainer, use_smp_model=cfg.use_smp_model)
    if "qwen" in cfg.model.model_type:
        model_module = SageMakerDeepSeekDistilledQwenModel(cfg.model, trainer, use_smp_model=cfg.use_smp_model)
    elif "deepseek_r1" in cfg.model.model_type or "deepseek_v3" in cfg.model.model_type:
        pass  # TODO add a model class for the first-party DeepSeek models (DeepSeek-R1, DeepSeek-V3...)

    trainer.fit(model_module, datamodule=data_module)


@hydra.main(config_path="conf", config_name="smp_deepseek_config", version_base="1.2")
@validate_config()
def main(cfg: DictConfig) -> None:
    train(cfg)


if __name__ == "__main__":
    setup_args_for_sm()
    main()
