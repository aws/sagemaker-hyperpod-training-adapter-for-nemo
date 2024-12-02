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

import hydra
from nemo.utils import logging
from omegaconf import DictConfig
from omegaconf.omegaconf import OmegaConf

from hyperpod_nemo_adapter.collections.model.nlp import SageMakerLlamaModel
from hyperpod_nemo_adapter.collections.parts import SageMakerTrainerBuilder
from hyperpod_nemo_adapter.utils.config_utils import validate_config
from hyperpod_nemo_adapter.utils.exp_manager import exp_manager
from hyperpod_nemo_adapter.utils.sm_utils import setup_args_for_sm


def train(cfg: DictConfig) -> None:
    logging.info("\n\n************** Experiment configuration ***********")
    logging.info(f"\n{OmegaConf.to_yaml(cfg)}")
    trainer, data_module = SageMakerTrainerBuilder(cfg).create_trainer()
    exp_manager(trainer, cfg.exp_manager)
    model_module = SageMakerLlamaModel(cfg.model, trainer, use_smp_model=cfg.use_smp_model)
    trainer.fit(model_module, datamodule=data_module)


@hydra.main(config_path="conf", config_name="smp_llama_config", version_base="1.2")
@validate_config()
def main(cfg: DictConfig) -> None:
    train(cfg)


if __name__ == "__main__":
    setup_args_for_sm()
    main()
