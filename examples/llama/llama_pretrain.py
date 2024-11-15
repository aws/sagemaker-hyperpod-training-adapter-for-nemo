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
