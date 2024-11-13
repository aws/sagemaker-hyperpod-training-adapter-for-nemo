import hydra
import torch
import torch._dynamo
from nemo.utils import logging
from omegaconf import DictConfig
from omegaconf.omegaconf import OmegaConf

torch._dynamo.config.suppress_errors = True


from sagemaker_nemo_adaptor.collections.model.nlp import SageMakerMixtralModel
from sagemaker_nemo_adaptor.collections.parts import SageMakerTrainerBuilder
from sagemaker_nemo_adaptor.utils.config_utils import validate_config
from sagemaker_nemo_adaptor.utils.exp_manager import exp_manager
from sagemaker_nemo_adaptor.utils.sm_utils import setup_args_for_sm


def train(cfg: DictConfig) -> None:
    logging.info("\n\n************** Experiment configuration ***********")
    logging.info(f"\n{OmegaConf.to_yaml(cfg)}")
    trainer, data_module = SageMakerTrainerBuilder(cfg).create_trainer()
    exp_manager(trainer, cfg.exp_manager)
    model_module = SageMakerMixtralModel(cfg.model, trainer, use_smp_model=cfg.use_smp_model)
    trainer.fit(model_module, datamodule=data_module)


@hydra.main(config_path="conf", config_name="smp_mixtral_config", version_base="1.2")
@validate_config()
def main(cfg: DictConfig) -> None:
    train(cfg)


if __name__ == "__main__":
    setup_args_for_sm()
    main()
