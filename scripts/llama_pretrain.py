import hydra
from nemo.utils import logging
from omegaconf import DictConfig
from omegaconf.omegaconf import OmegaConf
from pytorch_lightning import Trainer

from sagemaker_nemo_adaptor.collections.data import DummyDataModule, GPTDataModule
from sagemaker_nemo_adaptor.collections.model.nlp import SageMakerLlamaModel
from sagemaker_nemo_adaptor.collections.parts import SageMakerFSDPStrategy


def train(cfg: DictConfig) -> None:
    logging.info("\n\n************** Experiment configuration ***********")
    logging.info(f"\n{OmegaConf.to_yaml(cfg)}")

    smp_config_dict = {
        "activation_loading_horizon": cfg.activation_loading_horizon,
        "sm_activation_offloading": cfg.offload_activations > 0,
    }
    if cfg.shard_degree is not None:
        smp_config_dict["hybrid_shard_degree"] = cfg.shard_degree
    smp_config_dict["tensor_parallel_degree"] = cfg.tensor_model_parallel_degree
    smp_config_dict["expert_parallel_degree"] = cfg.expert_model_parallel_degree
    smp_config_dict["random_seed"] = cfg.seed
    print(f"cfg checker: opt betas {cfg.optim.betas}")
    strategy = SageMakerFSDPStrategy(use_smp=cfg.use_smp, smp_config_dict=smp_config_dict, model_type=cfg.model_type)

    trainer = Trainer(
        strategy=strategy, max_steps=cfg.trainer.max_steps
    )  # TODO: could be configurable with cfg.trainer

    model = SageMakerLlamaModel(cfg, trainer)
    dm = DummyDataModule(cfg, trainer) if cfg.data.use_synthetic_data else GPTDataModule(cfg, trainer)
    trainer.fit(model, datamodule=dm)


@hydra.main(config_path="conf", config_name="smp_llama_config", version_base="1.2")
def main(cfg: DictConfig) -> None:
    train(cfg)


if __name__ == "__main__":
    main()
