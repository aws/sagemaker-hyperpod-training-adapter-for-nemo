import pytest
from pydantic import ValidationError

from sagemaker_nemo_adaptor.conf.config_schemas import (
    BaseCheckpointCallbackConfig,
    BaseConfig,
    BaseExpManager,
    BaseModelConfig,
    BaseModelDataConfig,
    BaseModelOptimizerConfig,
    BaseModelOptimizerScheduler,
    BaseRunConfig,
    BaseTrainerConfig,
    ConfigAllow,
    ConfigForbid,
    ConfigWithSMPForbid,
    ModelConfigWithSMP,
    SageMakerParallelConfig,
)
from sagemaker_nemo_adaptor.constants import ModelType
from tests.fixtures.loggers import sagemaker_logger  # noqa F401


def create_BaseModelConfig():
    return BaseModelConfig(do_finetune=False)


class Test_SageMakerParallelConfig:
    def test_happy_path(self):
        config = self.build_config()

        try:
            validated = SageMakerParallelConfig.model_validate(config)
            assert validated is not None
        except Exception as e:
            pytest.fail(f"Unexpectedly failed to validate config: {e}")

    def test_no_args(self):
        try:
            validated = SageMakerParallelConfig()
            assert validated.tensor_model_parallel_degree == 1
        except Exception as e:
            pytest.fail(f"Unexpectedly failed to validate config: {e}")

    def test_tp_outside_of_range(self):
        config = self.build_config(tensor_model_parallel_degree=0)

        with pytest.raises(ValidationError) as e:
            SageMakerParallelConfig.model_validate(config)

    def test_tp_not_power_of_two(self):
        config = self.build_config(tensor_model_parallel_degree=3)

        with pytest.raises(ValidationError) as e:
            SageMakerParallelConfig.model_validate(config)

    def test_default_value(self):
        config = self.build_config()
        del config["tensor_model_parallel_degree"]

        try:
            validated = SageMakerParallelConfig.model_validate(config)
            assert validated.tensor_model_parallel_degree == 1
        except Exception as e:
            pytest.fail(f"Unexpectedly failed to validate config: {e}")

    def build_config(self, **kwargs) -> dict:
        return {
            "tensor_model_parallel_degree": 1,
            "expert_model_parallel_degree": 1,
            **kwargs,
        }


class Test_BaseModelOptimizerScheduler:
    def test_happy_path(self):
        config = self.build_config()

        try:
            validated = BaseModelOptimizerScheduler.model_validate(config)
            assert validated is not None
        except Exception as e:
            pytest.fail(f"Unexpectedly failed to validate config: {e}")

    def test_no_args(self):
        try:
            validated = BaseModelOptimizerScheduler()
            assert validated.name == "CosineAnnealing"
        except Exception as e:
            pytest.fail(f"Unexpectedly failed to validate config: {e}")

    def test_outside_of_range(self):
        invalid_val = -1
        config = self.build_config(warmup_steps=invalid_val)

        with pytest.raises(ValidationError) as e:
            BaseModelOptimizerScheduler.model_validate(config)

    def test_default_value(self):
        config = self.build_config()
        del config["warmup_steps"]

        try:
            validated = BaseModelOptimizerScheduler.model_validate(config)
            assert validated.warmup_steps == 500
        except Exception as e:
            pytest.fail(f"Unexpectedly failed to validate config: {e}")

    def build_config(self, **kwargs) -> dict:
        return {
            "name": "CosineAnnealing",
            "warmup_steps": 500,
            "constant_steps": 0,
            "min_lr": 2e-5,
            **kwargs,
        }


class Test_BaseModelOptimizerConfig:
    def test_happy_path(self):
        config = self.build_config()

        try:
            validated = BaseModelOptimizerConfig.model_validate(config)
            assert validated is not None
        except Exception as e:
            pytest.fail(f"Unexpectedly failed to validate config: {e}")

    def test_no_args(self):
        try:
            validated = BaseModelOptimizerConfig()
            assert validated is not None
            assert validated.betas == [0.9, 0.98]
            assert validated.sched.name == "CosineAnnealing"
        except Exception as e:
            pytest.fail(f"Unexpectedly failed to validate config: {e}")

    def test_outside_of_range(self):
        invalid_val = -1
        config = self.build_config(lr=invalid_val)

        with pytest.raises(ValidationError) as e:
            BaseModelOptimizerConfig.model_validate(config)

    def test_default_value(self):
        config = self.build_config()
        del config["name"]

        try:
            validated = BaseModelOptimizerConfig.model_validate(config)
            assert validated.name == "adamw"
        except Exception as e:
            pytest.fail(f"Unexpectedly failed to validate config: {e}")

    def build_config(self, **kwargs) -> dict:
        return {
            "name": "adamw",
            "lr": 2e-4,
            "weight_decay": 0.01,
            "betas": [0.9, 0.98],
            "sched": {
                "name": "CosineAnnealing",
                "warmup_steps": 500,
                "constant_steps": 0,
                "min_lr": 2e-5,
            },
            **kwargs,
        }


class Test_BaseModelDataConfig:
    def test_happy_path(self):
        config = self.build_config()

        try:
            validated = BaseModelDataConfig.model_validate(config)
            assert validated is not None
        except Exception as e:
            pytest.fail(f"Unexpectedly failed to validate config: {e}")

    def test_no_args(self):
        with pytest.raises(ValueError) as err_info:
            BaseModelDataConfig()

        assert "train_dir" in str(err_info.value)

    def test_outside_of_range(self):
        invalid_val = "invalid_value"
        config = self.build_config(dataset_type=invalid_val)

        with pytest.raises(ValidationError) as e:
            BaseModelDataConfig.model_validate(config)

    def test_default_value(self):
        config = self.build_config()
        del config["dataset_type"]

        try:
            validated = BaseModelDataConfig.model_validate(config)
            assert validated.dataset_type == "hf"
        except Exception as e:
            pytest.fail(f"Unexpectedly failed to validate config: {e}")

    def test_before_model_validation(self):
        config = self.build_config(use_synthetic_data=False)
        del config["val_dir"]

        with pytest.raises(ValueError) as e:
            BaseModelDataConfig.model_validate(config)

    def build_config(self, **kwargs) -> dict:
        return {
            "train_dir": ["/fsx/datasets/train_ids_wsvocab_redo_2048_smaller"],
            "val_dir": ["/fsx/datasets/llama_new/val"],
            "dataset_type": "hf",
            "use_synthetic_data": False,
            "zipped_data": False,
            **kwargs,
        }


class Test_BaseModelConfig:
    def test_happy_path(self):
        config = self.build_config()

        try:
            validated = BaseModelConfig.model_validate(config)
            assert validated is not None
        except Exception as e:
            pytest.fail(f"Unexpectedly failed to validate config: {e}")

    def test_no_args(self):
        try:
            validated = create_BaseModelConfig()
            assert validated.model_type == ModelType.LLAMA_V3.value
            assert type(validated.optim) is BaseModelOptimizerConfig
            assert type(validated.data) is BaseModelDataConfig
        except Exception as e:
            pytest.fail(f"Unexpectedly failed to validate config: {e}")

    def test_outside_of_range(self):
        invalid_val = -1
        config = self.build_config(hidden_width=invalid_val)

        with pytest.raises(ValidationError):
            BaseModelConfig.model_validate(config)

    def test_default_value(self):
        config = self.build_config()
        del config["model_type"]

        try:
            validated = BaseModelConfig.model_validate(config)
            assert validated.model_type == ModelType.LLAMA_V3.value
        except Exception as e:
            pytest.fail(f"Unexpectedly failed to validate config: {e}")

    def test_before_model_validation(self):
        config = self.build_config(max_context_width=10)
        del config["max_position_embeddings"]

        try:
            validated = BaseModelConfig.model_validate(config)
            assert validated.max_position_embeddings == 10
        except Exception as e:
            pytest.fail(f"Unexpectedly failed to validate config: {e}")

    def test_after_model_validation(self, mocker, sagemaker_logger):
        config = self.build_config(max_context_width=3)
        warning_spy = mocker.spy(sagemaker_logger, "warning")

        try:
            validated = BaseModelConfig.model_validate(config)
            assert validated.max_context_width == 3
            assert warning_spy.call_count == 1
            assert "power of 2" in warning_spy.call_args[0][0]
        except Exception as e:
            pytest.fail(f"Unexpectedly failed to validate config: {e}")

    def build_config(self, **kwargs) -> dict:
        return {
            "do_finetune": False,
            "model_type": ModelType.LLAMA_V3.value,
            "hidden_width": 4096,
            "max_content_width": 4096,
            "max_position_embeddings": 2048,
            "optim": BaseModelOptimizerConfig().model_dump(),
            **kwargs,
        }


class Test_BaseTrainerConfig:
    def test_happy_path(self):
        config = self.build_config()

        try:
            validated = BaseTrainerConfig.model_validate(config)
            assert validated is not None
        except Exception as e:
            pytest.fail(f"Unexpectedly failed to validate config: {e}")

    def test_no_args(self):
        try:
            validated = BaseTrainerConfig()
            assert validated.max_steps == 50
        except Exception as e:
            pytest.fail(f"Unexpectedly failed to validate config: {e}")

    def test_default_value(self):
        config = self.build_config()
        del config["max_steps"]

        try:
            validated = BaseTrainerConfig.model_validate(config)
            assert validated.max_steps == 50
        except Exception as e:
            pytest.fail(f"Unexpectedly failed to validate config: {e}")

    def test_outside_of_range(self):
        invalid_val = -1
        config = self.build_config(max_steps=invalid_val)

        with pytest.raises(ValidationError):
            BaseTrainerConfig.model_validate(config)

    def build_config(self, **kwargs) -> dict:
        return {
            "max_steps": 10000,
            "precision": "bf16",
            **kwargs,
        }


class Test_BaseCheckpointCallbackConfig:
    def test_happy_path(self):
        config = self.build_config()

        try:
            validated = BaseCheckpointCallbackConfig.model_validate(config)
            assert validated is not None
        except Exception as e:
            pytest.fail(f"Unexpectedly failed to validate config: {e}")

    def test_no_args(self):
        try:
            validated = BaseCheckpointCallbackConfig()
            assert validated.save_top_k == 10
        except Exception as e:
            pytest.fail(f"Unexpectedly failed to validate config: {e}")

    def test_default_value(self):
        config = self.build_config()
        del config["save_top_k"]

        try:
            validated = BaseCheckpointCallbackConfig.model_validate(config)
            assert validated.save_top_k == 10
        except Exception as e:
            pytest.fail(f"Unexpectedly failed to validate config: {e}")

    def test_outside_of_range(self):
        invalid_val = -1
        config = self.build_config(save_top_k=invalid_val)

        with pytest.raises(ValidationError) as e:
            BaseCheckpointCallbackConfig.model_validate(config)

    def build_config(self, **kwargs) -> dict:
        return {
            "save_top_k": 10,
            **kwargs,
        }


class Test_BaseExpManager:
    def test_happy_path(self):
        config = self.build_config()

        try:
            validated = BaseExpManager.model_validate(config)
            assert validated is not None
        except Exception as e:
            pytest.fail(f"Unexpectedly failed to validate config: {e}")

    def test_no_args(self):
        try:
            validated = BaseExpManager()
            assert validated.exp_dir == "/fsx/exp/"
        except Exception as e:
            pytest.fail(f"Unexpectedly failed to validate config: {e}")

    def test_default_value(self):
        config = self.build_config()

        try:
            validated = BaseExpManager.model_validate(config)
            assert validated.name == "my_experiment"
            assert type(validated.checkpoint_callback_params) is BaseCheckpointCallbackConfig
        except Exception as e:
            pytest.fail(f"Unexpectedly failed to validate config: {e}")

    def test_outside_of_range(self):
        invalid_val = 123
        config = self.build_config(name=invalid_val)

        with pytest.raises(ValidationError):
            BaseExpManager.model_validate(config)

    def build_config(self, **kwargs) -> dict:
        return {
            "exp_dir": "/some/dir",
            **kwargs,
        }


class Test_BaseRunConfig:
    def test_happy_path(self):
        config = self.build_config()

        try:
            validated = BaseRunConfig.model_validate(config)
            assert validated is not None
        except Exception as e:
            pytest.fail(f"Unexpectedly failed to validate config: {e}")

    def test_no_args(self):
        try:
            validated = BaseRunConfig()
            assert validated.name == "llama-8b"
        except Exception as e:
            pytest.fail(f"Unexpectedly failed to validate config: {e}")

    def test_default_value(self):
        config = self.build_config()
        del config["name"]

        try:
            validated = BaseRunConfig.model_validate(config)
            assert validated.name == "llama-8b"
        except Exception as e:
            pytest.fail(f"Unexpectedly failed to validate config: {e}")

    def test_outside_of_range(self):
        invalid_val = 123
        config = self.build_config(name=invalid_val)

        with pytest.raises(ValidationError):
            BaseRunConfig.model_validate(config)

    def build_config(self, **kwargs) -> dict:
        return {
            "name": "test_name",
            "results_dir": "/some/dir",
            "time_limit": "6-00:00:00",
            **kwargs,
        }


class Test_BaseConfig:
    def test_happy_path(self):
        config = self.build_config()

        try:
            validated = BaseConfig.model_validate(config)
            assert validated is not None
        except Exception as e:
            pytest.fail(f"Unexpectedly failed to validate config: {e}")

    def test_no_args(self):
        with pytest.raises(ValidationError):
            BaseConfig()

    def test_default_value(self):
        config = self.build_config(name=["test_name"])
        del config["name"]

        try:
            validated = BaseConfig.model_validate(config)
            assert validated.name == ["hf_llama_8b"]
        except Exception as e:
            pytest.fail(f"Unexpectedly failed to validate config: {e}")

    def test_requires_model(self):
        config = self.build_config()
        del config["model"]

        with pytest.raises(ValidationError) as err_info:
            BaseConfig.model_validate(config)

        assert "'model' is required" in str(err_info.value)

    def test_outside_of_range(self):
        invalid_val = "invalide_option"
        config = self.build_config(distributed_backend=invalid_val)

        with pytest.raises(ValidationError) as e:
            BaseConfig.model_validate(config)

    def build_config(self, **kwargs) -> dict:
        return {
            "distributed_backend": "nccl",
            "trainer": BaseTrainerConfig().model_dump(),
            "model": create_BaseModelConfig().model_dump(),
            **kwargs,
        }


class Test_ModelConfigWithSMP:
    def test_inheritance(self):
        config = self.build_config()

        try:
            validated = ModelConfigWithSMP.model_validate(config)
            assert isinstance(validated, BaseModelConfig)
            assert isinstance(validated, SageMakerParallelConfig)
        except Exception as e:
            pytest.fail(f"Unexpectedly failed to validate config: {e}")

    def build_config(self, **kwargs) -> dict:
        return {
            "do_finetune": False,
            **kwargs,
        }


class Test_ConfigForbid:
    def test_inheritance(self):
        config = self.build_config()

        try:
            validated = ConfigForbid.model_validate(config)
            assert isinstance(validated, BaseConfig)
            assert isinstance(validated.model, BaseModelConfig)
        except Exception as e:
            pytest.fail(f"Unexpectedly failed to validate config: {e}")

    def build_config(self, **kwargs) -> dict:
        return {
            "distributed_backend": "nccl",
            "trainer": BaseTrainerConfig().model_dump(),
            "model": create_BaseModelConfig().model_dump(),
            **kwargs,
        }


class Test_ConfigAllow:
    def test_inheritance(self):
        config = self.build_config()
        config["model"]["custom_test"] = "test"

        try:
            validated = ConfigAllow.model_validate(config)
            assert isinstance(validated, BaseConfig)
            assert isinstance(validated.model, BaseModelConfig)
            assert validated.model.custom_test == "test"
        except Exception as e:
            pytest.fail(f"Unexpectedly failed to validate config: {e}")

    def build_config(self, **kwargs) -> dict:
        return {
            "distributed_backend": "nccl",
            "trainer": BaseTrainerConfig().model_dump(),
            "model": BaseModelConfig(do_finetune=False).model_dump(),
            **kwargs,
        }


class Test_ConfigWithSMP:
    def test_inheritance(self):
        config = self.build_config()

        try:
            validated = ConfigWithSMPForbid.model_validate(config)
            assert isinstance(validated, BaseConfig)
            assert isinstance(validated.model, ModelConfigWithSMP)
        except Exception as e:
            pytest.fail(f"Unexpectedly failed to validate config: {e}")

    def build_config(self, **kwargs) -> dict:
        return {
            "distributed_backend": "nccl",
            "trainer": BaseTrainerConfig().model_dump(),
            "model": ModelConfigWithSMP(do_finetune=False).model_dump(),
            **kwargs,
        }
