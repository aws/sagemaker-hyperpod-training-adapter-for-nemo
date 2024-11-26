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

import pytest
from nemo.core.classes.modelPT import ModelPT
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from packaging import version as pversion
from pytorch_lightning import Trainer

import hyperpod_nemo_adapter.collections.model.sagemaker_base_model as sbm

# local modules
from hyperpod_nemo_adapter.collections.model.sagemaker_base_model import (
    SageMakerNLPBaseModel,
)
from tests.fixtures.adapter_config import full_config  # noqa F401
from tests.fixtures.loggers import sagemaker_logger  # noqa F401
from tests.utils import NestedDotMap

MODULE_PATH = "hyperpod_nemo_adapter.collections.model.sagemaker_base_model"

"""
UTILITY FUNCTIONS
"""


def build_base_model(model_cfg: DictConfig = {}, trainer=None, use_smp_model=True):
    if trainer is None:
        trainer = Trainer()

    return SageMakerNLPBaseModel(model_cfg, trainer, use_smp_model)


"""
TESTS
"""


def test_init(full_config):
    """__init__()"""
    base = build_base_model(full_config.model)
    assert isinstance(base, ModelPT)
    assert isinstance(base, SageMakerNLPBaseModel)


transformers_threshold_version = "4.37.1"
transformers_below_version = "4.37.0"


class TestBuildModel:
    """build_model()"""

    transformers_threshold_version = "4.37.1"
    transformers_below_version = "4.37.0"

    @pytest.mark.skip(reason="need refactor")
    @pytest.mark.parametrize(
        ("version", "exp_args_len", "exp_kwargs_len"),
        [
            [transformers_below_version, 1, 3],
            [transformers_threshold_version, 1, 4],
        ],
    )
    def test_w_hf_model_name_or_path(self, full_config, mocker, version, exp_args_len, exp_kwargs_len):
        from_pretrained_stub = mocker.stub()
        auto_model_mock, _ = self.get_test_mocks(mocker, version)
        auto_model_mock.from_pretrained = from_pretrained_stub

        # prepare
        full_config.model.do_finetune = True
        full_config.model.hf_model_name_or_path = "test/path"
        base = build_base_model(full_config.model)

        # test
        base.build_model(full_config.model)

        # assertions
        args, kwargs = from_pretrained_stub.call_args
        from_pretrained_stub.assert_called_once()
        assert len(args) == exp_args_len
        assert len(kwargs) == exp_kwargs_len
        assert args[0] == full_config.model.hf_model_name_or_path

    @pytest.mark.skip(reason="need refactor")
    def test_w_hf_model_name_or_path_and_no_flash_attention(self, full_config, mocker):
        from_pretrained_stub = mocker.stub()
        auto_model_mock, _ = self.get_test_mocks(mocker, self.transformers_threshold_version)
        auto_model_mock.from_pretrained = from_pretrained_stub

        # prepare
        full_config.model.do_finetune = True
        full_config.model.hf_model_name_or_path = "test/path"
        full_config.model.use_flash_attention = False
        base = build_base_model(full_config.model)

        # test
        base.build_model(full_config.model)

        # assertions
        args, kwargs = from_pretrained_stub.call_args
        from_pretrained_stub.assert_called_once()
        assert len(args) == 1
        assert len(kwargs) == 3
        assert "attn_implementation" not in kwargs

    @pytest.mark.parametrize(
        ("version", "exp_args_len", "exp_kwargs_len"),
        [
            [transformers_below_version, 1, 0],
            [transformers_threshold_version, 1, 1],
        ],
    )
    def test_no_hf_model_name_or_path(self, full_config, mocker, version, exp_args_len, exp_kwargs_len):
        from_config_stub = mocker.stub()
        auto_model_mock, _ = self.get_test_mocks(mocker, version)
        auto_model_mock.from_config = from_config_stub
        test_model_config = {}

        # prepare
        full_config.model.hf_model_name_or_path = None
        base = build_base_model(full_config.model)

        # test
        base.build_model(test_model_config)

        # assertions
        args, kwargs = from_config_stub.call_args
        from_config_stub.assert_called_once()
        assert len(args) == exp_args_len
        assert len(kwargs) == exp_kwargs_len
        assert args[0] == test_model_config

    def test_no_hf_model_name_or_path_and_no_flash_attention(self, full_config, mocker):
        from_config_stub = mocker.stub()
        auto_model_mock, _ = self.get_test_mocks(mocker, self.transformers_threshold_version)
        auto_model_mock.from_config = from_config_stub
        test_model_args = {}

        # prepare
        full_config.model.hf_model_name_or_path = None
        full_config.model.use_flash_attention = False
        base = build_base_model(full_config.model)

        # test
        base.build_model(test_model_args)

        # assertions
        args, kwargs = from_config_stub.call_args
        from_config_stub.assert_called_once()
        assert len(args) == 1
        assert args[0] == test_model_args
        assert len(kwargs) == 0

    def get_test_mocks(self, mocker, mocked_version: str):
        auto_model_mock = mocker.patch(MODULE_PATH + ".AutoModelForCausalLM")
        parsed_version = pversion.parse(mocked_version)
        TF_VERSION_mock = mocker.patch.object(sbm, "TF_VERSION", new=parsed_version)
        return auto_model_mock, TF_VERSION_mock


class TestTrainingStep:
    """training_step()"""

    class TrainerMock:
        def __init__(self):
            self.datamodule = NestedDotMap({"get_batch": None})

    @pytest.mark.skip(reason="need refactor")
    def test_fp8_and_use_smp_model(self, full_config, mocker):
        fp8_autocast_mock = mocker.patch(MODULE_PATH + ".transformer_engine.pytorch.fp8_autocast")
        test_loss = 22

        # prepare
        full_config.model.fp8 = True
        base = build_base_model(full_config.model, Trainer(), True)
        base.trainer = TestTrainingStep.TrainerMock()
        base.trainer.datamodule.get_batch = mocker.Mock(return_value=[[], None, []])
        base.model = model_mock = mocker.Mock(return_value={"loss": test_loss})
        base.fp8_recipe = "test_fp8_recipe"

        # test
        base.training_step(None, None)

        # assertions
        base.trainer.datamodule.get_batch.assert_called_once()
        fp8_autocast_mock.assert_called_once()
        model_mock.assert_called_once()
        assert base.loss == test_loss

    @pytest.mark.skip(reason="need refactor")
    @pytest.mark.parametrize(
        ("fp8", "use_smp_model"),
        [
            [True, False],
            [False, True],
            [False, False],
        ],
    )
    def test_no_fp8_and_use_smp_model(self, full_config, mocker, fp8, use_smp_model):
        fp8_autocast_mock = mocker.patch(MODULE_PATH + ".transformer_engine.pytorch.fp8_autocast")
        test_loss = 22

        # prepare
        full_config.model.fp8 = fp8
        base = build_base_model(full_config.model, Trainer(), use_smp_model)
        base.trainer = TestTrainingStep.TrainerMock()
        base.trainer.datamodule.get_batch = mocker.Mock(return_value=[[], None, []])
        base.model = model_mock = mocker.Mock(return_value={"loss": test_loss})

        # test
        base.training_step(None, None)

        # assertions
        base.trainer.datamodule.get_batch.assert_called_once()
        fp8_autocast_mock.assert_not_called()
        model_mock.assert_called_once()
        assert base.loss == test_loss


class TestSetupOptimization:
    """setup_optimization()"""

    max_steps = None
    setup_optimization_spy = None
    _get_max_steps_spy = None

    @pytest.fixture(autouse=True)
    def around_each_test(self, mocker):
        T = TestSetupOptimization

        # set mocks only once
        if T.setup_optimization_spy is None:
            T.max_steps = 5
            T.setup_optimization_spy = mocker.patch("nemo.core.classes.modelPT.ModelPT.setup_optimization")
            T._get_max_steps_spy = mocker.patch(
                MODULE_PATH + ".SageMakerNLPBaseModel._get_max_steps",
                return_value=T.max_steps,
            )

        mocker.resetall()
        yield

    def test_no_args(self, full_config):
        # from config file, max_steps should not be defined
        assert full_config.model.optim.sched.get("max_steps") is None

        optimizer_with_max_steps = self.get_optim_cfg_with_max_steps(full_config)
        base = build_base_model(full_config.model)
        base.setup_optimization()

        self.shared_assertions(optimizer_with_max_steps)

    def test_with_args(self, full_config):
        # from config file, max_steps should not be defined
        assert full_config.model.optim.sched.get("max_steps") is None

        optimizer_with_max_steps = self.get_optim_cfg_with_max_steps(full_config)
        base = build_base_model(full_config.model)

        base.setup_optimization(optim_config=full_config.model.optim, optim_kwargs={})

        self.shared_assertions(optimizer_with_max_steps)

    def get_optim_cfg_with_max_steps(self, full_config):
        T = TestSetupOptimization
        optimizer_with_max_steps = OmegaConf.to_container(full_config.model.optim)
        optimizer_with_max_steps["sched"]["max_steps"] = T.max_steps

        return optimizer_with_max_steps

    def shared_assertions(self, optimizer_with_max_steps):
        T = TestSetupOptimization

        T._get_max_steps_spy.assert_called_once()
        kwargs = T.setup_optimization_spy.call_args[1]
        assert len(kwargs) == 2
        assert isinstance(kwargs["optim_config"], DictConfig)
        assert OmegaConf.to_container(kwargs["optim_config"]) == optimizer_with_max_steps
        assert kwargs["optim_kwargs"] == {}

    @pytest.mark.skip(reason="need refactor")
    def test_on_train_batch_end(self, full_config, mocker):
        """on_train_batch_end()"""

        test_process_log = "test_process_log"
        base = build_base_model(full_config.model)
        _process_loss_mock = mocker.patch.object(base, "_process_loss", return_value=test_process_log)
        log_mock = mocker.patch.object(base, "log")

        # test
        base.on_train_batch_end()

        # assertions
        _process_loss_mock.assert_called_once()
        log_mock.assert_called_once_with("loss", test_process_log, prog_bar=True)


class TestProcessLoss:
    """_process_loss()"""

    def test_w_log_reduced_training_loss(self, full_config, mocker):
        test_loss_detach_item = 10
        test_world_size = 2
        dist_all_reduce_mock = mocker.patch(MODULE_PATH + ".dist")

        # prepare
        dist_all_reduce_mock.all_reduce = mocker.Mock(return_value=test_loss_detach_item)
        dist_all_reduce_mock.get_world_size = mocker.Mock(return_value=test_world_size)
        full_config.model.log_reduced_training_loss = True
        base = build_base_model(full_config.model)
        base.loss = NestedDotMap(
            {
                "detach": mocker.Mock(
                    return_value=NestedDotMap({"item": mocker.Mock(return_value=test_loss_detach_item)})
                )
            }
        )

        # test
        res = base._process_loss(base.loss)

        # assertions
        assert res == test_loss_detach_item / test_world_size  # 10 / 2 == 5
        dist_all_reduce_mock.all_reduce.assert_called_once()
        dist_all_reduce_mock.get_world_size.assert_called_once()

    @pytest.mark.skip(reason="need refactor")
    def test_no_log_reduced_training_loss(self, full_config, mocker):
        test_loss_item = 10

        # prepare
        full_config.model.log_reduced_training_loss = False
        base = build_base_model(full_config.model)
        base.loss = NestedDotMap({"item": mocker.Mock(return_value=test_loss_item)})

        # test
        res = base._process_loss()

        # assertions
        assert res == test_loss_item


class Test_GetMaxSteps:
    """_get_max_steps()"""

    def test_lr_decay_iters_is_defined(self, full_config):
        assert full_config.model.lr_decay_iters is not None

        base = build_base_model(full_config.model)
        res = base._get_max_steps()
        assert res == full_config.model.lr_decay_iters

    def test_trainer_is_missing(self, full_config, mocker, sagemaker_logger):
        warning_spy = mocker.spy(sagemaker_logger, "warning")

        # prepare
        full_config.model.lr_decay_iters = None
        base = build_base_model(full_config.model)
        base._trainer = None

        # test
        res = base._get_max_steps()

        # assertions
        assert res == -1
        assert warning_spy.call_count == 1
        assert "no trainer is set" in warning_spy.call_args[0][0]

    def test_max_steps_and_max_epochs_gt_zero(self, mocker, full_config, sagemaker_logger):
        warning_spy = mocker.spy(sagemaker_logger, "warning")

        # prepare
        full_config.model.lr_decay_iters = None
        base = build_base_model(full_config.model)
        base._trainer = Trainer(max_steps=11, max_epochs=22)

        # test
        res = base._get_max_steps()

        # assertions
        assert res == 11
        assert warning_spy.call_count == 1
        assert "is already set" in warning_spy.call_args[0][0]

    @pytest.mark.parametrize(
        "test_trainer",
        [
            Trainer(max_steps=-1, max_epochs=None),
            Trainer(max_steps=-1, max_epochs=-1),
        ],
    )
    def test_max_stpes_gt_zero(self, mocker, full_config, sagemaker_logger, test_trainer):
        warning_spy = mocker.spy(sagemaker_logger, "warning")

        # prepare
        full_config.model.lr_decay_iters = None
        base = build_base_model(full_config.model)
        base._trainer = test_trainer

        # test
        res = base._get_max_steps()

        # assertions
        assert res == -1
        assert warning_spy.call_count == 1
        assert "neither" in warning_spy.call_args[0][0]

    def test_max_epochs_is_none(self, mocker, full_config, sagemaker_logger):
        warning_spy = mocker.spy(sagemaker_logger, "warning")

        # prepare
        full_config.model.lr_decay_iters = None
        base = build_base_model(full_config.model)
        base._trainer = Trainer(max_steps=-1, max_epochs=22)
        base._train_dl = None

        # test
        res = base._get_max_steps()

        # assertions
        assert res == -1
        assert warning_spy.call_count == 1
        assert "train dataloader" in warning_spy.call_args[0][0]

    def test_limit_train_batches_is_defined(self, full_config):
        # prepare
        full_config.model.lr_decay_iters = None
        base = build_base_model(full_config.model)
        base._trainer = trainer = Trainer(max_steps=-1, max_epochs=2, limit_train_batches=0.8)
        dm = OmegaConf.create(
            {
                "_train_dl": [1, 2, 3, 4],
            }
        )
        base._train_dl = dm._train_dl
        base.datamodule = dm

        # test
        res = base._get_max_steps()

        # assertions
        expected = 6
        num_global_batches = len(dm._train_dl)  # len = 4
        # 4 * .8 == 3.2, rounded down to 3
        limit_batches = int(trainer.limit_train_batches * num_global_batches)
        steps_per_epoch = min(num_global_batches, limit_batches)  # min(4, 3)
        assert res == steps_per_epoch * trainer.max_epochs
        assert res == expected

    def test_limit_train_batches_is_none(self, full_config):
        # prepare
        full_config.model.lr_decay_iters = None
        base = build_base_model(full_config.model)
        base._trainer = trainer = Trainer(max_steps=-1, max_epochs=2, limit_train_batches=None)
        dm = OmegaConf.create(
            {
                "_train_dl": [1, 2, 3, 4],
            }
        )
        base._train_dl = dm._train_dl
        base.datamodule = dm

        # test
        res = base._get_max_steps()

        # assertions
        expected = 8
        num_global_batches = len(dm._train_dl)  # len = 4
        assert res == num_global_batches * trainer.max_epochs
        assert res == expected
