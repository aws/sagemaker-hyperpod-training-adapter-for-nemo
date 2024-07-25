import pytest
from nemo.collections.nlp.models.nlp_model import NLPModel
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig
from pytorch_lightning import Trainer

# local modules
from sagemaker_nemo_adaptor.collections.model.sagemaker_base_model import (
    SageMakerNLPBaseModel,
)
from tests.fixtures.adaptor_config import testing_config  # noqa F401
from tests.fixtures.loggers import sagemaker_logger  # noqa F401

"""
UTILITY FUNCTIONS
"""


def build_base_model(cfg: DictConfig = {}, trainer=None):
    if trainer is None:
        trainer = Trainer()

    return SageMakerNLPBaseModel(cfg, trainer)


"""
TESTS
"""


def test_init(testing_config):
    base = build_base_model(testing_config)
    assert isinstance(base, NLPModel)
    assert isinstance(base, SageMakerNLPBaseModel)


class TestSetupOptimization:
    max_steps = None
    setup_optimization_spy = None
    _get_max_steps_spy = None

    @pytest.fixture(autouse=True)
    def around_each_test(self, mocker):
        T = TestSetupOptimization

        # set mocks only once
        if T.setup_optimization_spy is None:
            T.max_steps = 5
            T.setup_optimization_spy = mocker.patch("nemo.collections.nlp.models.nlp_model.NLPModel.setup_optimization")
            T._get_max_steps_spy = mocker.patch(
                "sagemaker_nemo_adaptor.collections.model.sagemaker_base_model.SageMakerNLPBaseModel._get_max_steps",
                return_value=T.max_steps,
            )

        mocker.resetall()
        yield

    def test_no_args(self, testing_config):
        # from config file, max_steps should not be defined
        assert testing_config.optim.sched.get("max_steps") is None

        optimizer_with_max_steps = self.get_optim_cfg_with_max_steps(testing_config)
        base = build_base_model(testing_config)
        base.setup_optimization()

        self.shared_assertions(optimizer_with_max_steps)

    def test_with_args(self, testing_config):
        # from config file, max_steps should not be defined
        assert testing_config.optim.sched.get("max_steps") is None

        optimizer_with_max_steps = self.get_optim_cfg_with_max_steps(testing_config)
        base = build_base_model(testing_config)

        base.setup_optimization(optim_config=testing_config.optim, optim_kwargs={})

        self.shared_assertions(optimizer_with_max_steps)

    def get_optim_cfg_with_max_steps(self, testing_config):
        T = TestSetupOptimization
        optimizer_with_max_steps = OmegaConf.to_container(testing_config.optim)
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


class Test_GetMaxSteps:
    def test_lr_decay_iters_is_defined(self, testing_config):
        assert testing_config.lr_decay_iters is not None

        base = build_base_model(testing_config)
        res = base._get_max_steps()
        assert res == testing_config.lr_decay_iters

    def test_trainer_is_missing(self, testing_config, mocker, sagemaker_logger):
        warning_spy = mocker.spy(sagemaker_logger, "warning")

        # prepare
        testing_config.lr_decay_iters = None
        base = build_base_model(testing_config)
        base._trainer = None

        # test
        res = base._get_max_steps()

        # assertions
        assert res == -1
        assert warning_spy.call_count == 1
        assert "no trainer is set" in warning_spy.call_args[0][0]

    def test_max_steps_and_max_epochs_gt_zero(self, mocker, testing_config, sagemaker_logger):
        warning_spy = mocker.spy(sagemaker_logger, "warning")

        # prepare
        testing_config.lr_decay_iters = None
        base = build_base_model(testing_config)
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
    def test_max_stpes_gt_zero(self, mocker, testing_config, sagemaker_logger, test_trainer):
        warning_spy = mocker.spy(sagemaker_logger, "warning")

        # prepare
        testing_config.lr_decay_iters = None
        base = build_base_model(testing_config)
        base._trainer = test_trainer

        # test
        res = base._get_max_steps()

        # assertions
        assert res == -1
        assert warning_spy.call_count == 1
        assert "neither" in warning_spy.call_args[0][0]

    def test_max_epochs_is_none(self, mocker, testing_config, sagemaker_logger):
        warning_spy = mocker.spy(sagemaker_logger, "warning")

        # prepare
        testing_config.lr_decay_iters = None
        base = build_base_model(testing_config)
        base._trainer = Trainer(max_steps=-1, max_epochs=22)
        base._train_dl = None

        # test
        res = base._get_max_steps()

        # assertions
        assert res == -1
        assert warning_spy.call_count == 1
        assert "train dataloader" in warning_spy.call_args[0][0]

    def test_limit_train_batches_is_defined(self, testing_config):
        # prepare
        testing_config.lr_decay_iters = None
        base = build_base_model(testing_config)
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

    def test_limit_train_batches_is_none(self, testing_config):
        # prepare
        testing_config.lr_decay_iters = None
        base = build_base_model(testing_config)
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
