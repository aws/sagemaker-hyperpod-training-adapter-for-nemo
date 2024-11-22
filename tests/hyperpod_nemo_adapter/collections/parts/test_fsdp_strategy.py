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

from nemo.collections.nlp.parts.nlp_overrides import NLPFSDPStrategy
from pytorch_lightning import Trainer

from hyperpod_nemo_adapter.collections.parts.fsdp_strategy import SageMakerFSDPStrategy
from hyperpod_nemo_adapter.utils.app_state import SageMakerAppState
from tests.fixtures.adapter_config import full_config  # noqa F401

MODULE_PATH = "hyperpod_nemo_adapter.collections.parts.fsdp_strategy"

"""
TESTS
"""


def test_init(full_config):
    full_config.use_smp_model = True
    strategy = SageMakerFSDPStrategy(full_config)
    assert isinstance(strategy, NLPFSDPStrategy)
    assert isinstance(strategy, SageMakerFSDPStrategy)
    assert strategy.cfg == full_config
    assert strategy.use_smp_model == True


class Test_set_mixed_precision_recipe:
    """Pending: Requires further implementation"""


class Test_setup_model:
    """Pending: Requires further implementation"""


def test_setup(mocker, full_config):
    full_config.use_smp_model = True
    strategy = SageMakerFSDPStrategy(full_config)
    parent_cls = strategy.__class__.__bases__[0]
    grandparent_cls = parent_cls.__bases__[0]
    parent_cls.setup = parent_setup_mock = mocker.Mock()
    grandparent_cls.setup = grandparent_setup_mock = mocker.Mock()

    # test
    test_trainer = Trainer()
    strategy.setup(test_trainer)

    # assertions
    parent_setup_mock.assert_not_called()
    grandparent_setup_mock.assert_called_once_with(test_trainer)


def test_setup_environment(mocker, full_config):
    # tst setup
    test_tsm_rank = 21
    tsm_mock = mocker.patch(MODULE_PATH + ".tsm", return_value=test_tsm_rank)

    # strategy setup
    full_config.use_smp_model = True
    strategy = SageMakerFSDPStrategy(full_config)
    parent_cls = strategy.__class__.__bases__[0]
    grandparent_cls = parent_cls.__bases__[0]
    parent_cls.setup_environment = parent_setup_mock = mocker.Mock()
    grandparent_cls.setup_environment = grandparent_setup_mock = mocker.Mock()

    # attributes setup
    world_size = strategy.world_size
    global_rank = strategy.global_rank
    local_rank = strategy.local_rank
    test_tensor_parallel_degree = 4
    test_context_parallel_degree = 2
    strategy.smp_config_dict = {
        "tensor_parallel_degree": test_tensor_parallel_degree,
        "context_parallel_degree": test_context_parallel_degree,
    }

    # test
    strategy.setup_environment()

    # assertions
    parent_setup_mock.assert_not_called()
    grandparent_setup_mock.assert_called_once()
    tsm_mock.init.assert_called_once()

    app_state = SageMakerAppState()
    assert app_state.world_size == world_size
    assert app_state.global_rank == global_rank
    assert app_state.local_rank == local_rank
    assert app_state.data_parallel_size == world_size // (test_tensor_parallel_degree * test_context_parallel_degree)
    assert app_state._is_megatron_initialized == True


class Test_lightning_module_state_dict:
    """Pending: Requires further implementation"""


class Test_optimizer_state:
    """Pending: Requires further implementation"""


class Test_load_model_state_dict:
    """Pending: Requires further implementation"""


class Test_load_optimizer_state_dict:
    """Pending: Requires further implementation"""


class Test_save_checkpoint:
    """Pending: Requires further implementation"""


class Test_load_checkpoint:
    """Pending: Requires further implementation"""


class Test_remove_checkpoint:
    """Pending: Requires further implementation"""


class Test_restore_checkpoint_after_setup:
    """Pending: Requires further implementation"""
