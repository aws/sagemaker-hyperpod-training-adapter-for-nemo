import pytest

from sagemaker_nemo_adaptor.collections.data import DummyDataModule


class TestDummyDataModule:
    @pytest.fixture
    def dummy_data_module(self, mocker):

        cfg = mocker.MagicMock()
        cfg.model.vocab_size = 10
        cfg.model.max_context_width = 20
        trainer = mocker.MagicMock()

        dummy_data_module = DummyDataModule(trainer=trainer, cfg=cfg)

        return dummy_data_module

    def test_val_dataloader(self, dummy_data_module):
        val_dataloader = dummy_data_module.val_dataloader()
        assert val_dataloader is None

    def test_get_batch(self, dummy_data_module):
        data = [(1, 2), (3, 4)]
        batch = dummy_data_module.get_batch(data)
        assert batch == ((1, 2), (3, 4), (1, 2))

    def test_get_val_batch(self, dummy_data_module):
        data = [(1, 2), (3, 4)]
        batch = dummy_data_module.get_val_batch(data)
        assert batch == ((1, 2), (3, 4), (1, 2))
