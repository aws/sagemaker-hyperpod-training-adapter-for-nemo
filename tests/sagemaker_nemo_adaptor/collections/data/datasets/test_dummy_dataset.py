import pytest
import torch

from sagemaker_nemo_adaptor.collections.data.datasets import DummyDataset


@pytest.fixture
def dummy_dataset():
    return DummyDataset()


def test_init(dummy_dataset):
    assert dummy_dataset.vocab_size == 1024
    assert dummy_dataset.seqlen == 2048
    assert dummy_dataset.length == 100000
    assert torch.all(dummy_dataset.mask == torch.ones((2048,)))


def test_getitem(dummy_dataset):
    item = dummy_dataset[0]
    assert len(item) == 2
    assert item[0].shape == (2048,)
    assert item[0].dtype == torch.long
    assert torch.all(item[1] == torch.ones((2048,)))


def test_len(dummy_dataset):
    assert len(dummy_dataset) == 100000


def test_data_type_bert():
    with pytest.raises(NotImplementedError):
        DummyDataset(data_type="bert")
