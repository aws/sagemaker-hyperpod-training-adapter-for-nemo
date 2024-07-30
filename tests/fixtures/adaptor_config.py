import pytest
from hydra import compose, initialize


@pytest.fixture()
def full_config():
    with initialize(version_base=None, config_path="../../scripts/conf"):
        cfg = compose(config_name="smp_llama_config")
        return cfg
