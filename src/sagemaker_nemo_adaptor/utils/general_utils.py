import os

from packaging import version as pversion

from sagemaker_nemo_adaptor.constants import MULTI_MODAL_HF_VERSION


def is_power_of_two(n: int) -> bool:
    "Brian Kernighan's Algorithm"
    return n > 0 and n & (n - 1) == 0


def is_slurm_run():
    """Check if the script is running under SLURM."""
    return "SLURM_JOB_ID" in os.environ


def can_use_multimodal() -> bool:
    import transformers

    current_version = pversion.parse(transformers.__version__)
    required_version = pversion.parse(MULTI_MODAL_HF_VERSION)

    return current_version >= required_version
