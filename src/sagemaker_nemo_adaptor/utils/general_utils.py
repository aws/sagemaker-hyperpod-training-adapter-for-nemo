import os


def is_power_of_two(n: int) -> bool:
    "Brian Kernighan's Algorithm"
    return n > 0 and n & (n - 1) == 0


def is_slurm_run():
    """Check if the script is running under SLURM."""
    return "SLURM_JOB_ID" in os.environ
