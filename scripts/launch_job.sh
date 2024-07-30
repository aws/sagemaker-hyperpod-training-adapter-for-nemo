#!/bin/bash

set -ex


# Sample cmd for the current script:
#                  1               2          3            4             >=5
# ./launch_fsdp.sh $CONTAINER_NAME $NUM_NODES $MASTER_NODE $SHELL_SCRIPT $@

# **Assumes** the actual launch script ($SHELL_SCRIPT) takes >=2 arguments:
#   1. num nodes
#   2. master node
#   3. Optionally more args as needed


# - Passed in args.
CONTAINER_NAME=$1
# HOSTFILE=$2
SHELL_SCRIPT=$2  # Absolute or relative to `pwd`

# - Derived args: N.A.

echo "envs are"
env



MASTER_ADDR=$(scontrol show hostnames | sort | head -n 1)
NNODES=$SLURM_NTASKS
# docker exec -w `pwd` $CONTAINER_NAME bash -c "SLURM_PROCID=$SLURM_PROCID SLURM_JOB_ID=$SLURM_JOB_ID SLURM_JOB_NAME=$SLURM_JOB_NAME $SHELL_SCRIPT --hostfile $HOSTFILE $EXTRA_ARGS"
docker exec -w `pwd` $CONTAINER_NAME bash -c "SLURM_PROCID=$SLURM_PROCID SLURM_JOB_ID=$SLURM_JOB_ID SLURM_JOB_NAME=$SLURM_JOB_NAME $SHELL_SCRIPT -n $NNODES --master_address $MASTER_ADDR"
