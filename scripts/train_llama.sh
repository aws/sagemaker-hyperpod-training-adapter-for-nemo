#!/usr/bin/env bash

parse_inputs() {
    while [[ $# -gt 0 ]]; do
        key="$1"
        case $key in
        -n | --nnodes)
            NNODES="$2"
            shift 2
            ;;
        --master_address)
            MASTER_ADDR=$2
            shift 2
            ;;
        *)
            shift 1
            ;;
        esac
    done
}

parse_inputs $@

echo "HF is $hostfile"
# if [ -z "$hostfile" ]; then
#     echo "Hostfile needs to be passed"
#     exit 1
# fi

# NNODES=$(cat $hostfile | wc -l)



export NCCL_PROTO="simple"
export NCCL_SOCKET_IFNAME="^lo,docker"
export RDMAV_FORK_SAFE=1
export FI_EFA_USE_DEVICE_RDMA=1
export NCCL_DEBUG_SUBSYS=off
export NCCL_DEBUG="INFO"
export SM_NUM_GPUS=8
# export MASTER_ADDR=$(head -n 1 $hostfile)
export GPU_NUM_DEVICES=8



echo nnode

TORCH_CMD="torchrun --nnodes=${NNODES} --nproc_per_node=8"

# If nsys path provided, profile using Nsys, but only on 1st node. Requires job to be launched using sbatch
if [[ -n $nsys_path ]]; then
    profile_nsys=1
    if [[ $SLURM_PROCID -eq 1 ]]; then
        NSYS_CMD="nsys profile -w true -t cuda,nvtx,osrt,cudnn,cublas -s cpu  --capture-range=cudaProfilerApi --cuda-memory-usage=true --cudabacktrace=true -x true -o $nsys_path --force-overwrite=true"
        TORCH_CMD="$NSYS_CMD $TORCH_CMD"
    fi
else
    profile_nsys=0
fi

export PYTHONPATH="${PYTHONPATH}:/fsx/users/chehaoha1/kandinsky/adapter/SageMakerNeMoAdaptor/src" # TODO: remove when package installation ready
echo "cmd $TORCH_CMD"
$TORCH_CMD \
    --rdzv_endpoint=$MASTER_ADDR:29400 --rdzv_id=100 --rdzv_backend=c10d \
    llama_pretrain.py
