#!/bin/bash
#SBATCH --output=slurm-%x-%j.out


set -xo pipefail


DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROGRAM="$0"

# export DOCKER_IMAGE="658645717510.dkr.ecr.us-west-2.amazonaws.com/smdistributed-modelparallel:2.2.0-gpu-py310-cu121-ubuntu20.04-sagemaker-smpv2.3.1"
export DOCKER_IMAGE="855988369404.dkr.ecr.us-west-2.amazonaws.com/chehaoha-test:2.2.0-gpu-py310-cu121-ubuntu20.04-sagemaker-smpv2.3.1-nemo"

export NODES=4
export TEST_NAME="chehaoha_llama3_kan"
export OUT_DIR="llama3_job"
export CONTAINER_NAME="llama3_benchmark_container"

export PARTITION="benchmark"


PREFIX="/fsx/users/chehaoha1/kandinsky/rubik/SMModelParallelExamples"
BIN="${PREFIX}/SMModelParallelExamples/bin"
export PATH="${BIN}:${PATH}"


export CURDIR="/fsx/users/chehaoha1/kandinsky/adapter/SageMakerNeMoAdaptor/scripts"
export PATH="${CURDIR}:${PATH}"


"${BIN}/smprun" \
    -v2 \
    -i "${DOCKER_IMAGE}" \
    -cn "${CONTAINER_NAME}" \
    "${CURDIR}/train_llama.sh"
