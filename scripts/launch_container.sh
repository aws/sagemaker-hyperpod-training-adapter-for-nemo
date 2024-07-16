#!/usr/bin/env bash
#
set -e

export CONTAINER_NAME=${2:-"smp"}
export CONTAINER_IMAGE=${1:-"658645717510.dkr.ecr.us-west-2.amazonaws.com/smdistributed-modelparallel:2.2.0-gpu-py310-cu121-ubuntu20.04-sagemaker-smpv2.3.1"}

echo "image is $CONTAINER_IMAGE"
# Set the docker ECR path to your DLC
aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin 855988369404.dkr.ecr.us-west-2.amazonaws.com
aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin 658645717510.dkr.ecr.us-west-2.amazonaws.com

FSX_MOUNT="-v /fsx/:/fsx"
NFS_MOUNT="-v /nfs/:/nfs"

docker stop $(docker ps -q -a) || true
docker rm $(docker ps -q -a) || true

docker pull ${CONTAINER_IMAGE}

docker run --runtime=nvidia --gpus 8 \
    --privileged \
    --rm \
    -d \
    --name $CONTAINER_NAME \
    --uts=host --ulimit stack=67108864 --ulimit memlock=-1 --ipc=host --net=host \
    --device=/dev/infiniband/uverbs0 \
    --device=/dev/infiniband/uverbs1 \
    --device=/dev/infiniband/uverbs2 \
    --device=/dev/infiniband/uverbs3 \
    --device=/dev/gdrdrv \
    --security-opt seccomp=unconfined  \
    ${FSX_MOUNT} \
    ${NFS_MOUNT} \
    ${CONTAINER_IMAGE} sleep infinity

# Allow containers to talk to each other
docker exec -itd ${CONTAINER_NAME} bash -c "printf \"Port 2022\n\" >> /etc/ssh/sshd_config"
docker exec -itd ${CONTAINER_NAME} bash -c "printf \"  Port 2022\n\" >> /root/.ssh/config"
docker exec -itd ${CONTAINER_NAME} bash -c "service ssh start"

exit 0
