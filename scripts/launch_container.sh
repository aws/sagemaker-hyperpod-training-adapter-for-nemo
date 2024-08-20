#!/usr/bin/env bash
#
set -euo pipefail

TOKEN="$(curl -sS -X PUT "http://169.254.169.254/latest/api/token" -H "X-aws-ec2-metadata-token-ttl-seconds: 21600")"
INSTANCE_TYPE="$(curl -sS -H "X-aws-ec2-metadata-token: $TOKEN" http://169.254.169.254/latest/meta-data/instance-type)"

export CONTAINER_IMAGE=${1:-"658645717510.dkr.ecr.us-west-2.amazonaws.com/smdistributed-modelparallel:2.2.0-gpu-py310-cu121-ubuntu20.04-sagemaker-smpv2.3.1"}
export CONTAINER_NAME=${2:-"smp"}

echo "image is $CONTAINER_IMAGE"
# Set the docker ECR path to your DLC
aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin 855988369404.dkr.ecr.us-west-2.amazonaws.com
aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin 658645717510.dkr.ecr.us-west-2.amazonaws.com

# Append required devices
device=("--device=/dev/gdrdrv")
while IFS= read -r -d '' d; do
  device+=("--device=${d}")
done < <(find "/dev/infiniband" -name "uverbs*" -print0)

FSX_MOUNT="-v /fsx/:/fsx"
NFS_MOUNT="-v /nfs/:/nfs"

docker ps -q -a | xargs -I{} docker stop {}
docker ps -q -a | xargs -I{} docker rm -f {}
docker pull "${CONTAINER_IMAGE}"

docker pull "$CONTAINER_IMAGE"

extra=()
if [[ "$INSTANCE_TYPE" =~ "p4d" ]]; then
  extra+=(--runtime=nvidia)
  extra+=(-v "${NFS_MOUNT}")
fi

docker run --gpus 8 \
    --privileged \
    --rm \
    -d \
    --name "$CONTAINER_NAME" \
    --uts=host --ulimit stack=67108864 --ulimit memlock=-1 --ipc=host --net=host \
    --security-opt seccomp=unconfined  \
    -v "${FSX_MOUNT}" \
    "${device[@]}" \
    "${extra[@]}" \
    "${CONTAINER_IMAGE}" sleep infinity

# Allow containers to talk to each other
docker exec -itd "$CONTAINER_NAME" bash -c "printf \"Port 2022\n\" >> /etc/ssh/sshd_config"
docker exec -itd "$CONTAINER_NAME" bash -c "printf \"  Port 2022\n\" >> /root/.ssh/config"
docker exec -itd "$CONTAINER_NAME" bash -c "service ssh start"

exit 0
