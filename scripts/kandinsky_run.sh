#!/bin/bash

#SBATCH --output=logs/%x_%j.out  # Redirects outputs to file in current_dir/logs
#SBATCH --error=logs/%x_%j.out  # Redirects err to same file in current_dir/logs


parse_inputs() {
    WORK_DIR=""
    while [[ $# -gt 0 ]]; do
        key="$1"
        case $key in
        -d | --dir)
            WORK_DIR="$2/"
            shift 2
            ;;
        -i | --image)
            CONTAINER_IMAGE=$2
            shift 2
            ;;
        -cn | --container_name)
            CONTAINER_NAME=$2
            shift 2
            ;;
        *)
            break
            ;;
        esac
    done

    SHELL_SCRIPT=$1
    shift 1
    EXTRA_ARGS=$@
}


parse_inputs $@


echo "jobs host $HOSTS nn $NNODES st $SLURM_NTASKS"

echo "Launching Container: \`$CONTAINER_IMAGE\` ..."

echo "container is $CONTAINER_IMAGE"


srun -l ${WORK_DIR}launch_container.sh $CONTAINER_IMAGE $CONTAINER_NAME

echo "Launching Job: \`$SHELL_SCRIPT\` ($MODE) ..."

srun -l ${WORK_DIR}launch_job.sh  $CONTAINER_NAME  $SHELL_SCRIPT
