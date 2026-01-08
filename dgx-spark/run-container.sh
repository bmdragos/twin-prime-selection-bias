#!/bin/bash
# Run NGC PyTorch container on DGX Spark for twin prime experiments
#
# Usage: ./run-container.sh
#
# Prerequisites:
#   1. SSH into DGX: ssh spark-dcf7.local
#   2. Sync code: rsync -avz --exclude='.venv' ... from Mac
#   3. Login to NGC (first time): docker login nvcr.io
#      Username: $oauthtoken
#      Password: <your-api-key from ngc.nvidia.com>

# Mount home directory project to /workspace/twin-prime-selection-bias
# This allows rsync to host, container sees changes immediately
docker run -it --runtime=nvidia --gpus=all \
    --name twinprime \
    -v ~/twin-prime-selection-bias:/workspace/twin-prime-selection-bias \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    nvcr.io/nvidia/pytorch:25.10-py3
