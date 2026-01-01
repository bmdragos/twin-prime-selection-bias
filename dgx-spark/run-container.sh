#!/bin/bash
# Run NGC PyTorch container on DGX Spark for twin prime experiments
#
# Usage: ./run-container.sh
#
# Prerequisites:
#   1. SSH into DGX: ssh spark-dcf7.local
#   2. Login to NGC: docker login nvcr.io
#      Username: $oauthtoken
#      Password: <your-api-key from ngc.nvidia.com>

docker run -it --runtime=nvidia --gpus=all \
    --name twinprime \
    -v $(dirname $(pwd)):/workspace/project \
    --ipc=host \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    nvcr.io/nvidia/pytorch:25.10-py3
