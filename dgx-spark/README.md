# DGX Spark Deployment

Run twin prime experiments on DGX Spark with GB10 GPU.

## Quick Start

```bash
# 1. Copy project to DGX
scp -r /Users/bd/Math/twin-prime-selection-bias spark-dcf7.local:~/

# 2. SSH in
ssh spark-dcf7.local

# 3. Run container (first time: docker login nvcr.io)
cd ~/twin-prime-selection-bias/dgx-spark
./run-container.sh

# 4. Inside container: setup (first time only)
cd /workspace/project/dgx-spark
./setup.sh

# 5. Run K=10^8 experiment
cd /workspace/project
python run_gpu.py
```

## Subsequent Runs

```bash
ssh spark-dcf7.local
docker start twinprime
docker exec -it twinprime bash

# Inside container:
cd /workspace/project
python run_gpu.py              # K=10^8 (default)
python run_gpu.py --K 1e9      # K=10^9 if you dare
```

## Sync Results Back

```bash
# From Mac:
scp -r spark-dcf7.local:~/twin-prime-selection-bias/data/results ./data/results-dgx
```

## Why NGC Container?

GB10 (Blackwell sm_121) requires NVIDIA's custom builds. Standard PyPI packages fail with `sm_121 is not compatible`. The NGC container has pre-built CUDA toolchain that works.
