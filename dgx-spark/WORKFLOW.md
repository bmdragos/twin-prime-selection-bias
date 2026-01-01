# DGX Spark Workflow

## SSH Connection

```bash
ssh spark-dcf7.local
# or
ssh bmdragos@192.168.1.159
```

## Container Management

### First Time Setup

```bash
# On DGX Spark
cd ~/twin-prime-selection-bias/dgx-spark
./run-container.sh     # Creates 'twinprime' container

# Inside container
./setup.sh             # Installs deps, tests GPU
```

### Start Existing Container

```bash
ssh spark-dcf7.local
docker start twinprime
docker exec -it twinprime bash
```

### Stop Container

```bash
docker stop twinprime
```

### Delete Container (to recreate)

```bash
docker rm twinprime
```

## Code Updates

**From Mac**, use tarball approach (avoids permission issues):

```bash
# Create tarball of changed files
cd /Users/bd/Math/twin-prime-selection-bias
tar -czf /tmp/update.tar.gz src/gpu_factorization.py run_gpu.py

# Copy to Spark host, then into container
scp /tmp/update.tar.gz spark-dcf7.local:/tmp/
ssh spark-dcf7.local "docker cp /tmp/update.tar.gz twinprime:/tmp/ && \
    docker exec twinprime bash -c 'cd /workspace/twin-prime-selection-bias && tar -xzf /tmp/update.tar.gz'"
```

Or for full project sync:
```bash
cd /Users/bd/Math
tar --exclude='.venv' --exclude='__pycache__' --exclude='.git' \
    -czf /tmp/twinprime.tar.gz twin-prime-selection-bias

scp /tmp/twinprime.tar.gz spark-dcf7.local:/tmp/
ssh spark-dcf7.local "docker cp /tmp/twinprime.tar.gz twinprime:/workspace/ && \
    docker exec twinprime bash -c 'cd /workspace && tar -xzf twinprime.tar.gz'"
```

## Running Experiments

```bash
# SSH into container
ssh spark-dcf7.local
docker exec -it twinprime bash

# Inside container
cd /workspace/twin-prime-selection-bias

# Run experiments
python run_gpu.py --K 1e7 --timestamp    # K=10^7, ~60s
python run_gpu.py --K 1e8 --timestamp    # K=10^8, ~3 min
python run_gpu.py --K 1e9 --timestamp    # K=10^9, ~30 min (untested)
```

## Retrieving Results

```bash
# From Mac
ssh spark-dcf7.local "docker cp twinprime:/workspace/twin-prime-selection-bias/data/results /tmp/results"
scp -r spark-dcf7.local:/tmp/results ./data/results-dgx
```

## GPU Utilization Notes

Current bottlenecks:
- **SPF sieve (50%)**: CPU-bound, ~84s for K=10^8
- **Model computation (40%)**: CPU transfer matrix products, ~70s
- **GPU omega (<5%)**: Very fast, ~3.5s total

GPU shows brief spikes because:
1. Kernel launches (fast)
2. Returns to idle during CPU model computation

For sustained GPU utilization, would need to:
- Parallelize SPF sieve (segmented sieve)
- Move transfer matrix to GPU (matrix multiplication)

## Dashboard

DGX dashboard at: http://192.168.1.159/ (check exact port)
