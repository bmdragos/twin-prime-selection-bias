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
# 1. From Mac: sync code to DGX host
rsync -avz --exclude='.venv' --exclude='__pycache__' --exclude='.git' \
  /Users/bd/Math/twin-prime-selection-bias/ spark-dcf7.local:~/twin-prime-selection-bias/

# 2. SSH in and create container
ssh spark-dcf7.local
cd ~/twin-prime-selection-bias/dgx-spark
./run-container.sh     # Creates 'twinprime' container with volume mount

# 3. Inside container: install deps
./setup.sh
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

The container mounts `~/twin-prime-selection-bias` from the host. Just rsync to the host:

```bash
# From Mac - sync all code changes
rsync -avz --exclude='.venv' --exclude='__pycache__' --exclude='.git' \
  /Users/bd/Math/twin-prime-selection-bias/ spark-dcf7.local:~/twin-prime-selection-bias/
```

Container sees changes immediately via volume mount. No docker cp needed.

## Running Experiments

```bash
# SSH into container
ssh spark-dcf7.local
docker exec -it twinprime bash -c "cd /workspace/twin-prime-selection-bias && python run_gpu.py"

# Or interactive session
docker exec -it twinprime bash
cd /workspace/twin-prime-selection-bias
python run_gpu.py --K 1e7 --timestamp    # K=10^7, ~60s
python run_gpu.py --K 1e8 --timestamp    # K=10^8, ~3 min
python run_gpu.py --K 1e9 --timestamp    # K=10^9, ~190s
```

## Retrieving Results

```bash
# From Mac - results are on host via volume mount
rsync -avz spark-dcf7.local:~/twin-prime-selection-bias/data/results/ ./data/results-dgx/
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
