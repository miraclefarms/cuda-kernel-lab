#!/usr/bin/env bash
# Start the reproduction container with the flags the benchmarks actually need.
#
#   ./docker/run.sh                 # interactive shell
#   ./docker/run.sh <command...>    # run one command and exit
set -euo pipefail

IMAGE="${CUDA_KERNEL_LAB_IMAGE:-cuda-kernel-lab:13.3.1}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# --cap-add=SYS_ADMIN is what lets ncu read hardware counters. It is dropped
# automatically if the host forbids it; the benchmarks still run, only profiling
# is lost. --ipc=host avoids the 64MB default shm, which large allocations hit.
exec docker run --rm -it \
  --gpus all \
  --cap-add=SYS_ADMIN \
  --ipc=host \
  --ulimit memlock=-1 \
  -v "${REPO}:/workspace" \
  -w /workspace \
  "${IMAGE}" \
  "$@"
