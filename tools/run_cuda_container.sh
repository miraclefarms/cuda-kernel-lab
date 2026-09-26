#!/bin/bash
# Start the CUDA 13.3.1 devel container in a fully privileged profile-friendly
# configuration:
#   * --privileged            -> root inside == root outside, so nvidia-smi -lgc
#                                (clock locking) and ncu perf-counter access work.
#   * --gpus all              -> nvidia runtime mounts all GPU devices.
#   * NVIDIA_DRIVER_CAPABILITIES=all -> exposes compute/utility/graphics/etc.
#   * --cap-add SYS_ADMIN/SYS_PTRACE/IPC_LOCK + --ipc=host -> needed by ncu to
#                                attach and read PMU/performance counters.
#   * --pid=host --network=host -> simplified cross-process profiling.
#
# Usage:
#   ./run_cuda_container.sh [COMMAND...]
#   (default COMMAND is bash; pass e.g. "ncu --version" to run one-shot)

IMAGE="${IMAGE:-nvidia/cuda:13.3.1-devel-ubuntu24.04}"
NAME="${NAME:-cuda1331-devel}"
WORKDIR="${WORKDIR:-/workspace}"
HOST_MOUNT="${HOST_MOUNT:-$(pwd)}"

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "Image $IMAGE not found, pulling..."
  docker pull "$IMAGE"
fi

# Reuse or (re)start the container.
if docker ps -a --format '{{.Names}}' | grep -qx "$NAME"; then
  docker rm -f "$NAME" >/dev/null 2>&1
fi

docker run -it -d \
  --name "$NAME" \
  --gpus all \
  --privileged \
  --cap-add=SYS_ADMIN \
  --cap-add=SYS_PTRACE \
  --cap-add=IPC_LOCK \
  --ipc=host \
  --pid=host \
  --network=host \
  --ulimit memlock=-1 \
  -v "$HOME:$HOME" \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e TERM="$TERM" \
  -v "$HOST_MOUNT:$WORKDIR" \
  -w "$WORKDIR" \
  "$IMAGE" \
  "${@:-bash}"
