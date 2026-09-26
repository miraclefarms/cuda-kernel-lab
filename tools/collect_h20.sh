#!/usr/bin/env bash
# One H20 experiment at a time. Run inside an existing CUDA development
# environment from a clean checkout. No host/container identity is embedded.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

usage() {
  cat <<'EOF'
Usage: ./tools/collect_h20.sh <build|preflight|01|02|02-unlocked|03> [physical GPU index]

  build        configure sm_90a and compile the three article targets
  preflight N  check the chosen physical GPU and print matrix fields
  01 N         locked-clock streaming baseline + static occupancy scan
  02 N         locked-clock measurement-protocol sweep + raw samples
  02-unlocked N  same sweep without locking, for the clock-control comparison
  03 N         locked-clock cp.async tile/shape sweep

Environment: H20_SM_CLOCK_MHZ=1800 (default), H20_RUN_TAG=<unique suffix> for
repeat runs on the same day. The script never overwrites an existing CSV.
EOF
}

PHASE="${1:-}"
GPU="${2:-}"
case "$PHASE" in
  build)
    cmake -B build -DCMAKE_CUDA_ARCHITECTURES=90a
    cmake --build build --target bench-probe occupancy-scan \
      kernel-02-measurement-discipline kernel-03-smem-staging -j
    exit 0
    ;;
  preflight|01|02|02-unlocked|03) ;;
  *) usage; exit 2 ;;
esac

if ! [[ "$GPU" =~ ^[0-9]+$ ]]; then
  usage >&2
  exit 2
fi
if [ "$PHASE" = preflight ]; then
  exec ./tools/preflight.sh --gpu "$GPU" --matrix
fi

if [ "$(python3 -c 'import sys; sys.path.insert(0,"tools"); from run import git_dirty; print(git_dirty())')" != no ]; then
  echo "Source tree is dirty. Commit code/doc changes before collecting publishable data." >&2
  exit 1
fi
./tools/preflight.sh --gpu "$GPU"

GPU_NAME="$(nvidia-smi -i "$GPU" --query-gpu=name --format=csv,noheader | head -1)"
if [[ "$GPU_NAME" != *H20* ]]; then
  echo "GPU $GPU is not an H20: $GPU_NAME" >&2
  exit 1
fi

DATE="$(date +%F)"
TAG="${H20_RUN_TAG:-}"
if [ "$PHASE" = 02-unlocked ]; then
  TAG="unlocked${TAG:+-$TAG}"
fi
if [ -n "$TAG" ] && ! [[ "$TAG" =~ ^[A-Za-z0-9_-]+$ ]]; then
  echo "H20_RUN_TAG must contain only letters, digits, _ or -" >&2
  exit 2
fi
STEM="$DATE-h20"
if [ -n "$TAG" ]; then STEM="$STEM-$TAG"; fi
case "$PHASE" in
  01) KERNEL=01-execution-model ;;
  02|02-unlocked) KERNEL=02-measurement-discipline ;;
  03) KERNEL=03-smem-staging ;;
esac
OUT_DIR="results/$KERNEL"
CSV="$OUT_DIR/$STEM.csv"
REPORT="$OUT_DIR/$STEM.quiet.json"
mkdir -p "$OUT_DIR"
if [ -e "$CSV" ] || [ -e "$CSV.invalid" ] || [ -e "$REPORT" ]; then
  echo "Output already exists for $STEM; set a unique H20_RUN_TAG before retrying." >&2
  exit 1
fi

RUN_ARGS=(--kernel "$KERNEL" --machine h20)
if [ -n "$TAG" ]; then RUN_ARGS+=(--tag "$TAG"); fi
LOCKED=0
cleanup() {
  local status=$?
  trap - EXIT
  if [ "$LOCKED" -eq 1 ]; then
    if ! nvidia-smi -i "$GPU" -rgc >&2; then
      echo "Failed to reset GPU $GPU clocks; reset them manually before the next run." >&2
      status=1
    fi
  fi
  exit "$status"
}
trap cleanup EXIT

if [ "$PHASE" != 02-unlocked ]; then
  CLOCK="${H20_SM_CLOCK_MHZ:-1800}"
  if ! [[ "$CLOCK" =~ ^[0-9]+$ ]]; then
    echo "H20_SM_CLOCK_MHZ must be an integer MHz value" >&2
    exit 2
  fi
  nvidia-smi -i "$GPU" -lgc "$CLOCK,$CLOCK" >&2
  LOCKED=1
  RUN_ARGS+=(--clocks-locked)
fi

if [ "$PHASE" = 01 ]; then
  OCCUPANCY_DIR="$OUT_DIR/occupancy"
  mkdir -p "$OCCUPANCY_DIR"
  OCCUPANCY="$OCCUPANCY_DIR/$STEM.csv"
  if [ -e "$OCCUPANCY" ]; then
    echo "Occupancy output already exists: $OCCUPANCY" >&2
    exit 1
  fi
  CUDA_VISIBLE_DEVICES="$GPU" ./build/kernels/01-execution-model/occupancy-scan > "$OCCUPANCY"
  RUN_ARGS+=(--binary build/kernels/01-execution-model/bench-probe)
fi

if [ "$PHASE" = 02 ] || [ "$PHASE" = 02-unlocked ]; then
  SAMPLES="$OUT_DIR/$STEM-samples.csv"
  if [ -e "$SAMPLES" ]; then
    echo "Sample output already exists: $SAMPLES" >&2
    exit 1
  fi
  RUN_ARGS+=(-- --samples-out "$SAMPLES")
fi

python3 tools/gpu_quiet_gate.py --need 1 --candidates "$GPU" \
  --csv "$CSV" --report "$REPORT" -- \
  python3 tools/run.py "${RUN_ARGS[@]}"
python3 tools/validate_h20_results.py "$CSV"
