#!/usr/bin/env bash
# Snapshot the machine into docs/environment-matrix.md format.
#
# Run this once per machine before the first benchmark, and again whenever the
# driver or toolkit changes. tools/run.py captures the same fields per run; this
# script is for the human-readable matrix.
set -euo pipefail

echo "## $(hostname | sed 's/.*/<redacted>/')  —  $(date -Iseconds)"
echo
echo '```'
nvidia-smi --query-gpu=index,name,driver_version,memory.total,clocks.max.sm,clocks.max.mem,compute_mode,ecc.mode.current,mig.mode.current \
  --format=csv 2>/dev/null || echo "nvidia-smi unavailable"
echo
echo "nvcc: $(nvcc --version 2>/dev/null | grep release || echo 'not found')"
echo
echo "-- profiling permission --"
if [ -r /proc/driver/nvidia/params ]; then
  grep -i RestrictProfiling /proc/driver/nvidia/params || echo "RestrictProfilingToAdminUsers: not reported"
else
  echo "/proc/driver/nvidia/params unreadable"
fi
echo "ncu smoke test:"
ncu --version >/dev/null 2>&1 && echo "  ncu present" || echo "  ncu NOT present"
echo
echo "-- clock lock capability --"
nvidia-smi -q -d SUPPORTED_CLOCKS >/dev/null 2>&1 && echo "  supported clocks readable" || echo "  cannot read supported clocks"
echo "  (locking needs root: nvidia-smi -lgc <min>,<max>)"
echo '```'
