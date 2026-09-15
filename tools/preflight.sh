#!/usr/bin/env bash
# Machine pre-flight. Run this FIRST on any machine, before writing or timing
# anything, and again whenever the driver or toolkit changes.
#
# Replaces the older env_capture.sh: that only printed a snapshot, which meant a
# blocking condition could be read past and discovered three experiments later.
# This gates.
#
#   ./tools/preflight.sh            # check and print
#   ./tools/preflight.sh --matrix   # also emit a markdown block for environment-matrix.md
#
# Exit 0 = safe to measure. Exit 1 = a hard blocker; fix it before collecting data.
set -uo pipefail

HARD_FAIL=0
SOFT=()
MATRIX=0
[ "${1:-}" = "--matrix" ] && MATRIX=1

say()  { printf '%-34s %s\n' "$1" "$2"; }
fail() { printf '%-34s \033[31mBLOCK\033[0m  %s\n' "$1" "$2"; HARD_FAIL=1; }
warn() { printf '%-34s \033[33mWARN\033[0m   %s\n' "$1" "$2"; SOFT+=("$2"); }
ok()   { printf '%-34s \033[32mok\033[0m     %s\n' "$1" "$2"; }

echo "== cuda-kernel-lab pre-flight =="
echo

if ! command -v nvidia-smi >/dev/null 2>&1; then
  fail "nvidia-smi" "not found — no driver on this machine"
  exit 1
fi

# --- 1. driver: CUDA 13.x wants >= 580.65.06, but a forward-compat UMD can
# lift the user-mode driver without touching the kernel module. The a100 box
# runs CUDA 13.3 through the cuda-13.0 compat UMD (580.178.04) on a 575 kernel
# module; newer compat UMDs (590/595/610) fail cuInit. Whether a compat path
# works is empirical, so probe it instead of assuming. tools/cuda_compat.py is
# the single source of truth, shared with run.py's CSV record.
COMPAT_SCRIPT="$(cd "$(dirname "$0")" && pwd)/cuda_compat.py"
COMPAT_NEEDED=0 COMPAT_USABLE=1 COMPAT_DIR="" COMPAT_UMD="" COMPAT_REASON=""
if command -v python3 >/dev/null 2>&1 && [ -f "$COMPAT_SCRIPT" ]; then
  eval "$(python3 "$COMPAT_SCRIPT" --shell 2>/dev/null)" || true
fi

DRIVER="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | tr -d ' ')"
DRV_MAJOR="${DRIVER%%.*}"
if [ "${DRV_MAJOR:-0}" -ge 580 ] 2>/dev/null; then
  ok "driver $DRIVER" ">= 580.65.06, CUDA 13.x supported"
elif [ "$COMPAT_USABLE" = "1" ]; then
  warn "driver $DRIVER" "< 580.65.06; forward-compat UMD ${COMPAT_UMD} via ${COMPAT_DIR} works — export LD_LIBRARY_PATH before building/measuring"
else
  fail "driver $DRIVER" "CUDA 13.x needs >= 580.65.06; no forward-compat UMD passed cuInit"
fi

# --- 2. identity: which card is this actually
GPU="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
MEM="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)"
COUNT="$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l | tr -d ' ')"
say "gpu" "$GPU  (${MEM} MiB, ${COUNT} visible)"
case "$GPU" in
  *H20*)  say "machine codename" "h20  — verify HBM3 (4.0 TB/s) vs HBM3e (4.8 TB/s) in machine-peaks.json" ;;
  *H200*) say "machine codename" "h200" ;;
  *A100*) say "machine codename" "a100" ;;
  *)      warn "machine codename" "unrecognized GPU; add it to bench/machine-peaks.json before measuring" ;;
esac

# --- 3. MIG: a sliced GPU has no full SM view and its numbers are not comparable
MIG="$(nvidia-smi --query-gpu=mig.mode.current --format=csv,noheader | head -1 | tr -d ' ')"
if [ "$MIG" = "Enabled" ]; then
  fail "MIG" "enabled — no full SM view, cluster behaviour restricted, data not comparable"
else
  ok "MIG" "${MIG:-N/A}"
fi

# --- 4. exclusivity: what corrupts timing is active GPU use, not the mere
# presence of a context. A resident inference server that is idle is fine; a
# process that is actually running time-slices with our kernels and, measured on
# this repo, halves the batch (hot) bandwidth. So the criterion is sampled
# utilization == 0, NOT process count — a co-tenant can even be invisible in the
# process list across a container PID namespace while still using the GPU.
# 判据是「利用率是否为 0」而非进程数：常驻但空闲的 context 允许存在，真正在用 GPU
# 才禁止采集。跨容器 PID 命名空间下进程列表可能看不到占用者，利用率更可靠。
PROCS="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l | tr -d ' ')"
BUSY=""
for _ in 1 2 3 4 5; do
  while IFS=, read -r idx util; do
    idx="${idx// /}"; util="${util// /}"
    if [ "${util:-0}" -gt 0 ] 2>/dev/null; then
      case " $BUSY " in *" $idx:"*) ;; *) BUSY="$BUSY $idx:${util}%" ;; esac
    fi
  done < <(nvidia-smi --query-gpu=index,utilization.gpu --format=csv,noheader,nounits 2>/dev/null)
  sleep 0.2
done
if [ -n "$BUSY" ]; then
  fail "GPU utilization" "in use:$BUSY — time-slicing corrupts timing; do not collect"
else
  ok "GPU utilization" "0% on all visible GPUs"
fi
if [ "${PROCS:-0}" -gt 0 ]; then
  say "resident processes" "$PROCS context(s) present — allowed while idle"
else
  say "resident processes" "none"
fi

# --- 5. clock locking: decides whether absolute TFLOPS may be quoted at all
if nvidia-smi -q -d SUPPORTED_CLOCKS >/dev/null 2>&1; then
  if [ "$(id -u)" = "0" ]; then
    ok "clock lock" "root — lock with 'nvidia-smi -lgc <min>,<max>', release with '-rgc'"
  else
    warn "clock lock" "not root — cannot lock; run tools/run.py WITHOUT --clocks-locked and report ratios"
  fi
else
  warn "clock lock" "supported clocks unreadable — treat as unlockable"
fi

# --- 6. ncu counters
if command -v ncu >/dev/null 2>&1; then
  NCU_V="$(ncu --version 2>/dev/null | grep -i version | head -1)"
  if [ -r /proc/driver/nvidia/params ] && \
     grep -q 'RestrictProfilingToAdminUsers: 1' /proc/driver/nvidia/params 2>/dev/null && \
     [ "$(id -u)" != "0" ]; then
    warn "ncu counters" "RestrictProfilingToAdminUsers=1 and not root — no counter evidence this session"
  else
    ok "ncu" "${NCU_V:-present}"
  fi
else
  warn "ncu" "not installed — no counter evidence; wall-clock + bandwidth inference only"
fi

# --- 7. toolchain
if command -v nvcc >/dev/null 2>&1; then
  ok "nvcc" "$(nvcc --version | grep release | sed 's/.*release //')"
else
  fail "nvcc" "not found — build the container (docs/container.md) or install CUDA 13.x"
fi
command -v cmake >/dev/null 2>&1 && ok "cmake" "$(cmake --version | head -1 | sed 's/cmake version //')" \
                                 || fail "cmake" "not found"
python3 -c "import matplotlib" 2>/dev/null && ok "matplotlib" "present" \
                                           || warn "matplotlib" "missing — tools/plot.py will fail"

# --- 8. clean tree: dirty-tree data may not be published
if [ -n "$(git -C "$(dirname "$0")/.." status --porcelain 2>/dev/null)" ]; then
  warn "git tree" "dirty — data collected now is marked git_dirty=yes and cannot go into an article"
else
  ok "git tree" "clean"
fi

echo
if [ "$HARD_FAIL" = "1" ]; then
  echo "RESULT: BLOCKED — fix the items marked BLOCK before collecting any data."
else
  echo "RESULT: safe to measure."
  [ "${#SOFT[@]}" -gt 0 ] && echo "        ${#SOFT[@]} warning(s) — they constrain what the numbers may claim."
fi

if [ "$MATRIX" = "1" ]; then
  echo
  echo "---- paste into docs/environment-matrix.md ----"
  echo "| 驱动版本 | $DRIVER |"
  echo "| CUDA Toolkit | $(nvcc --version 2>/dev/null | grep release | sed 's/.*release //' || echo '未安装') |"
  if [ "$COMPAT_NEEDED" = "1" ] && [ "$COMPAT_USABLE" = "1" ]; then
    echo "| CUDA UMD | ${COMPAT_UMD}（forward-compat，\`LD_LIBRARY_PATH=${COMPAT_DIR}\`）|"
  else
    echo "| CUDA UMD | 随内核驱动（无需 forward-compat）|"
  fi
  echo "| 容器镜像 | 待填 |"
  echo "| bench-probe 输出 | 待填（运行 ./build/kernels/01-execution-model/bench-probe 回填）|"
  echo "| 实测 streaming 上限 / 标称 | 待填（bench-probe 的 best cold/hot，同时写入 bench/machine-peaks.json）|"
  echo "| 锁频权限 | $( [ "$(id -u)" = 0 ] && echo '✅ root 可锁' || echo '❌ 非 root' ) |"
  echo "| ncu 计数器权限 | $(command -v ncu >/dev/null 2>&1 && echo '待确认（见上）' || echo '❌ 未安装') |"
  echo "| MIG | ${MIG:-N/A} |"
  echo "| 独占（util=0） | $( [ -z "$BUSY" ] && echo '✅ 所有可见卡 utilization.gpu = 0' || echo "❌ 在用:$BUSY" ) |"
fi

exit "$HARD_FAIL"
