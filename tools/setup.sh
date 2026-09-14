#!/usr/bin/env bash
# First-run provisioning for the reproduction container.
#
# Run this the moment you land in the container, before collecting any data:
#
#   ./tools/setup.sh              # install what is missing, build, run preflight
#   ./tools/setup.sh --no-build   # install + validate only
#   ./tools/setup.sh --install-only
#
# Idempotent: a second run installs nothing and just re-validates. apt needs
# root (the devel image and docker/run.sh give you root). Everything installed
# here is also in docker/Dockerfile, so this doubles as the recovery path for a
# plain CUDA 13.x base and as the repo's environment gate. Set
# SETUP_SKIP_INSTALL=1 to validate without touching apt.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export DEBIAN_FRONTEND=noninteractive

DO_BUILD=1
DO_PREFLIGHT=1
for arg in "$@"; do
  case "$arg" in
    --no-build)     DO_BUILD=0 ;;
    --install-only) DO_BUILD=0; DO_PREFLIGHT=0 ;;
    -h|--help)
      cat <<'EOF'
usage: tools/setup.sh [--no-build] [--install-only]

  --no-build      install and validate, but do not build
  --install-only  install missing packages only

env:
  SETUP_SKIP_INSTALL=1  validate only, never call apt
EOF
      exit 0 ;;
    *) echo "unknown argument: $arg (try --help)" >&2; exit 2 ;;
  esac
done

WARNINGS=0
say()  { printf '%-26s %s\n' "$1" "$2"; }
ok()   { say "$1" "$(printf '\033[32mok\033[0m')   $2"; }
warn() { say "$1" "$(printf '\033[33mWARN\033[0m') $2"; WARNINGS=$((WARNINGS + 1)); }
have() { command -v "$1" >/dev/null 2>&1; }

echo "== cuda-kernel-lab setup =="
echo

# --- 1. toolchain packages. Only what is missing, so a re-run is a no-op.
CORE_PKGS="build-essential ca-certificates cmake ninja-build git python3 python3-matplotlib python3-numpy"
NSIGHT_PKGS="cuda-nsight-compute-13-3 cuda-nsight-systems-13-3"

if [ "${SETUP_SKIP_INSTALL:-0}" = "1" ]; then
  say "apt install" "skipped (SETUP_SKIP_INSTALL=1)"
else
  missing=""
  for pkg in $CORE_PKGS; do
    dpkg -s "$pkg" >/dev/null 2>&1 || missing="$missing $pkg"
  done

  if [ -z "$missing" ]; then
    ok "apt install" "core toolchain already present"
  elif [ "$(id -u)" != "0" ]; then
    warn "apt install" "not root; missing:$missing (re-run as root)"
  elif ! have apt-get; then
    warn "apt install" "no apt-get; cannot install:$missing"
  else
    say "apt install" "missing:$missing"
    if apt-get update && apt-get install -y --no-install-recommends $missing; then
      ok "apt install" "core toolchain installed"
    else
      warn "apt install" "core install failed; check network / apt sources"
    fi
  fi

  # Nsight is optional: without it there is no counter evidence, which the
  # methodology permits, so a failure here warns instead of stopping the setup.
  nsight_missing=""
  for pkg in $NSIGHT_PKGS; do
    dpkg -s "$pkg" >/dev/null 2>&1 || nsight_missing="$nsight_missing $pkg"
  done
  if [ -n "$nsight_missing" ] && [ "$(id -u)" = "0" ] && have apt-get; then
    apt-get install -y --no-install-recommends $nsight_missing >/dev/null 2>&1 \
      && ok "apt install" "nsight installed" \
      || warn "apt install" "nsight unavailable — profiling will be missing"
  fi

  # Nsight lands in a versioned directory; resolve it at run time rather than
  # guessing the path, exactly as docker/Dockerfile does at build time.
  for tool in ncu ncu-ui nsys; do
    have "$tool" && continue
    found="$(find /opt/nvidia -maxdepth 3 -name "$tool" -type f -perm -u+x 2>/dev/null | head -1)"
    if [ -n "$found" ]; then
      ln -sf "$found" "/usr/local/bin/$tool" && ok "link $tool" "$found"
    fi
  done
fi

echo
# --- 2. verify the toolchain the results will be attributed to.
have nvcc  && ok "nvcc"  "$(nvcc --version 2>/dev/null | grep release | sed 's/.*release //')" \
           || warn "nvcc" "missing — build docker/ or install CUDA 13.x"
have cmake && ok "cmake" "$(cmake --version | head -1 | sed 's/cmake version //')" \
           || warn "cmake" "missing"
have ninja && ok "ninja" "$(ninja --version 2>/dev/null)" || warn "ninja" "missing (make still works)"
have git   && ok "git"   "$(git --version | sed 's/git version //')" || warn "git" "missing"
python3 -c "import matplotlib, numpy" 2>/dev/null \
  && ok "python" "$(python3 -c 'import matplotlib; print("matplotlib", matplotlib.__version__)')" \
  || warn "python" "matplotlib/numpy missing — tools/plot.py will fail"
have ncu  && ok "ncu"  "$(ncu --version 2>/dev/null | grep -i version | head -1)" \
          || warn "ncu"  "missing — no counter evidence this session"
have nsys && ok "nsys" "$(nsys --version 2>/dev/null | head -1)" || warn "nsys" "missing"

echo
# --- 3. GPU: compute capability drives the build, driver gates CUDA 13.
ARCH="${CMAKE_CUDA_ARCHITECTURES:-}"
if have nvidia-smi; then
  driver="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | tr -d ' ')"
  cc="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' ')"
  if [ "$(printf '%s' "${driver%%.*}" | tr -cd '0-9')" -ge 580 ] 2>/dev/null; then
    ok "driver" "$driver (CUDA 13.x ok)"
  else
    warn "driver" "$driver is below 580.65.06 — CUDA 13.x will not run"
  fi
  ok "gpu" "$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
  if [ -z "$ARCH" ]; then
    case "$cc" in
      8.*)  ARCH=80 ;;
      9.*)  ARCH=90a ;;
      10.*) ARCH=100 ;;
      12.*) ARCH=120 ;;
    esac
  fi
  say "compute capability" "${cc:-unknown}  ->  -DCMAKE_CUDA_ARCHITECTURES=${ARCH:-<unknown>}"
else
  warn "nvidia-smi" "no GPU exposed; start with docker/run.sh (--gpus all)"
fi

# --- 4. build (only when the architecture is known).
if [ "$DO_BUILD" = "1" ]; then
  if [ -z "$ARCH" ]; then
    warn "build" "unknown arch; set CMAKE_CUDA_ARCHITECTURES and re-run"
  elif ! have nvcc; then
    warn "build" "no nvcc"
  else
    log="$REPO/build/setup-build.log"
    mkdir -p "$REPO/build"
    if cmake -B "$REPO/build" -DCMAKE_CUDA_ARCHITECTURES="$ARCH" >"$log" 2>&1 \
       && cmake --build "$REPO/build" -j >>"$log" 2>&1; then
      ok "build" "build/ ready (arch $ARCH)"
    else
      warn "build" "failed; last lines of $log:"
      tail -n 15 "$log" | sed 's/^/    /'
    fi
  fi
fi

# --- 5. the repo's own environment gate.
if [ "$DO_PREFLIGHT" = "1" ]; then
  echo
  if [ -x "$REPO/tools/preflight.sh" ]; then
    if ! "$REPO/tools/preflight.sh" --matrix; then
      warn "preflight" "BLOCKED — fix the BLOCK items before collecting data"
    fi
  else
    warn "preflight" "tools/preflight.sh missing"
  fi
fi

echo
if [ "$WARNINGS" -eq 0 ]; then
  echo "setup complete: environment ready."
else
  echo "setup complete with $WARNINGS warning(s) — read them before measuring."
fi
