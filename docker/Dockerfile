# Reproduction image for cuda-kernel-lab.
#
# Pinned by digest, not just by tag: a floating tag would let the toolchain move
# under a published number, which is exactly what this repo exists not to do.
#
# Base: CUDA 13.3.1 devel on Ubuntu 24.04. 13.3 is the newest general-availability
# 13.x line at the time of writing and carries everything this series needs —
# CUDA Tile (13.1), cuTile Python (13.2), the open Tile IR spec (13.3). 13.4 is a
# developer preview whose headline additions are Rubin-only, and this project has
# no Rubin hardware.
FROM nvidia/cuda:13.3.1-devel-ubuntu24.04@sha256:4ff859525f99de5782aa73607ce24219b07dddd48d12b97c1c301d7e1cfb0a87

ENV DEBIAN_FRONTEND=noninteractive

# Nsight Compute is installed explicitly rather than assumed: the devel image's
# contents vary across CUDA lines, and a missing ncu would only surface halfway
# through a profiling run.
# Ubuntu 24.04 marks its Python environment externally-managed (PEP 668), so the
# plotting stack comes from apt rather than pip.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        cmake \
        cuda-nsight-compute-13-3 \
        cuda-nsight-systems-13-3 \
        git \
        ninja-build \
        python3 \
        python3-matplotlib \
        python3-numpy \
    && rm -rf /var/lib/apt/lists/*

# Nsight installs under a versioned directory whose name tracks the Nsight
# release, not the CUDA one, so it is resolved at build time instead of guessed.
RUN set -eux; \
    for tool in ncu ncu-ui nsys; do \
      found="$(find /opt/nvidia -maxdepth 3 -name "$tool" -type f -perm -u+x 2>/dev/null | head -1)"; \
      if [ -n "$found" ]; then ln -sf "$found" "/usr/local/bin/$tool"; fi; \
    done; \
    ncu --version | head -1

ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV CUDA_MODULE_LOADING=LAZY

WORKDIR /workspace

# Sanity line for `docker run <image>` with no command: prints the toolchain the
# results in this container will be attributed to.
CMD ["bash", "-lc", "nvcc --version && cmake --version | head -1 && (ncu --version | head -1 || echo 'ncu unavailable')"]
