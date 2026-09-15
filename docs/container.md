# 复现容器

目标：**读者只需要这个容器 + 这个仓库，就能跑出文章里的每一个数字。** 宿主机除了
NVIDIA 驱动和 container toolkit 之外不需要装任何东西。

## 基线镜像

```
nvidia/cuda:13.3.1-devel-ubuntu24.04
sha256:4ff859525f99de5782aa73607ce24219b07dddd48d12b97c1c301d7e1cfb0a87
```

**按 digest 固定，不只按 tag。** 浮动 tag 会让工具链在已发表的数字底下悄悄换掉，而这
正是本仓库存在的理由之一。

为什么是 13.3 而不是 13.4：13.3 是当前正式发布线里最新的一条，本系列需要的东西都在
里面——CUDA Tile（13.1 引入）、cuTile Python DSL（13.2）、开源 Tile IR 规范（13.3）。
13.4 仍是 developer preview，其头条新增（Rubin `sm_107`、TI16）需要本项目没有的硬件。

镜像在基础镜像上补了：CMake、Ninja、git、Nsight Compute (`ncu`)、Nsight Systems
(`nsys`)、以及出图用的 python3-matplotlib / numpy（Ubuntu 24.04 的 Python 是
externally-managed，所以走 apt 而不是 pip）。

## 宿主机前提

| 项 | 要求 | 检查 |
|----|------|------|
| NVIDIA 驱动 | **≥ 580.65.06**（CUDA 13.x 最低要求）；低于 580 时见下方 forward-compat 退路 | `nvidia-smi --query-gpu=driver_version --format=csv` |
| NVIDIA Container Toolkit | 已安装 | `docker run --rm --gpus all nvidia/cuda:13.3.1-base-ubuntu24.04 nvidia-smi` |
| GPU | Ampere 或更新 | — |

**驱动是唯一可能直接否掉整套技术选型的前提。** 远程集群常年运行 535/550/570 驱动，
低于 580 时 CUDA 13 二进制默认报 `error 35`。**拿到 root 就升内核驱动；拿不到就走 UMD
forward-compat**：镜像里自带 `/usr/local/cuda-13.x/compat/libcuda.so.*`，把
`LD_LIBRARY_PATH` 指向本机内核模块能配上的那个 compat 目录即可，不动内核模块。哪一个是
能用的（a100 那台是 `cuda-13.0` / UMD 580.178.04，更新的一律 `803`）由
`tools/cuda_compat.py` 实测探测，`preflight.sh` 据此判定 BLOCK 还是 WARN。上机第一件事
就是跑 `preflight.sh`。

## 构建与运行

```bash
docker build -t cuda-kernel-lab:13.3.1 docker/
./docker/run.sh                                  # 交互 shell
./docker/run.sh nvcc --version                   # 跑一条命令就退出
```

`docker/run.sh` 带的几个 flag 都有具体原因：

| flag | 原因 |
|------|------|
| `--gpus all` | 暴露 GPU |
| `--cap-add=SYS_ADMIN` | **ncu 读硬件计数器需要它。**宿主机不允许时会被丢弃，benchmark 照跑，只是没有 profiling |
| `--ipc=host` | 避开默认 64MB shm，大分配会撞上 |
| `--ulimit memlock=-1` | 避免 pinned memory 被 rlimit 卡住 |
| `-v $REPO:/workspace` | 代码与结果落在宿主机，容器退出不丢数据 |

## 首次进入容器：`tools/setup.sh`

进容器第一件事，先把它跑一遍再采集任何数据：

```bash
./tools/setup.sh                # 装缺失的包 → 按 GPU 推导 arch 建 build/ → preflight --matrix
./tools/setup.sh --no-build     # 只装工具 + 校验
SETUP_SKIP_INSTALL=1 ./tools/setup.sh   # 只校验，不碰 apt（镜像已装好时用）
```

幂等，可重复跑。它只补 `dpkg -s` 查出来缺的包，安装清单与 Dockerfile 一致，所以在基础
CUDA 镜像或裸机上也能用；Nsight 装不上只告警（没有计数器证据是被方法论允许的）。
`build/` 的 arch 由 `nvidia-smi` 的 compute capability 推导（8.x→80，9.x→90a，10.x→100，
12.x→120），也可用 `CMAKE_CUDA_ARCHITECTURES` 覆盖。退出码非 0 表示 preflight 有 BLOCK 项。

## 完整复现流程

```bash
docker build -t cuda-kernel-lab:13.3.1 docker/
./docker/run.sh bash -lc '
  ./tools/setup.sh &&                                # 装工具 + 建 build + 预检
  ./build/kernels/01-execution-model/bench-probe &&
  python3 tools/run.py --kernel 00-template --machine h200 &&
  python3 tools/plot.py results/00-template/*.csv -o figures/00-template/
'
```

## ncu 权限

即使加了 `--cap-add=SYS_ADMIN`，宿主机驱动侧仍可能限制计数器采集：

```bash
# 宿主机上检查（需要 root 才能改）
cat /proc/driver/nvidia/params | grep RestrictProfilingToAdminUsers
```

值为 1 时非 root 采不到计数器。拿不到权限就按 `measurement-methodology.md` 的规定，
退到 wall-clock + 带宽利用率推断，并在文章里显式声明本篇无计数器证据。

## MIG

MIG 实例下看不到完整 SM 视图，cluster 行为受限，数据与整卡不可比。H20 尤其常被切分。
容器内 `nvidia-smi -L` 会显示 MIG 设备，确认拿到的是整卡再跑。

## 已知限制

- **镜像未在真机验证过。** 本仓库骨架是在没有 GPU 的机器上搭的，Dockerfile 与 CUDA
  代码都还没实际构建过。首次上机构建失败是预期内的，修完在此处记录。
- Nsight 的安装目录名跟随 Nsight 自己的版本号而非 CUDA 版本号，所以镜像里是构建时
  查找并软链到 `/usr/local/bin`，而不是写死路径。
- 未固定 apt 包版本。要做到完全字节级可复现，需要进一步锁 apt snapshot；当前的取舍是
  digest 锁住基础镜像（CUDA 工具链本体），apt 层允许安全更新。
