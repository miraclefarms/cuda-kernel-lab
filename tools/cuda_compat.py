#!/usr/bin/env python3
"""Detect whether CUDA 13.x can run on this machine's kernel driver.

CUDA 13.x nominally requires a >= 580.65.06 kernel driver. This machine class
(the a100 box) has a 575 kernel module, which makes a CUDA 13.3 binary fail with
error 35. NVIDIA's forward-compatibility packages lift the *user-mode* driver
(UMD) without touching the kernel module, and that is enough: the a100 box runs
CUDA 13.3 through the `cuda-13.0` compat UMD (580.178.04). Newer compat UMDs do
not automatically work with an older kernel module — 13.1/13.2/13.3 (590/595/610)
fail cuInit with 803 — so which compat directory works is an empirical fact that
has to be probed, not assumed.

This script is the single place that answers "is CUDA 13 usable, and through
what?". `tools/preflight.sh` calls it to decide BLOCK vs WARN, and `tools/run.py`
records the chosen compat directory in every CSV so a reader can tell how a run
on a sub-580 driver was possible.

CLI:
    cuda_compat.py                 human summary; exit 0 if usable, 1 if not
    cuda_compat.py --json          machine-readable dict
    cuda_compat.py --shell         shell assignments, safe to eval
    cuda_compat.py --check         silent; exit 0 if usable, 1 if not
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import pathlib
import re
import subprocess
import sys

REQUIRED_KERNEL_MAJOR = 580
COMPAT_GLOBS = ["/usr/local/cuda-*/compat", "/usr/local/cuda/compat"]
# Where a matching native (non-compat) libcuda is normally installed. A container
# image can put its cuda-*/compat directory ahead of these in ld.so.conf, which
# makes a >= 580 kernel driver load an older, incompatible UMD and fail cuInit with
# 803 — even though the correct native UMD is present. tools/run.py and the
# collection scripts prepend the native directory in that case.
NATIVE_SEARCH_DIRS = [
    "/usr/lib/x86_64-linux-gnu",
    "/usr/lib64",
    "/usr/lib",
    "/usr/local/nvidia/lib64",
    "/usr/local/nvidia/lib",
]
_VERSION_RE = re.compile(r"libcuda\.so\.(\d+\.\d+\.\d+)")


def kernel_driver() -> str:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=driver_version",
                              "--format=csv,noheader"],
                             capture_output=True, text=True, check=True).stdout
        return out.splitlines()[0].strip() if out.splitlines() else ""
    except (subprocess.CalledProcessError, FileNotFoundError, IndexError):
        return ""


def _libcuda_version(path: pathlib.Path) -> str:
    versions = []
    for entry in pathlib.Path(path).glob("libcuda.so.*"):
        m = _VERSION_RE.fullmatch(entry.name)
        # Container runtimes often leave zero-byte placeholders for driver libs
        # that were not mounted; they are not loadable UMDs.
        if m and entry.is_file() and entry.stat().st_size > 0:
            versions.append(m.group(1))
    return sorted(versions)[-1] if versions else ""


def native_libcuda_dir(driver: str) -> str:
    """Directory holding a native libcuda matching the kernel driver major."""
    major = driver.split(".")[0] if driver else ""
    if not major:
        return ""
    for directory in NATIVE_SEARCH_DIRS:
        base = pathlib.Path(directory)
        if not base.is_dir():
            continue
        exact = base / f"libcuda.so.{driver}"
        if exact.is_file() and exact.stat().st_size > 0:
            return directory
        for entry in base.glob("libcuda.so.*"):
            m = _VERSION_RE.fullmatch(entry.name)
            if m and entry.stat().st_size > 0 and m.group(1).split(".")[0] == major:
                return directory
    return ""


def _probe_default_libcuda() -> tuple[bool, str]:
    """Try the loader's normal resolution, i.e. the library a run would pick up.

    A plain `CDLL("libcuda.so.1")` respects LD_LIBRARY_PATH and ld.so.conf, so
    this catches a compat UMD that shadows a valid native one.
    """
    snippet = ("import ctypes\n"
               "l=ctypes.CDLL('libcuda.so.1')\n"
               "print(l.cuInit(0))\n")
    try:
        out = subprocess.run([sys.executable, "-c", snippet],
                             capture_output=True, text=True, timeout=30)
    except (subprocess.TimeoutExpired, OSError) as exc:
        return False, str(exc)
    rc = out.stdout.strip()
    if rc == "0":
        return True, "cuInit=0"
    return False, f"cuInit={rc or 'n/a'}" + (f" ({out.stderr.strip()})" if out.stderr.strip() else "")


def _probe_dir(path: pathlib.Path) -> tuple[bool, str]:
    """Try to load this compat libcuda and cuInit. Returns (ok, detail)."""
    lib = pathlib.Path(path) / "libcuda.so.1"
    if not lib.exists():
        return False, "no libcuda.so.1"
    # dlopen must happen in a throwaway process: cuInit can only run once per
    # process, and this script may itself hold a context from another probe.
    snippet = (
        "import ctypes,sys\n"
        f"l=ctypes.CDLL({str(lib)!r})\n"
        "rc=l.cuInit(0)\n"
        "print(rc)\n"
    )
    try:
        out = subprocess.run([sys.executable, "-c", snippet],
                             capture_output=True, text=True, timeout=30)
    except (subprocess.TimeoutExpired, OSError) as exc:
        return False, str(exc)
    rc = out.stdout.strip()
    if rc == "0":
        return True, "cuInit=0"
    return False, f"cuInit={rc or 'n/a'}" + (f" ({out.stderr.strip()})" if out.stderr.strip() else "")


def candidate_compat_dirs() -> list[pathlib.Path]:
    dirs: dict[str, pathlib.Path] = {}
    for pattern in COMPAT_GLOBS:
        for d in glob.glob(pattern):
            p = pathlib.Path(d)
            if not p.is_dir():
                continue
            key = str(p.resolve())
            # Prefer the explicitly versioned name (cuda-13.3/compat) over the
            # /usr/local/cuda/compat symlink when both point at the same dir.
            if key not in dirs or (not _cuda_in_name(dirs[key]) and _cuda_in_name(p)):
                dirs[key] = p
    def version_key(p: pathlib.Path) -> tuple:
        m = re.search(r"cuda-(\d+(?:\.\d+)?)", str(p))
        return tuple(int(x) for x in m.group(1).split(".")) if m else (0,)
    return sorted(dirs.values(), key=version_key, reverse=True)


def _cuda_in_name(p: pathlib.Path) -> bool:
    return re.search(r"cuda-\d", str(p)) is not None


def detect() -> dict:
    drv = kernel_driver()
    major = int(drv.split(".")[0]) if drv.split(".")[0].isdigit() else 0
    needed = major < REQUIRED_KERNEL_MAJOR
    result = {
        "kernel_driver": drv,
        "required_major": REQUIRED_KERNEL_MAJOR,
        "needed": needed,
        "usable": not needed,
        "compat_dir": "",
        "compat_umd": "",
        "prepend_dir": "",
        "reason": "kernel driver >= 580; no compat needed" if not needed else "",
        "probed": [],
    }
    if not needed:
        # The kernel driver is new enough, but the loader may still resolve an
        # incompatible compat UMD first (ld.so.conf ordering in some images).
        # Verify the library a run actually picks up before declaring victory.
        ok, detail = _probe_default_libcuda()
        if ok:
            return result
        native = native_libcuda_dir(drv)
        if native:
            native_ok, native_detail = _probe_dir(pathlib.Path(native))
            result["probed"].append({"dir": native, "umd": _libcuda_version(pathlib.Path(native)),
                                     "ok": native_ok, "detail": native_detail})
            if native_ok:
                result.update(prepend_dir=native,
                              reason=f"native UMD in {native} is shadowed on the loader "
                                     f"path (default {detail}); prepend it")
                return result
        result.update(usable=False,
                      reason=f"driver {drv} >= 580 but no working libcuda on the loader "
                             f"path ({detail})")
        return result

    for d in candidate_compat_dirs():
        ok, detail = _probe_dir(d)
        result["probed"].append({"dir": str(d), "umd": _libcuda_version(d),
                                 "ok": ok, "detail": detail})
        if ok:
            result.update(usable=True, compat_dir=str(d),
                          compat_umd=_libcuda_version(d),
                          reason=f"forward-compat UMD {_libcuda_version(d)} works on "
                                 f"kernel {drv}")
            return result
    result["reason"] = (f"kernel {drv} < {REQUIRED_KERNEL_MAJOR} and no compat dir "
                        f"passed cuInit")
    return result


def find_in_ld_library_path() -> tuple[str, str]:
    """The compat dir a running process would pick up, if any.
    Used by tools/run.py to record what it actually executed against. Native
    (non-compat) libcuda directories are not forward-compat UMDs and must not be
    recorded in the `cuda_compat` column.
    """
    for entry in os.environ.get("LD_LIBRARY_PATH", "").split(":"):
        if not entry:
            continue
        p = pathlib.Path(entry)
        if "compat" in str(p) and (p / "libcuda.so.1").exists():
            return str(p), _libcuda_version(p)
    return "", ""


def _shell_quote(value: str) -> str:
    return "'" + value.replace("'", "'\\''") + "'"


def loader_prepend_dir() -> str:
    """Directory to prepend to LD_LIBRARY_PATH so a run loads a working libcuda.

    Empty when the loader already resolves a compatible UMD. This is the hook
    tools/run.py uses to survive images whose ld.so.conf shadows the native UMD.
    """
    try:
        return detect().get("prepend_dir", "")
    except Exception:
        return ""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--json", action="store_true")
    g.add_argument("--shell", action="store_true")
    g.add_argument("--check", action="store_true")
    g.add_argument("--ld-path", action="store_true",
                   help="print a directory to prepend to LD_LIBRARY_PATH (empty if none)")
    args = ap.parse_args()

    info = detect()

    if args.check:
        return 0 if info["usable"] else 1
    if args.ld_path:
        print(info["prepend_dir"])
        return 0 if info["usable"] else 1
    if args.json:
        print(json.dumps(info))
        return 0 if info["usable"] else 1
    if args.shell:
        print(f"COMPAT_NEEDED={1 if info['needed'] else 0}")
        print(f"COMPAT_USABLE={1 if info['usable'] else 0}")
        print(f"COMPAT_DIR={_shell_quote(info['compat_dir'])}")
        print(f"COMPAT_UMD={_shell_quote(info['compat_umd'])}")
        print(f"COMPAT_LD_PREPEND={_shell_quote(info['prepend_dir'])}")
        print(f"COMPAT_REASON={_shell_quote(info['reason'])}")
        return 0 if info["usable"] else 1

    if info["usable"] and not info["needed"]:
        if info["prepend_dir"]:
            print(f"ok: kernel driver {info['kernel_driver']} supports CUDA 13.x, but the "
                  f"loader picks a shadowing UMD")
            print(f"    export LD_LIBRARY_PATH={info['prepend_dir']}:${{LD_LIBRARY_PATH:-}}")
        else:
            print(f"ok: kernel driver {info['kernel_driver']} supports CUDA 13.x")
    elif info["usable"]:
        print(f"ok: {info['reason']}")
        print(f"    export LD_LIBRARY_PATH={info['compat_dir']}:${{LD_LIBRARY_PATH:-}}")
    else:
        print(f"BLOCK: {info['reason']}")
        for p in info["probed"]:
            print(f"    probed {p['dir']} (UMD {p['umd']}): {p['detail']}")
    return 0 if info["usable"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
