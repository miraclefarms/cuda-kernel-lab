#!/usr/bin/env python3
"""Run a command only inside a verified GPU-quiet window.

The measurement discipline says a benchmark's target card must be idle: another
process time-slicing the GPU corrupts the timing, and a resident-but-idle context
does not. This gate enforces that around an arbitrary command:

    PRE   poll utilization until one (or --need) candidate card(s) stay at
          util == 0 for a full quiet window
    RUN   launch the command with CUDA_VISIBLE_DEVICES pinned to the chosen
          card(s); no sampling while it runs
    POST  after the command exits, require the chosen card(s) to stay quiet for
          another window. If they do not, a co-tenant was on the GPU and the run
          is marked invalid.

The POST window is what catches a co-tenant that was never visible in PRE: a
training job that keeps running after our task exits shows up as util > 0. The
honest limitation is that a co-tenant which starts and stops entirely inside the
RUN window is not observable, because we deliberately do not sample during RUN.
That caveat is emitted in the report rather than papered over.

Exit codes:
    0  pass
    1  task itself failed (GPU window was clean)
    2  POST window not quiet — co-tenant detected, data invalid
    3  PRE timed out with no quiet window
    4  environment / tool error (nvidia-smi unusable, --need unsatisfiable)

stdout carries exactly one JSON verdict line; the task's own stdout is forwarded
to stderr, along with progress, so stdout can be piped straight into a parser.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import signal
import subprocess
import sys
import time

SAMPLE_QUERY = ["nvidia-smi", "--query-gpu=index,utilization.gpu",
                "--format=csv,noheader,nounits"]
SAMPLE_RETRIES = 3
SAMPLE_RETRY_WAIT_S = 1.0

CAVEAT = ("no sampling during RUN: a co-tenant that starts and stops entirely "
          "inside the task window is not observable; PRE+POST only catch "
          "persistent co-tenants that outlive the task")

EXIT_PASS = 0
EXIT_TASK_FAILED = 1
EXIT_POST_FAILED = 2
EXIT_PRE_TIMEOUT = 3
EXIT_ENV_ERROR = 4


class EnvError(RuntimeError):
    """nvidia-smi is unusable or the requested cards cannot be honoured."""


def sample_utilization() -> dict[int, int]:
    """Return {physical_index: utilization.gpu} for every visible GPU.

    A single failed call is retried; an unparseable or [N/A] value is an error,
    never silently treated as idle. Reading a broken sample as 0 would let a
    corrupted environment through the gate, which is the one thing it exists to
    prevent.
    """
    last_err = "no attempt made"
    for _ in range(SAMPLE_RETRIES):
        try:
            out = subprocess.run(SAMPLE_QUERY, capture_output=True, text=True,
                                 check=True).stdout
        except (subprocess.CalledProcessError, FileNotFoundError) as exc:
            last_err = f"nvidia-smi failed: {exc}"
            time.sleep(SAMPLE_RETRY_WAIT_S)
            continue

        utils: dict[int, int] = {}
        parse_err = ""
        for line in out.splitlines():
            line = line.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 2 or not parts[0].isdigit():
                parse_err = f"unparseable nvidia-smi row: {line!r}"
                break
            if not parts[1].isdigit():
                parse_err = (f"GPU {parts[0]} utilization is {parts[1]!r}, not a "
                             f"number — [N/A] means the driver can't report it")
                break
            utils[int(parts[0])] = int(parts[1])
        if parse_err:
            last_err = parse_err
            time.sleep(SAMPLE_RETRY_WAIT_S)
            continue
        if not utils:
            last_err = "nvidia-smi returned no GPU rows"
            time.sleep(SAMPLE_RETRY_WAIT_S)
            continue
        return utils

    raise EnvError(last_err)


class QuietTracker:
    """Tracks how long each card has been continuously at util <= tolerance.

    Qualification is measured in elapsed time, not sample count: sleep jitter
    (and nvidia-smi latency) makes "6 samples at 10 s" a 50 s window on a fast
    machine, which would silently under-report the quiet span the caller asked
    for.
    """

    def __init__(self, indices: list[int], tolerance: int) -> None:
        self.tolerance = tolerance
        self.zero_since: dict[int, float | None] = {i: None for i in indices}

    def update(self, utils: dict[int, int], now: float) -> None:
        for idx in self.zero_since:
            util = utils.get(idx)
            if util is None or util > self.tolerance:
                self.zero_since[idx] = None
            elif self.zero_since[idx] is None:
                self.zero_since[idx] = now

    def qualifying(self, now: float, window_s: float, need: int) -> list[int]:
        """The `need` cards quiet the longest, or [] if fewer than `need` qualify."""
        ready = [(since, idx) for idx, since in self.zero_since.items()
                 if since is not None and now - since >= window_s]
        if len(ready) < need:
            return []
        ready.sort()
        return [idx for _, idx in ready[:need]]

    def span(self, idx: int, now: float) -> float:
        since = self.zero_since.get(idx)
        return 0.0 if since is None else now - since


def parse_candidates(raw: str | None) -> list[int] | None:
    if not raw:
        return None
    return [int(p.strip()) for p in raw.split(",") if p.strip()]


def launcher_env(selected: list[int], set_cvd: bool) -> dict[str, str]:
    env = os.environ.copy()
    if set_cvd:
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in selected)
    return env


def mark_invalid(csv_path: str, reason: str) -> str:
    """Drop the `.invalid` marker the methodology asks for next to a bad CSV."""
    marker = pathlib.Path(csv_path + ".invalid")
    marker.write_text(
        f"invalid: {reason}\n"
        f"marked_at: {time.strftime('%Y-%m-%dT%H:%M:%S%z')}\n"
        f"marked_by: tools/gpu_quiet_gate.py\n")
    return str(marker)


def run_pre(args: argparse.Namespace) -> tuple[list[int] | None, list[dict]]:
    first = sample_utilization()
    if args.candidates:
        missing = [i for i in args.candidates if i not in first]
        if missing:
            raise EnvError(f"candidate GPU(s) not visible: {missing}")
        candidates = args.candidates
    else:
        candidates = sorted(first)

    tracker = QuietTracker(candidates, args.tolerance)
    log: list[dict] = []
    start = time.monotonic()
    while True:
        now = time.monotonic()
        try:
            utils = sample_utilization()
        except EnvError as exc:
            raise EnvError(f"during PRE: {exc}")

        tracker.update(utils, now)
        log.append({"t_s": round(now - start, 2),
                    "utils": {str(k): v for k, v in sorted(utils.items())}})

        selected = tracker.qualifying(now, args.quiet_window, args.need)
        if selected:
            print(f"[gate] PRE quiet: GPU {selected} idle for "
                  f"{tracker.span(selected[0], now):.1f}s "
                  f"(need>= {args.quiet_window:.0f}s) — launching", file=sys.stderr)
            return selected, log

        busy = {str(i): utils.get(i) for i in candidates}
        print(f"[gate] PRE waiting: need {args.need} card(s) idle >= "
              f"{args.quiet_window:.0f}s; candidate utils {busy}", file=sys.stderr)

        if now - start >= args.max_wait:
            print(f"[gate] PRE timed out after {args.max_wait:.0f}s — no quiet "
                  f"window; not launching", file=sys.stderr)
            return None, log

        time.sleep(args.interval)


def run_post(args: argparse.Namespace,
             selected: list[int]) -> tuple[bool, list[dict], float]:
    tracker = QuietTracker(selected, args.tolerance)
    log: list[dict] = []
    start = time.monotonic()
    timeout = max(args.post_timeout, args.post_window * 2)
    while True:
        now = time.monotonic()
        try:
            utils = sample_utilization()
        except EnvError as exc:
            raise EnvError(f"during POST: {exc}")

        tracker.update(utils, now)
        log.append({"t_s": round(now - start, 2),
                    "utils": {str(k): v for k, v in sorted(utils.items())}})

        busy = {i: utils.get(i) for i in selected if (utils.get(i) or 0) > args.tolerance}
        if busy:
            print(f"[gate] POST FAIL: GPU(s) in use after task: {busy} — a "
                  f"co-tenant shared the GPU, data invalid", file=sys.stderr)
            return False, log, tracker.span(selected[0], now)

        if all(tracker.span(i, now) >= args.post_window for i in selected):
            print(f"[gate] POST quiet: GPU {selected} idle "
                  f"{tracker.span(selected[0], now):.1f}s "
                  f"(need>= {args.post_window:.0f}s)", file=sys.stderr)
            return True, log, tracker.span(selected[0], now)

        print(f"[gate] POST waiting: GPU {selected} at "
              f"{[tracker.span(i, now) for i in selected]}s quiet of "
              f"{args.post_window:.0f}s", file=sys.stderr)

        if now - start >= timeout:
            print(f"[gate] POST FAIL: quiet window not reached within "
                  f"{timeout:.0f}s", file=sys.stderr)
            return False, log, tracker.span(selected[0], now)

        time.sleep(args.interval)


def emit(args: argparse.Namespace, report: dict) -> None:
    line = json.dumps(report)
    print(line)
    if args.report:
        pathlib.Path(args.report).write_text(json.dumps(report, indent=2) + "\n")


def base_report(args: argparse.Namespace, cmd: list[str]) -> dict:
    return {
        "verdict": "error",
        "reason": "",
        "exit_code": EXIT_ENV_ERROR,
        "command": cmd,
        "interval_s": args.interval,
        "quiet_window_s": args.quiet_window,
        "post_window_s": args.post_window,
        "max_wait_s": args.max_wait,
        "tolerance_pct": args.tolerance,
        "need": args.need,
        "candidates": args.candidates,
        "selected_gpus": [],
        "pre": None,
        "post": None,
        "task_returncode": None,
        "caveat": CAVEAT,
    }


def run(args: argparse.Namespace, cmd: list[str]) -> tuple[dict, int]:
    report = base_report(args, cmd)

    selected, pre_log = run_pre(args)
    report["pre"] = summary(pre_log, selected)
    if selected is None:
        report.update(verdict="fail", reason="PRE timed out: no quiet window",
                      exit_code=EXIT_PRE_TIMEOUT)
        return report, EXIT_PRE_TIMEOUT
    report["selected_gpus"] = selected

    env = launcher_env(selected, not args.no_set_cvd)
    print(f"[gate] RUN: CUDA_VISIBLE_DEVICES={env.get('CUDA_VISIBLE_DEVICES', '<inherited>')} "
          f"cmd={' '.join(cmd)}", file=sys.stderr)
    # The task's stdout is forwarded to our stderr so that this tool's stdout
    # stays exactly one JSON verdict line, safe to pipe into a parser.
    proc = subprocess.Popen(cmd, env=env, start_new_session=True, stdout=sys.stderr)
    try:
        task_rc = proc.wait()
    except KeyboardInterrupt:
        print("[gate] interrupted — terminating task", file=sys.stderr)
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        task_rc = proc.wait()
        report.update(verdict="error", reason="interrupted by signal",
                      exit_code=EXIT_ENV_ERROR)
        report["task_returncode"] = task_rc
        return report, EXIT_ENV_ERROR
    report["task_returncode"] = task_rc
    print(f"[gate] RUN done: task exit {task_rc}", file=sys.stderr)

    quiet, post_log, _ = run_post(args, selected)
    report["post"] = summary(post_log, selected)

    if not quiet:
        report.update(verdict="fail",
                      reason="POST not quiet: a co-tenant used the GPU during/after the task",
                      exit_code=EXIT_POST_FAILED)
        if args.csv:
            marker = mark_invalid(args.csv, report["reason"])
            report["invalid_marker"] = marker
            print(f"[gate] wrote invalid marker: {marker}", file=sys.stderr)
        return report, EXIT_POST_FAILED

    report.update(verdict="pass", reason="PRE and POST windows quiet",
                  exit_code=EXIT_PASS)
    if task_rc != 0:
        report.update(verdict="fail", reason=f"task exited {task_rc}",
                      exit_code=EXIT_TASK_FAILED)
        return report, EXIT_TASK_FAILED
    return report, EXIT_PASS


def summary(log: list[dict], selected: list[int] | None) -> dict:
    if selected is None:
        indices = sorted({int(k) for entry in log for k in entry["utils"]})
    else:
        indices = selected
    max_util = 0
    for entry in log:
        for idx in indices:
            max_util = max(max_util, entry["utils"].get(str(idx), 0))
    return {
        "samples": len(log),
        "span_s": log[-1]["t_s"] if log else 0.0,
        "max_util_pct": max_util,
        "selected_gpus": selected or [],
    }


def self_test() -> int:
    """Exercise the window logic without a GPU."""
    def utils(idx: int, val: int) -> dict[int, int]:
        return {idx: val}

    failures: list[str] = []

    # 1. steady idle reaches the window
    t = QuietTracker([0], tolerance=0)
    for s in range(0, 71):
        t.update(utils(0, 0), float(s))
    if t.qualifying(70.0, 60, 1) != [0]:
        failures.append("steady idle did not qualify")

    # 2. a single busy sample resets the run
    t = QuietTracker([0], tolerance=0)
    for s in range(0, 61):
        t.update(utils(0, 0), float(s))
    t.update(utils(0, 1), 61.0)
    if t.qualifying(121.0, 60, 1) != []:
        failures.append("busy spike did not reset the quiet run")

    # 3. need=2 requires two cards, longest-quiet first
    t = QuietTracker([0, 1], tolerance=0)
    for s in range(0, 71):
        t.update({0: 0, 1: 5}, float(s))
    if t.qualifying(70.0, 60, 2) != []:
        failures.append("need=2 qualified with only one quiet card")
    if t.qualifying(70.0, 60, 1) != [0]:
        failures.append("need=1 did not pick the quiet card")
    t2 = QuietTracker([0, 1], tolerance=0)
    t2.update({0: 0, 1: 5}, 0.0)
    t2.update({0: 0, 1: 0}, 5.0)
    for s in range(6, 71):
        t2.update({0: 0, 1: 0}, float(s))
    if t2.qualifying(70.0, 60, 2) != [0, 1]:
        failures.append("need=2 did not return both cards")

    # 4. tolerance lets a small non-zero pass
    t = QuietTracker([0], tolerance=2)
    for s in range(0, 71):
        t.update(utils(0, 2), float(s))
    if t.qualifying(70.0, 60, 1) != [0]:
        failures.append("tolerance did not admit util<=2")

    # 5. a missing card is treated as busy, never idle
    t = QuietTracker([0, 1], tolerance=0)
    for s in range(0, 71):
        t.update({0: 0}, float(s))
    if t.qualifying(70.0, 60, 1) != [0]:
        failures.append("missing card should stay unqualified")
    if 1 in t.zero_since and t.zero_since[1] is not None:
        failures.append("missing card was tracked as quiet")

    # 6. time-based window: 7 samples at 10s is 60s, not 50s
    t = QuietTracker([0], tolerance=0)
    for k in range(0, 7):
        t.update(utils(0, 0), float(k * 10))
    if t.qualifying(60.0, 60, 1) != [0]:
        failures.append("a full 60s span should qualify")
    t = QuietTracker([0], tolerance=0)
    for k in range(0, 6):
        t.update(utils(0, 0), float(k * 10))
    if t.qualifying(50.0, 60, 1) != []:
        failures.append("a 50s span must not qualify for a 60s window")

    if failures:
        for f in failures:
            print(f"[self-test] FAIL: {f}", file=sys.stderr)
        return 1
    print("[self-test] ok")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--need", type=int, default=1,
                    help="cards that must be simultaneously quiet (default 1)")
    ap.add_argument("--candidates", default=None,
                    help="restrict candidate physical GPU indices, e.g. 0,2,3")
    ap.add_argument("--interval", type=float, default=10.0,
                    help="sampling period in seconds (default 10)")
    ap.add_argument("--quiet-window", dest="quiet_window", type=float, default=60.0,
                    help="required PRE quiet span in seconds (default 60)")
    ap.add_argument("--post-window", dest="post_window", type=float, default=30.0,
                    help="required POST quiet span in seconds (default 30)")
    ap.add_argument("--max-wait", dest="max_wait", type=float, default=1800.0,
                    help="give up on PRE after this many seconds (default 1800)")
    ap.add_argument("--post-timeout", dest="post_timeout", type=float, default=300.0,
                    help="give up on POST after this many seconds (default 300)")
    ap.add_argument("--tolerance", type=int, default=0,
                    help="largest utilization.gpu still counted as idle (default 0)")
    ap.add_argument("--csv", default=None,
                    help="on POST failure, create <CSV>.invalid next to this file")
    ap.add_argument("--report", default=None, help="write the full JSON report here")
    ap.add_argument("--no-set-cvd", dest="no_set_cvd", action="store_true",
                    help="do not overwrite CUDA_VISIBLE_DEVICES for the task")
    ap.add_argument("--self-test", dest="self_test", action="store_true",
                    help="exercise the quiet-window logic without a GPU")
    ap.add_argument("rest", nargs=argparse.REMAINDER,
                    help="-- <command> [args...]")
    args = ap.parse_args()

    if args.self_test:
        return self_test()

    cmd = args.rest[1:] if args.rest and args.rest[0] == "--" else args.rest
    if not cmd:
        ap.error("no command given; use: gpu_quiet_gate.py [opts] -- <command>")
    if args.need < 1:
        ap.error("--need must be >= 1")
    if args.interval <= 0:
        ap.error("--interval must be > 0")
    args.candidates = parse_candidates(args.candidates)

    def _sigterm(_signum: int, _frame: object) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _sigterm)

    report = base_report(args, cmd)
    try:
        report, code = run(args, cmd)
    except EnvError as exc:
        report.update(verdict="error", reason=str(exc), exit_code=EXIT_ENV_ERROR)
        code = EXIT_ENV_ERROR
        print(f"[gate] ENV ERROR: {exc}", file=sys.stderr)
    except KeyboardInterrupt:
        report.update(verdict="error", reason="interrupted", exit_code=EXIT_ENV_ERROR)
        code = EXIT_ENV_ERROR
    emit(args, report)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
