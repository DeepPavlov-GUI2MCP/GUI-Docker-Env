#!/usr/bin/env python3
"""htop-style monitor for Holo OSWorld eval runs."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
from urllib.parse import urlparse

DEFAULT_ROOT = Path("/home/pitchblack/dart-gui/GUI-Docker-Env")
DEFAULT_VLLM_URLS = "http://127.0.0.1:8010,http://127.0.0.1:8011"
DEFAULT_DESKTOP_HOSTS = (
    "dart-desktop-2=http://94.139.251.45:50003,"
    "coordinator=http://127.0.0.1:50003"
)
DEFAULT_HOST_MAP = (
    "94.139.251.45:50003=dart-desktop-2,"
    "127.0.0.1:50003=coordinator"
)

METRIC_PATTERNS = {
    "kv_cache_usage_perc": re.compile(r"^vllm:kv_cache_usage_perc\{.*?\}\s+([0-9.eE+-]+)$"),
    "num_requests_running": re.compile(r"^vllm:num_requests_running\{.*?\}\s+([0-9.eE+-]+)$"),
    "num_requests_waiting": re.compile(r"^vllm:num_requests_waiting\{.*?\}\s+([0-9.eE+-]+)$"),
}


DEFAULT_RATE_WINDOW_SECONDS = 300


@dataclass
class RateTracker:
    window_seconds: float
    samples: deque[Tuple[float, int]] = field(default_factory=deque)

    def record(self, finished: int, now: Optional[float] = None) -> None:
        timestamp = time.time() if now is None else now
        self.samples.append((timestamp, finished))
        cutoff = timestamp - self.window_seconds
        while self.samples and self.samples[0][0] < cutoff:
            self.samples.popleft()

    def rate_per_hour(self) -> Optional[float]:
        if len(self.samples) < 2:
            return None
        start_time, start_finished = self.samples[0]
        end_time, end_finished = self.samples[-1]
        elapsed = end_time - start_time
        if elapsed <= 0:
            return None
        delta = end_finished - start_finished
        if delta < 0:
            return None
        return delta / elapsed * 3600.0

    def rate_per_minute(self) -> Optional[float]:
        rate_hour = self.rate_per_hour()
        if rate_hour is None:
            return None
        return rate_hour / 60.0

    def window_label(self) -> str:
        if len(self.samples) < 2:
            return f"{int(self.window_seconds // 60)}m"
        covered = self.samples[-1][0] - self.samples[0][0]
        if covered >= self.window_seconds * 0.95:
            return f"{int(self.window_seconds // 60)}m"
        return f"{int(covered)}s"


@dataclass
class DesktopHost:
    name: str
    base_url: str
    ssh_host: Optional[str] = None


@dataclass
class VllmGpuStats:
    label: str
    url: str
    ok: bool
    kv_cache_pct: float = 0.0
    running: int = 0
    waiting: int = 0
    error: str = ""


@dataclass
class HostResourceStats:
    name: str
    ok: bool
    cpu_pct: float = 0.0
    mem_used_gib: float = 0.0
    mem_total_gib: float = 0.0
    load1: float = 0.0
    gpu_mem_used_gib: List[float] = field(default_factory=list)
    gpu_mem_total_gib: List[float] = field(default_factory=list)
    gpu_util_pct: List[float] = field(default_factory=list)
    error: str = ""


@dataclass
class TaskBucket:
    finished: int = 0
    failed: int = 0
    in_progress: int = 0
    pending: int = 0


@dataclass
class DesktopServerStats:
    name: str
    ok: bool
    emulators: int = 0
    in_use: int = 0
    pending: int = 0
    error: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monitor Holo eval: vLLM KV, host load, task progress")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--eval-json", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path)
    parser.add_argument("--results-glob", type=str)
    parser.add_argument(
        "--all-matches",
        action="store_true",
        help="Monitor every directory matched by --results-glob instead of only the latest one",
    )
    parser.add_argument("--refresh-seconds", type=float, default=2.0)
    parser.add_argument("--title", type=str, default="Holo Eval Monitor")
    parser.add_argument("--vllm-urls", type=str, default=DEFAULT_VLLM_URLS)
    parser.add_argument("--desktop-hosts", type=str, default=DEFAULT_DESKTOP_HOSTS)
    parser.add_argument("--host-map", type=str, default=DEFAULT_HOST_MAP)
    parser.add_argument("--rollouter-ssh", type=str, default="dart-rollouter")
    parser.add_argument(
        "--no-rollouter-ssh",
        action="store_true",
        help="Skip SSH CPU/RAM/GPU stats for dart-rollouter",
    )
    parser.add_argument(
        "--rate-window-seconds",
        type=float,
        default=DEFAULT_RATE_WINDOW_SECONDS,
        help="Rolling window for finished-task rate (default: 300 = 5 minutes)",
    )
    parser.add_argument(
        "--model-subdir",
        type=str,
        default="pyautogui/screenshot/Hcompany/Holotron-3-Nano",
        help="Path under results dir to domain/task folders",
    )
    return parser.parse_args()


def parse_name_value_pairs(raw: str) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk or "=" not in chunk:
            continue
        name, value = chunk.split("=", 1)
        mapping[name.strip()] = value.strip()
    return mapping


def parse_desktop_hosts(raw: str) -> List[DesktopHost]:
    hosts: List[DesktopHost] = []
    for name, base_url in parse_name_value_pairs(raw).items():
        ssh_host = None if name == "coordinator" else name
        hosts.append(DesktopHost(name=name, base_url=base_url.rstrip("/"), ssh_host=ssh_host))
    return hosts


def load_tasks(eval_json: Path) -> List[Tuple[str, str]]:
    payload = json.loads(eval_json.read_text(encoding="utf-8"))
    tasks: List[Tuple[str, str]] = []
    for domain in sorted(payload):
        for task_id in payload[domain]:
            tasks.append((domain, task_id))
    return tasks


def resolve_result_dirs(
    root: Path,
    results_dir: Path | None,
    results_glob: str | None,
    all_matches: bool,
) -> List[Path]:
    if results_dir is not None:
        candidate = results_dir if results_dir.is_absolute() else root / results_dir
        return [candidate.resolve()] if candidate.is_dir() else []
    assert results_glob is not None
    matches = [path.resolve() for path in root.glob(results_glob) if path.is_dir()]
    if not matches:
        return []
    matches.sort(key=lambda path: (path.stat().st_mtime, path.name))
    return matches if all_matches else [matches[-1]]


def task_root(result_dir: Path, model_subdir: str, domain: str, task_id: str) -> Path:
    direct = result_dir / model_subdir / domain / task_id
    if direct.is_dir():
        return direct
    matches = sorted(result_dir.glob(f"**/{domain}/{task_id}"))
    return matches[0] if matches else direct


def read_host_key(task_dir: Path) -> str:
    host_path = task_dir / "host.txt"
    if host_path.is_file():
        return host_path.read_text(encoding="utf-8").strip()
    return ""


def classify_task(task_dir: Path) -> str:
    if not task_dir.is_dir():
        return "pending"
    listing = set(os.listdir(task_dir))
    if "result.txt" in listing:
        return "finished"
    if "failed.txt" in listing:
        return "failed"
    return "in_progress"


def collect_task_stats(
    tasks: Iterable[Tuple[str, str]],
    result_dir: Path,
    model_subdir: str,
    host_map: Dict[str, str],
) -> Tuple[TaskBucket, Dict[str, TaskBucket], Dict[str, str]]:
    total = TaskBucket()
    per_host: Dict[str, TaskBucket] = defaultdict(TaskBucket)
    host_keys: Dict[str, str] = {}

    for domain, task_id in tasks:
        task_dir = task_root(result_dir, model_subdir, domain, task_id)
        state = classify_task(task_dir)
        host_key = read_host_key(task_dir)
        host_name = host_map.get(host_key, "unknown" if host_key else "unassigned")
        host_keys[f"{domain}/{task_id}"] = host_name

        bucket = total
        setattr(bucket, state, getattr(bucket, state) + 1)
        host_bucket = per_host[host_name]
        setattr(host_bucket, state, getattr(host_bucket, state) + 1)

    return total, dict(per_host), host_keys


def http_get(url: str, timeout: float = 3.0) -> str:
    request = urllib.request.Request(url, headers={"Accept": "text/plain"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8", errors="replace")


def parse_metric(text: str, metric_name: str) -> float:
    pattern = METRIC_PATTERNS[metric_name]
    for line in text.splitlines():
        match = pattern.match(line.strip())
        if match:
            return float(match.group(1))
    return 0.0


def collect_vllm_stats(urls: List[str]) -> List[VllmGpuStats]:
    stats: List[VllmGpuStats] = []
    for index, base in enumerate(urls):
        label = urlparse(base).port or base
        metrics_url = f"{base.rstrip('/')}/metrics"
        try:
            body = http_get(metrics_url)
            stats.append(
                VllmGpuStats(
                    label=f"gpu{index}",
                    url=base,
                    ok=True,
                    kv_cache_pct=100.0 * parse_metric(body, "kv_cache_usage_perc"),
                    running=int(parse_metric(body, "num_requests_running")),
                    waiting=int(parse_metric(body, "num_requests_waiting")),
                )
            )
        except (urllib.error.URLError, TimeoutError, ValueError) as exc:
            stats.append(VllmGpuStats(label=f"gpu{index}", url=base, ok=False, error=str(exc)))
    return stats


def run_command(command: List[str], timeout: float = 5.0) -> Tuple[int, str, str]:
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        return completed.returncode, completed.stdout.strip(), completed.stderr.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
        return 1, "", str(exc)


def parse_free_mem_gib(text: str) -> Tuple[float, float]:
    for line in text.splitlines():
        if not line.startswith("Mem:"):
            continue
        parts = line.split()
        if len(parts) < 3:
            continue
        total = _mem_value_to_gib(parts[1])
        used = _mem_value_to_gib(parts[2])
        return used, total
    return 0.0, 0.0


def _mem_value_to_gib(raw: str) -> float:
    normalized = raw.strip().replace(",", ".")
    match = re.match(r"^([\d.]+)\s*([KMGTP]?)i?B?$", normalized, re.IGNORECASE)
    if not match:
        return 0.0
    value = float(match.group(1))
    unit = (match.group(2) or "B").upper()
    multipliers = {
        "K": 1 / (1024**2),
        "M": 1 / 1024,
        "G": 1.0,
        "T": 1024.0,
        "P": 1024.0**2,
        "B": 1 / (1024**3),
    }
    return value * multipliers.get(unit, 1.0)


def parse_loadavg(text: str) -> float:
    for line in text.splitlines():
        parts = line.split()
        if len(parts) >= 3 and all(part.replace(".", "", 1).isdigit() for part in parts[:3]):
            return float(parts[0].replace(",", "."))
    return 0.0


def parse_cpu_pct_from_top(text: str) -> float:
    for line in text.splitlines():
        if line.startswith("%Cpu(s):"):
            parts = [part.strip() for part in line.split(",")]
            for part in parts:
                if part.endswith(" id"):
                    idle = float(part.split()[0])
                    return max(0.0, min(100.0, 100.0 - idle))
    return 0.0


def collect_local_resources(name: str) -> HostResourceStats:
    _, free_out, err = run_command(["free", "-h"])
    if err and not free_out:
        return HostResourceStats(name=name, ok=False, error=err)
    _, load_out, _ = run_command(["cat", "/proc/loadavg"])
    cpu_pct = 0.0
    if shutil.which("top"):
        _, top_out, _ = run_command(["top", "-bn1"], timeout=3.0)
        cpu_pct = parse_cpu_pct_from_top(top_out)
    used, total = parse_free_mem_gib(free_out)
    return HostResourceStats(
        name=name,
        ok=True,
        cpu_pct=cpu_pct,
        mem_used_gib=used,
        mem_total_gib=total,
        load1=parse_loadavg(load_out),
    )


def collect_ssh_resources(name: str, ssh_host: str, include_gpu: bool = False) -> HostResourceStats:
    remote_script = "free -h | head -2; cat /proc/loadavg; top -bn1 | head -3"
    if include_gpu:
        remote_script += "; nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits"
    code, out, err = run_command(["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5", ssh_host, remote_script])
    if code != 0 or not out:
        return HostResourceStats(name=name, ok=False, error=err or f"ssh exit {code}")

    chunks = out.split(";")
    body = chunks[0]
    used, total = parse_free_mem_gib(body)
    load1 = parse_loadavg(body)
    cpu_pct = parse_cpu_pct_from_top(body)

    gpu_used: List[float] = []
    gpu_total: List[float] = []
    gpu_util: List[float] = []
    if include_gpu and len(chunks) > 1:
        for line in chunks[1].splitlines():
            parts = [part.strip() for part in line.split(",") if part.strip()]
            if len(parts) != 3:
                continue
            gpu_used.append(float(parts[0]) / 1024.0)
            gpu_total.append(float(parts[1]) / 1024.0)
            gpu_util.append(float(parts[2]))

    return HostResourceStats(
        name=name,
        ok=True,
        cpu_pct=cpu_pct,
        mem_used_gib=used,
        mem_total_gib=total,
        load1=load1,
        gpu_mem_used_gib=gpu_used,
        gpu_mem_total_gib=gpu_total,
        gpu_util_pct=gpu_util,
    )


def collect_desktop_server(host: DesktopHost) -> DesktopServerStats:
    try:
        body = http_get(f"{host.base_url}/status")
        payload = json.loads(body)
        token_info = next(item for item in payload.get("tokens", []) if item.get("token") == "dart")
        return DesktopServerStats(
            name=host.name,
            ok=True,
            emulators=int(payload.get("total_emulators", 0)),
            in_use=int(token_info.get("current", 0)),
            pending=int(token_info.get("pending", 0)),
        )
    except (urllib.error.URLError, TimeoutError, StopIteration, json.JSONDecodeError, KeyError) as exc:
        return DesktopServerStats(name=host.name, ok=False, error=str(exc))


def fmt_pct(value: float) -> str:
    return f"{value:5.1f}%"


def fmt_gib(used: float, total: float) -> str:
    if total <= 0:
        return "-"
    return f"{used:.1f}/{total:.1f} GiB"


def bar(pct: float, width: int = 20) -> str:
    pct = max(0.0, min(100.0, pct))
    filled = int(round((pct / 100.0) * width))
    return "[" + ("#" * filled).ljust(width, "-") + "]"


def render_table(headers: List[str], rows: List[List[str]]) -> List[str]:
    widths = [len(header) for header in headers]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], len(cell))
    lines = ["  ".join(header.ljust(widths[index]) for index, header in enumerate(headers))]
    lines.append("  ".join("-" * widths[index] for index in range(len(headers))))
    for row in rows:
        lines.append("  ".join(cell.ljust(widths[index]) for index, cell in enumerate(row)))
    return lines


def format_throughput(rate_tracker: RateTracker, remaining_tasks: int) -> str:
    rate_hour = rate_tracker.rate_per_hour()
    if rate_hour is None:
        return "throughput: -  (need more samples)"
    rate_min = rate_hour / 60.0
    window = rate_tracker.window_label()
    parts = [
        f"throughput: {rate_min:.2f} tasks/min",
        f"{rate_hour:.1f} tasks/hr",
        f"({window} window)",
    ]
    if remaining_tasks > 0 and rate_hour > 0:
        eta_hours = remaining_tasks / rate_hour
        eta = dt.timedelta(seconds=int(eta_hours * 3600))
        parts.append(f"ETA ~{eta} for {remaining_tasks} left")
    return "  ".join(parts)


def render_frame(
    *,
    title: str,
    refresh_seconds: float,
    result_dirs: List[Path],
    total: TaskBucket,
    per_host_tasks: Dict[str, TaskBucket],
    desktop_hosts: List[DesktopHost],
    desktop_stats: Dict[str, DesktopServerStats],
    vllm_stats: List[VllmGpuStats],
    rollouter_stats: Optional[HostResourceStats],
    host_resources: List[HostResourceStats],
    started_at: float,
    rate_tracker: RateTracker,
) -> str:
    now = time.time()
    elapsed = dt.timedelta(seconds=int(now - started_at))
    remaining = total.in_progress + total.pending
    throughput = format_throughput(rate_tracker, remaining)

    lines: List[str] = [
        f"{title}  refresh={refresh_seconds:g}s  elapsed={elapsed}",
        throughput,
        f"results={', '.join(path.name for path in result_dirs) if result_dirs else '-'}",
        f"updated={dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "vLLM KV cache (dart-rollouter via tunnel)",
    ]

    vllm_rows: List[List[str]] = []
    for stat in vllm_stats:
        if stat.ok:
            vllm_rows.append(
                [
                    stat.label,
                    stat.url.replace("http://", ""),
                    fmt_pct(stat.kv_cache_pct),
                    bar(stat.kv_cache_pct, 16),
                    str(stat.running),
                    str(stat.waiting),
                ]
            )
        else:
            vllm_rows.append([stat.label, stat.url.replace("http://", ""), "ERR", stat.error[:40], "-", "-"])
    lines.extend(render_table(["GPU", "URL", "KV%", "USAGE", "RUN", "WAIT"], vllm_rows))

    if rollouter_stats is not None:
        lines.extend(["", "dart-rollouter host"])
        if rollouter_stats.ok:
            gpu_bits = []
            for index, (used, total, util) in enumerate(
                zip(
                    rollouter_stats.gpu_mem_used_gib,
                    rollouter_stats.gpu_mem_total_gib,
                    rollouter_stats.gpu_util_pct,
                )
            ):
                gpu_bits.append(f"gpu{index} {used:.0f}/{total:.0f}GiB util={util:.0f}%")
            lines.append(
                f"  cpu={fmt_pct(rollouter_stats.cpu_pct)}  "
                f"ram={fmt_gib(rollouter_stats.mem_used_gib, rollouter_stats.mem_total_gib)}  "
                f"load1={rollouter_stats.load1:.2f}"
            )
            if gpu_bits:
                lines.append("  " + "  ".join(gpu_bits))
        else:
            lines.append(f"  unavailable: {rollouter_stats.error}")

    lines.extend(["", "Desktop host CPU/RAM"])
    host_rows: List[List[str]] = []
    for stat in host_resources:
        if stat.ok:
            host_rows.append(
                [
                    stat.name,
                    fmt_pct(stat.cpu_pct),
                    fmt_gib(stat.mem_used_gib, stat.mem_total_gib),
                    f"{stat.load1:.2f}",
                ]
            )
        else:
            host_rows.append([stat.name, "ERR", stat.error[:28], "-"])
    lines.extend(render_table(["HOST", "CPU", "RAM", "LOAD1"], host_rows))

    lines.extend(
        [
            "",
            f"Task progress (total {total.finished + total.failed + total.in_progress + total.pending})",
            f"  finished={total.finished}  failed={total.failed}  "
            f"in_progress={total.in_progress}  pending={total.pending}",
            "",
            "By desktop host",
        ]
    )

    host_names = sorted(
        set(per_host_tasks) | {host.name for host in desktop_hosts},
        key=lambda name: (name == "unknown", name == "unassigned", name),
    )
    by_host_rows: List[List[str]] = []
    for name in host_names:
        bucket = per_host_tasks.get(name, TaskBucket())
        server = desktop_stats.get(name)
        emu = str(server.in_use) if server and server.ok else "-"
        by_host_rows.append(
            [
                name,
                str(bucket.finished),
                str(bucket.failed),
                str(bucket.in_progress),
                str(bucket.pending),
                emu,
            ]
        )
    lines.extend(render_table(["HOST", "DONE", "FAIL", "RUN", "TODO", "EMU"], by_host_rows))

    lines.append("")
    lines.append("q=quit")
    return "\n".join(lines)


def clear_screen() -> None:
    sys.stdout.write("\x1b[H\x1b[J")
    sys.stdout.flush()


def hide_cursor() -> None:
    sys.stdout.write("\x1b[?25l")
    sys.stdout.flush()


def show_cursor() -> None:
    sys.stdout.write("\x1b[?25h")
    sys.stdout.flush()


def main() -> None:
    args = parse_args()
    if args.results_dir is None and not args.results_glob:
        raise SystemExit("one of --results-dir or --results-glob is required")

    root = args.root.resolve()
    eval_json = args.eval_json.resolve()
    tasks = load_tasks(eval_json)
    desktop_hosts = parse_desktop_hosts(args.desktop_hosts)
    host_map = parse_name_value_pairs(args.host_map)
    vllm_urls = [url.strip() for url in args.vllm_urls.split(",") if url.strip()]

    started_at = time.time()
    rate_tracker = RateTracker(window_seconds=max(1.0, float(args.rate_window_seconds)))

    hide_cursor()
    try:
        while True:
            result_dirs = resolve_result_dirs(root, args.results_dir, args.results_glob, args.all_matches)
            if result_dirs:
                total, per_host_tasks, _ = collect_task_stats(
                    tasks, result_dirs[0], args.model_subdir, host_map
                )
            else:
                total, per_host_tasks = TaskBucket(), {}

            rate_tracker.record(total.finished)

            vllm_stats = collect_vllm_stats(vllm_urls)
            desktop_stats = {host.name: collect_desktop_server(host) for host in desktop_hosts}

            rollouter_stats = None
            if not args.no_rollouter_ssh and args.rollouter_ssh:
                rollouter_stats = collect_ssh_resources("dart-rollouter", args.rollouter_ssh, include_gpu=True)

            host_resources: List[HostResourceStats] = []
            for host in desktop_hosts:
                if host.name == "coordinator":
                    host_resources.append(collect_local_resources(host.name))
                elif host.ssh_host:
                    host_resources.append(collect_ssh_resources(host.name, host.ssh_host))

            frame = render_frame(
                title=args.title,
                refresh_seconds=args.refresh_seconds,
                result_dirs=result_dirs,
                total=total,
                per_host_tasks=per_host_tasks,
                desktop_hosts=desktop_hosts,
                desktop_stats=desktop_stats,
                vllm_stats=vllm_stats,
                rollouter_stats=rollouter_stats,
                host_resources=host_resources,
                started_at=started_at,
                rate_tracker=rate_tracker,
            )
            clear_screen()
            print(frame, flush=True)
            time.sleep(args.refresh_seconds)
    except KeyboardInterrupt:
        pass
    finally:
        show_cursor()
        clear_screen()


if __name__ == "__main__":
    main()
