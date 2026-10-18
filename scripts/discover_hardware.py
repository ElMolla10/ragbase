"""Read-only capacity inventory; no inference, downloads or hardware benchmarks.

Run on the actual host, outside Docker. Does not collect usernames, hostnames,
serial numbers, network addresses, environment variables or model prompts.
"""

import argparse
import json
import os
import platform
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path


def command_json(command):
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=20, check=True
        )
        return json.loads(result.stdout)
    except (OSError, subprocess.SubprocessError, ValueError):
        return {"status": "unavailable"}


def linux_inventory():
    memory = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, value = line.split(":", 1)
            if key in {"MemTotal", "MemAvailable", "SwapTotal", "SwapFree"}:
                memory[key + "_bytes"] = int(value.split()[0]) * 1024
    except (OSError, ValueError):
        memory["status"] = "unavailable"
    cpu = command_json(["lscpu", "--json"])
    allowed = {
        "Architecture:",
        "CPU(s):",
        "On-line CPU(s) list:",
        "Model name:",
        "Thread(s) per core:",
        "Core(s) per socket:",
        "Socket(s):",
        "NUMA node(s):",
        "Flags:",
    }
    return {
        "memory": memory,
        "cpu": {
            row["field"]: row["data"]
            for row in cpu.get("lscpu", [])
            if row["field"] in allowed
        },
        "process_cpu_affinity": sorted(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else None,
    }


def windows_inventory():
    script = """
    $cpu = @(Get-CimInstance Win32_Processor | Select-Object Name,NumberOfCores,NumberOfLogicalProcessors)
    $osInfo = Get-CimInstance Win32_OperatingSystem | Select-Object Caption,TotalVisibleMemorySize,FreePhysicalMemory
    @{cpu=$cpu; memory_and_os=$osInfo} | ConvertTo-Json -Depth 4
    """
    return command_json(["powershell", "-NoProfile", "-Command", script])


def collect(role):
    inventory = (
        windows_inventory() if platform.system() == "Windows" else linux_inventory()
    )
    docker = command_json(["docker", "info", "--format", "{{json .}}"])
    disk = shutil.disk_usage(Path.cwd())
    return {
        "schema_version": 1,
        "collected_at": datetime.now(UTC).isoformat(),
        "host_role": role,
        "system": platform.system(),
        "architecture": platform.machine(),
        "logical_cpus_visible_to_python": os.cpu_count(),
        "host": inventory,
        "working_filesystem_bytes": {"total": disk.total, "free": disk.free},
        "docker": {
            key: docker.get(key)
            for key in ("OSType", "Architecture", "NCPU", "MemTotal", "ServerVersion")
        },
        "limitations": [
            "Inventory only; memory bandwidth, inference speed and loading peaks are unmeasured.",
            "Laptop results do not qualify the company server.",
            "ISA, virtualization/cgroup limits, storage and competing workloads need operator review.",
        ],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=["laptop", "server"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(collect(args.role), indent=2) + "\n", encoding="utf-8"
    )
    print(f"Capacity inventory written to {args.output}")
