"""Resource gates never evict other workloads."""

import ctypes
import json
import os
import shutil
from pathlib import Path

import psutil
from dodo_memory_desktop.configuration import configured_path, runtime_path

GIB = 1024**3


def configured_cache(home=None):
    """Keep cache writes inside the explicitly configured state bulk directory."""
    home = configured_path(str(home)) if home is not None else runtime_path("neural")
    configuration = home / "cache-root.json"
    original = configured_path(str(home / "bulk"))
    if not configuration.exists() and not configuration.is_symlink():
        return original
    if configuration.resolve(strict=True) != configuration.absolute():
        raise ValueError("Neural cache configuration cannot contain redirects.")
    with configuration.open("rb") as stream:
        raw = stream.read(4097)
    if len(raw) > 4096:
        raise ValueError("Neural cache configuration exceeds 4096 bytes.")
    value = json.loads(raw)
    if not isinstance(value, dict) or set(value) != {"root"}:
        raise ValueError("Neural cache configuration must contain only root.")
    if not isinstance(value["root"], str) or not value["root"]:
        raise ValueError("Neural cache root must be a nonempty absolute path.")
    root = Path(value["root"])
    if not root.is_absolute():
        raise ValueError("Neural cache root must be an absolute path.")
    if root != original:
        raise ValueError("Neural cache root must be the configured state owner bulk directory.")
    if root.resolve(strict=True) != root.absolute() or not root.is_dir():
        raise ValueError("Neural cache root must be an existing directory without redirects.")
    return root


DEFAULT_CACHE = configured_cache()


class ResourcePressure(RuntimeError):
    pass


def enforce_limits(*, available_ram, free_disk, cache_bytes, additional_disk=0):
    if available_ram < 5 * GIB:
        raise ResourcePressure("Background work paused: less than 5 GiB available RAM.")
    if cache_bytes + additional_disk > 20 * GIB:
        raise ResourcePressure("Neural cache quota would exceed 20 GiB.")
    if free_disk - additional_disk < 100 * GIB:
        raise ResourcePressure("Background work paused: disk headroom would fall below 100 GiB.")


def cache_size(root):
    return sum(
        entry.stat().st_size
        for base, _, names in os.walk(root, followlinks=False)
        for name in names
        if (entry := Path(base) / name).is_file() and not entry.is_symlink()
    )


def check_resources(background=True, additional_disk=0, cache=DEFAULT_CACHE):
    cache = Path(cache)
    if not cache.anchor or not Path(cache.anchor).exists():
        raise ResourcePressure(f"Derived index drive unavailable: {cache.anchor}")
    ram = psutil.virtual_memory()
    free = shutil.disk_usage(cache.anchor).free
    size = cache_size(cache) if cache.exists() else 0
    enforce_limits(
        available_ram=ram.available,
        free_disk=free,
        cache_bytes=size,
        additional_disk=additional_disk,
    )
    if os.name == "nt" and shutil.disk_usage("C:/").free < 100 * GIB:
        raise ResourcePressure("Background work paused: C: free disk is below 100 GiB.")
    return {
        "available_ram": ram.available,
        "system_commit": system_commit(),
        "cache_bytes": size,
        "free_disk": free,
        "background": background,
    }


def system_commit():
    if os.name != "nt":
        return None

    class PerformanceInfo(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_ulong),
            ("CommitTotal", ctypes.c_size_t),
            ("CommitLimit", ctypes.c_size_t),
            ("CommitPeak", ctypes.c_size_t),
            ("PhysicalTotal", ctypes.c_size_t),
            ("PhysicalAvailable", ctypes.c_size_t),
            ("SystemCache", ctypes.c_size_t),
            ("KernelTotal", ctypes.c_size_t),
            ("KernelPaged", ctypes.c_size_t),
            ("KernelNonpaged", ctypes.c_size_t),
            ("PageSize", ctypes.c_size_t),
            ("HandleCount", ctypes.c_ulong),
            ("ProcessCount", ctypes.c_ulong),
            ("ThreadCount", ctypes.c_ulong),
        ]

    info = PerformanceInfo()
    info.cb = ctypes.sizeof(info)
    if not ctypes.windll.psapi.GetPerformanceInfo(ctypes.byref(info), info.cb):
        raise ctypes.WinError()
    return {
        "used": info.CommitTotal * info.PageSize,
        "limit": info.CommitLimit * info.PageSize,
        "peak": info.CommitPeak * info.PageSize,
    }
