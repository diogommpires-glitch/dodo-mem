"""Fresh metadata-only admission for owned vault data and operation headroom."""

import hashlib
import json
import math
import os
import shutil
import stat
import time
from pathlib import Path, PurePosixPath

from .parsing import _available_memory

GiB = 1024**3
MIN_FREE_BYTES = 100 * GiB
MIN_AVAILABLE_BYTES = 5 * GiB
MIN_FREE_INODES = 10000
ENTRY_RESERVE_BYTES = 8192
OWNED_ROOTS = ("vault", "originals", "derived", ".state", "generated")
MAX_DEPTH = 32
MAX_SCAN_METADATA_BYTES = 32 * 1024**2
MAX_POLICY_BYTES = 1024**4


class ResourceRefused(RuntimeError):
    def __init__(self, code, **details):
        self.code, self.details = code, details
        super().__init__("Resource admission refused: " + code)


def _integer(value, name, maximum=MAX_POLICY_BYTES):
    if type(value) is not int or not 0 <= value <= maximum:
        raise ValueError(f"{name} must be an explicit bounded non-negative integer")
    return value


def _redirect(info):
    return stat.S_ISLNK(info.st_mode) or bool(
        getattr(info, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 1024)
    )


def _unlinked_home(home):
    for path in (home, *home.parents):
        if _redirect(path.lstat()) or getattr(path, "is_junction", lambda: False)():
            raise ResourceRefused("data_link")
    if home.resolve(strict=True) != home or not home.is_dir():
        raise ResourceRefused("invalid_owned_home")


def _scan(home, exclusions, maximum, deadline):
    files, entries, content_bytes, fingerprint = 0, 0, 0, 0
    stack = [home / name for name in OWNED_ROOTS]
    pending_bytes = sum(len(os.fsencode(path)) for path in stack)
    while stack:
        if time.monotonic() >= deadline:
            raise ResourceRefused("scan_deadline")
        path = stack.pop()
        pending_bytes -= len(os.fsencode(path))
        relative = path.relative_to(home).as_posix()
        if any(
            relative == excluded or relative.startswith(excluded + "/") for excluded in exclusions
        ):
            continue
        try:
            before = path.lstat()
        except FileNotFoundError:
            if relative in OWNED_ROOTS:
                continue
            raise ResourceRefused("inventory_changed") from None
        if len(path.relative_to(home).parts) > MAX_DEPTH:
            raise ResourceRefused("scan_depth_limit")
        entries += 1
        if entries > maximum:
            raise ResourceRefused("scan_entry_limit")
        if _redirect(before) or getattr(path, "is_junction", lambda: False)():
            raise ResourceRefused("data_link")
        directory = stat.S_ISDIR(before.st_mode)
        if relative in OWNED_ROOTS and not directory:
            raise ResourceRefused("invalid_owned_namespace")
        if directory:
            if path.resolve(strict=True) != path:
                raise ResourceRefused("data_link")
            with os.scandir(path) as children:
                for child in children:
                    if time.monotonic() >= deadline:
                        raise ResourceRefused("scan_deadline")
                    # Bound pending names as well as visited entries; never materialise a huge directory.
                    if entries + len(stack) >= maximum + len(OWNED_ROOTS):
                        raise ResourceRefused("scan_entry_limit")
                    pending_bytes += len(os.fsencode(child.path))
                    if pending_bytes > MAX_SCAN_METADATA_BYTES:
                        raise ResourceRefused("scan_metadata_limit")
                    stack.append(Path(child.path))
            after = path.lstat()
            if _redirect(after) or (before.st_dev, before.st_ino, before.st_mtime_ns) != (
                after.st_dev,
                after.st_ino,
                after.st_mtime_ns,
            ):
                raise ResourceRefused("inventory_changed")
        elif stat.S_ISREG(before.st_mode):
            if before.st_nlink != 1:
                raise ResourceRefused("data_hardlink")
            if before.st_size < 0:
                raise ResourceRefused("scan_unavailable")
            files += 1
            content_bytes += before.st_size
        else:
            raise ResourceRefused("unsupported_data_entry")
        # Order-independent digest avoids retaining private filenames or sorting unbounded directories.
        record = [
            relative,
            before.st_dev,
            before.st_ino,
            0 if directory else before.st_size,
            directory,
        ]
        fingerprint = (
            fingerprint + int.from_bytes(hashlib.sha256(json.dumps(record).encode()).digest())
        ) % (1 << 256)
    return {
        "owned_file_bytes": content_bytes,
        "file_count": files,
        "entry_count": entries,
        "inventory_sha256": f"{fingerprint:064x}",
    }


def _inode_capacity(home, *, platform=None):
    platform = os.name if platform is None else platform
    if platform == "nt":
        return None, "dynamic_windows"
    try:
        info = os.statvfs(home)
    except (AttributeError, OSError):
        raise ResourceRefused("inode_statistics_unavailable") from None
    if type(info.f_files) is not int or info.f_files < 0:
        raise ResourceRefused("inode_statistics_unavailable")
    if info.f_files == 0:
        return None, "dynamic_filesystem"
    if type(info.f_favail) is not int or info.f_favail < 0:
        raise ResourceRefused("inode_statistics_unavailable")
    return info.f_favail, "finite"


def _capacity(home):
    try:
        free = shutil.disk_usage(home).free
    except OSError:
        raise ResourceRefused("disk_statistics_unavailable") from None
    if type(free) is not int or free < 0:
        raise ResourceRefused("disk_statistics_unavailable")
    inodes, mode = _inode_capacity(home)
    return free, inodes, mode


class ResourceBudget:
    def __init__(
        self, home, *, quota_bytes, excluded_runtime_roots=(), max_entries=100000, scan_seconds=10
    ):
        home = Path(home)
        if not home.is_absolute():
            raise ValueError("Budget home must be an explicit absolute owned vault root")
        _integer(quota_bytes, "quota_bytes")
        if quota_bytes == 0:
            raise ValueError("quota_bytes must be an explicit positive owner policy")
        if type(max_entries) is not int or not 1 <= max_entries <= 1000000:
            raise ValueError("Scan entry limit must be an integer from 1 to 1000000")
        if (
            type(scan_seconds) not in {int, float}
            or not math.isfinite(scan_seconds)
            or not 0 < scan_seconds <= 60
        ):
            raise ValueError("Metadata scan budget must be finite and at most 60 seconds")
        if not isinstance(excluded_runtime_roots, (list, tuple)):
            raise TypeError("Runtime exclusions require exact owner-declared relative roots")
        exclusions = []
        for value in excluded_runtime_roots:
            if not isinstance(value, str):
                raise TypeError("Runtime exclusion must be a canonical relative string")
            relative = PurePosixPath(value)
            if (
                value == ".state/private-evaluation"
                or not value.startswith(".state/")
                or relative.as_posix() != value
                or ".." in relative.parts
                or ":" in value
                or "\\" in value
                or any(ord(character) < 32 for character in value)
                or any(part.endswith((".", " ")) for part in relative.parts)
                or any(
                    value == protected or value.startswith(protected + "/")
                    for protected in (
                        ".state/native-capture",
                        ".state/ingest-logs",
                        ".state/results",
                        ".state/memory.sqlite",
                        ".state/memory.sqlite-wal",
                        ".state/memory.sqlite-shm",
                    )
                )
            ):
                raise ValueError(
                    "Only exact declared .state runtime roots may be excluded, never source capture trees"
                )
            if any(
                value == old or value.startswith(old + "/") or old.startswith(value + "/")
                for old in exclusions
            ):
                raise ValueError("Overlapping runtime exclusions are ambiguous")
            exclusions.append(value)
        self.home, self.quota_bytes = home, quota_bytes
        self.exclusions, self.max_entries, self.scan_seconds = (
            tuple(sorted(exclusions)),
            max_entries,
            scan_seconds,
        )

    def admit(
        self,
        additional_bytes,
        *,
        additional_entries=1,
        temporary_bytes=0,
        temporary_entries=0,
        backup_copies=1,
        require_memory=False,
    ):
        additional_bytes = _integer(additional_bytes, "additional_bytes")
        temporary_bytes = _integer(temporary_bytes, "temporary_bytes")
        additional_entries = _integer(additional_entries, "additional_entries", 1000000)
        temporary_entries = _integer(temporary_entries, "temporary_entries", 1000000)
        if type(backup_copies) is not int or not 1 <= backup_copies <= 2:
            raise ValueError("Reserve one or two explicit projected backup copies")
        if type(require_memory) is not bool:
            raise TypeError("require_memory must be a boolean")
        deadline = time.monotonic() + self.scan_seconds
        try:
            _unlinked_home(self.home)
            first = _scan(self.home, self.exclusions, self.max_entries, deadline)
            inventory = _scan(self.home, self.exclusions, self.max_entries, deadline)
            if inventory != first:
                raise ResourceRefused("inventory_changed")
            _unlinked_home(self.home)
            free, inodes, inode_mode = _capacity(self.home)
        except OSError:
            raise ResourceRefused("scan_or_capacity_unavailable") from None
        accounted = inventory["owned_file_bytes"] + inventory["entry_count"] * ENTRY_RESERVE_BYTES
        projected = accounted + additional_bytes + additional_entries * ENTRY_RESERVE_BYTES
        if projected > self.quota_bytes:
            raise ResourceRefused(
                "data_quota",
                accounted_bytes=accounted,
                projected_bytes=projected,
                quota_bytes=self.quota_bytes,
            )
        if temporary_bytes > self.quota_bytes:
            raise ResourceRefused("temporary_reservation_limit")
        projected_entries = inventory["entry_count"] + additional_entries
        needed_entries = additional_entries + temporary_entries + backup_copies * projected_entries
        required_free = (
            MIN_FREE_BYTES
            + additional_bytes
            + additional_entries * ENTRY_RESERVE_BYTES
            + temporary_bytes
            + temporary_entries * ENTRY_RESERVE_BYTES
            + backup_copies * projected
        )
        if free < required_free:
            raise ResourceRefused("disk_headroom", free_bytes=free, required_bytes=required_free)
        if inodes is not None and inodes < MIN_FREE_INODES + needed_entries:
            raise ResourceRefused(
                "inode_headroom", available=inodes, required=MIN_FREE_INODES + needed_entries
            )
        memory = None
        if require_memory:
            try:
                memory = _available_memory()
            except OSError:
                raise ResourceRefused("memory_statistics_unavailable") from None
            if type(memory) is not int or memory < MIN_AVAILABLE_BYTES:
                raise ResourceRefused("memory_floor", available_bytes=memory)
        if time.monotonic() >= deadline:
            raise ResourceRefused("scan_deadline")
        return {
            "policy_version": 1,
            **inventory,
            "owned_accounted_bytes": accounted,
            "projected_accounted_bytes": projected,
            "quota_bytes": self.quota_bytes,
            "projected_entries": projected_entries,
            "required_free_bytes": required_free,
            "free_disk_bytes": free,
            "backup_copies": backup_copies,
            "required_inode_entries": needed_entries,
            "free_inodes": inodes,
            "inode_policy": inode_mode,
            "available_memory_bytes": memory,
            "runtime_exclusions": len(self.exclusions),
        }


def storage_budget(home):
    """Load an explicit versioned policy; owner overrides cannot change reserve floors."""
    home = Path(home).absolute()
    override = home / "storage.local.json"
    policy_path = (
        override
        if override.exists()
        else Path(__file__).resolve().parents[1] / "profiles/storage-budget.json"
    )
    if policy_path.resolve(strict=True) != policy_path.absolute():
        raise ValueError("Storage policy cannot contain a redirection")
    with policy_path.open("rb") as stream:
        content = stream.read(65537)
    if len(content) > 65536:
        raise ValueError("Storage policy exceeds its 64 KiB limit")
    policy = json.loads(content)
    if (
        not isinstance(policy, dict)
        or set(policy)
        != {
            "policy_version",
            "quota_bytes",
            "excluded_runtime_roots",
            "max_entries",
            "scan_seconds",
        }
        or type(policy["policy_version"]) is not int
        or policy["policy_version"] != 1
    ):
        raise ValueError(
            "Storage policy requires explicit supported quota, runtime exclusions and scan bounds"
        )
    return ResourceBudget(
        home, **{name: value for name, value in policy.items() if name != "policy_version"}
    )


def admit_storage(home, additional_bytes, **options):
    return storage_budget(home).admit(additional_bytes, **options)
