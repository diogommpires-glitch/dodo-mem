"""Declare synthetic storage policy without copying the owner's live configuration."""

import json

import pytest

from dodo_memory import resources


@pytest.fixture(autouse=True)
def synthetic_storage_policy(tmp_path, monkeypatch):
    disk_usage = type(resources.shutil.disk_usage(tmp_path))
    monkeypatch.setattr(
        resources.shutil,
        "disk_usage",
        lambda _path: disk_usage(200 * resources.GiB, 50 * resources.GiB, 150 * resources.GiB),
    )
    monkeypatch.setattr(resources, "_available_memory", lambda: 8 * resources.GiB)
    policy = {
        "policy_version": 1,
        "quota_bytes": 4 * 1024**3,
        "excluded_runtime_roots": [],
        "max_entries": 10000,
        "scan_seconds": 10,
    }
    for home in (tmp_path, tmp_path / "memory"):
        home.mkdir(exist_ok=True)
        (home / "storage.local.json").write_text(json.dumps(policy), encoding="utf-8")
