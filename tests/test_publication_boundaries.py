"""Unconfigured private integrations cannot inherit the installed owner's endpoints."""

import importlib

import pytest

from dodo_memory import resources
from dodo_memory.bridge import LOCAL_MODEL_TOOLS
from dodo_memory.offline_docling import parse_offline_document
from dodo_memory.server import make_app
from dodo_memory.store import Vault


def test_snapshot_omits_notification_module_and_tool(tmp_path):
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("dodo_memory.completion_notifications")
    assert "notify_task_completed" not in LOCAL_MODEL_TOOLS
    make_app(Vault(tmp_path), {}, token="synthetic-token-" * 3)


def test_offline_parser_requires_runtime_path_before_any_file_access(tmp_path):
    with pytest.raises(TypeError, match="runtime_root"):
        parse_offline_document(
            tmp_path / "absent.pdf",
            work_dir=tmp_path,
            runtime_manifest_sha256="0" * 64,
            model_manifest_sha256="1" * 64,
        )


def test_disk_refusal_preserves_real_floor_with_synthetic_capacity(tmp_path, monkeypatch):
    assert 100 * resources.GiB == resources.MIN_FREE_BYTES
    disk_usage = type(resources.shutil.disk_usage(tmp_path))
    budget = resources.ResourceBudget(tmp_path, quota_bytes=4 * resources.GiB)
    monkeypatch.setattr(
        resources.shutil,
        "disk_usage",
        lambda _path: disk_usage(200 * resources.GiB, 0, resources.MIN_FREE_BYTES - 1),
    )
    with pytest.raises(resources.ResourceRefused) as refused:
        budget.admit(0, additional_entries=0)
    assert refused.value.code == "disk_headroom"
    monkeypatch.setattr(
        resources.shutil,
        "disk_usage",
        lambda _path: disk_usage(200 * resources.GiB, 0, resources.MIN_FREE_BYTES),
    )
    assert budget.admit(0, additional_entries=0)["free_disk_bytes"] == resources.MIN_FREE_BYTES


def test_memory_refusal_preserves_real_floor_with_synthetic_capacity(tmp_path, monkeypatch):
    assert 5 * resources.GiB == resources.MIN_AVAILABLE_BYTES
    budget = resources.ResourceBudget(tmp_path, quota_bytes=4 * resources.GiB)
    monkeypatch.setattr(resources, "_available_memory", lambda: resources.MIN_AVAILABLE_BYTES - 1)
    with pytest.raises(resources.ResourceRefused) as refused:
        budget.admit(0, additional_entries=0, require_memory=True)
    assert refused.value.code == "memory_floor"
    monkeypatch.setattr(resources, "_available_memory", lambda: resources.MIN_AVAILABLE_BYTES)
    admitted = budget.admit(0, additional_entries=0, require_memory=True)
    assert admitted["available_memory_bytes"] == resources.MIN_AVAILABLE_BYTES
