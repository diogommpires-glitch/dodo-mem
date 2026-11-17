import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from dodo_mem_neural import resources


def test_cache_root_requires_an_explicit_canonical_owner_bulk_directory(tmp_path):
    assert resources.configured_cache(tmp_path) == tmp_path / "bulk"
    bulk = tmp_path / "bulk"
    bulk.mkdir()
    configuration = tmp_path / "cache-root.json"
    configuration.write_text(json.dumps({"root": str(bulk)}))
    assert resources.configured_cache(tmp_path) == bulk
    configuration.write_text(json.dumps({"root": str(tmp_path)}))
    with pytest.raises(ValueError, match="owner bulk"):
        resources.configured_cache(tmp_path)
    configuration.write_text(json.dumps({"root": "bulk"}))
    with pytest.raises(ValueError, match="absolute"):
        resources.configured_cache(tmp_path)
    configuration.write_text(json.dumps({"root": str(bulk), "reserve_gib": 0}))
    with pytest.raises(ValueError, match="only root"):
        resources.configured_cache(tmp_path)


def test_cache_root_rejects_a_redirected_bulk_directory(tmp_path):
    bulk = tmp_path / "bulk"
    target = tmp_path / "elsewhere"
    target.mkdir()
    try:
        bulk.symlink_to(target, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"Directory symlinks unavailable: {error}")
    (tmp_path / "cache-root.json").write_text(json.dumps({"root": str(bulk)}))
    with pytest.raises(ValueError, match="redirect"):
        resources.configured_cache(tmp_path)


def test_default_cache_rejects_redirects_without_a_configuration_file(tmp_path):
    target = tmp_path / "outside"
    target.mkdir()
    state = tmp_path / "state"
    state.mkdir()
    try:
        (state / "bulk").symlink_to(target, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"Directory symlinks unavailable: {error}")
    with pytest.raises(ValueError, match="redirect"):
        resources.configured_cache(state)


def test_selected_cache_keeps_existing_ram_disk_and_quota_limits(tmp_path, monkeypatch):
    monkeypatch.setattr(
        resources.psutil, "virtual_memory", lambda: SimpleNamespace(available=6 * resources.GIB)
    )
    monkeypatch.setattr(
        resources.shutil, "disk_usage", lambda _: SimpleNamespace(free=99 * resources.GIB)
    )
    with pytest.raises(resources.ResourcePressure, match="100 GiB"):
        resources.check_resources(cache=tmp_path)


def test_new_process_uses_one_selected_root_for_runtime_indexes_and_parameters(tmp_path):
    state = tmp_path / "neural"
    bulk = state / "bulk"
    bulk.mkdir(parents=True)
    (state / "cache-root.json").write_text(json.dumps({"root": str(bulk)}))
    script = """
import inspect,json
from dodo_mem_neural import artifacts,resources,runtime,service,static_weights
print(json.dumps([
    str(resources.DEFAULT_CACHE),
    str(service.DEFAULT_CACHE),
    str(inspect.signature(artifacts.download).parameters['cache'].default),
    str(inspect.signature(runtime.NeuralRuntime).parameters['cache'].default),
    str(inspect.signature(static_weights.static_model_info).parameters['cache'].default),
    str(static_weights.PREPARED_E5.parent.parent),
]))
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        env={
            **os.environ,
            "USERPROFILE": str(tmp_path),
            "HOME": str(tmp_path),
            "DODO_MEM_RUNTIME_ROOT": str(tmp_path),
        },
        check=True,
        capture_output=True,
        text=True,
    )
    assert [Path(value) for value in json.loads(result.stdout)] == [bulk] * 6
