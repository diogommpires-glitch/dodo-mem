import hashlib
import json

import pytest

from dodo_mem_neural import shortcut_rollout


def fixture_manifest(tmp_path):
    old = tmp_path / "Desktop/Dodo Mem.lnk"
    old.parent.mkdir()
    old.write_bytes(b"original shortcut")
    stage = tmp_path / "rollout"
    stage.mkdir()
    backup = stage / "original.lnk"
    backup.write_bytes(old.read_bytes())
    candidate = stage / "candidate.lnk"
    candidate.write_bytes(b"updated shortcut")
    entry = {
        "target": str(old),
        "staged": str(candidate),
        "backup": str(backup),
        "before": hashlib.sha256(old.read_bytes()).hexdigest(),
        "after": hashlib.sha256(candidate.read_bytes()).hexdigest(),
    }
    manifest = stage / "manifest.json"
    manifest.write_text(json.dumps({"entries": [entry], "state": "staged"}))
    return manifest, old


def test_conflicting_shortcut_is_preserved_and_roundtrip_restores_exact_bytes(
    tmp_path, monkeypatch
):
    manifest, target = fixture_manifest(tmp_path)
    monkeypatch.setattr(shortcut_rollout, "SHORTCUT_TARGETS", (target,))
    target.write_bytes(b"concurrent user shortcut")
    with pytest.raises(ValueError, match="changed"):
        shortcut_rollout.install(manifest)
    assert target.read_bytes() == b"concurrent user shortcut"
    target.write_bytes(b"original shortcut")
    shortcut_rollout.install(manifest)
    assert target.read_bytes() == b"updated shortcut"
    shortcut_rollout.rollback(manifest)
    assert target.read_bytes() == b"original shortcut"


def test_new_shortcut_is_removed_only_after_matching_installed_hash(tmp_path, monkeypatch):
    manifest, target = fixture_manifest(tmp_path)
    target.unlink()
    document = json.loads(manifest.read_text())
    document["entries"][0].update(before=None, backup=None)
    manifest.write_text(json.dumps(document))
    monkeypatch.setattr(shortcut_rollout, "SHORTCUT_TARGETS", (target,))
    shortcut_rollout.install(manifest)
    assert target.read_bytes() == b"updated shortcut"
    shortcut_rollout.rollback(manifest)
    assert not target.exists()
