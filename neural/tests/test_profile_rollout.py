import hashlib
import json

import pytest

from dodo_mem_neural import profile_rollout


def fixture(tmp_path):
    home = tmp_path / "home"
    target = home / "local-ai/coder/cline-data/globalState.json"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"original profile")
    registry = home / "local-ai/models.json"
    registry.write_text(json.dumps({"coder": {}}))
    stage = home / "local-ai/dodo-mem-neural/staged/test"
    stage.mkdir(parents=True)
    candidate, backup = stage / "candidate.json", stage / "backup.json"
    candidate.write_bytes(b"memory profile")
    backup.write_bytes(target.read_bytes())
    manifest = stage / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "files": [
                    {
                        "target": str(target),
                        "staged": str(candidate),
                        "backup": str(backup),
                        "before_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                        "after_sha256": hashlib.sha256(candidate.read_bytes()).hexdigest(),
                    }
                ]
            }
        )
    )
    return home, target, manifest


def test_profile_activation_requires_gates_and_preserves_concurrent_edits(tmp_path, monkeypatch):
    home, target, manifest = fixture(tmp_path)
    with pytest.raises(ValueError, match="qualification"):
        profile_rollout.install(manifest, reports={}, home=home, appdata=home / "appdata")
    monkeypatch.setattr(profile_rollout, "qualify_reports", lambda _: None)
    target.write_bytes(b"new editor changes")
    with pytest.raises(ValueError, match="changed"):
        profile_rollout.install(manifest, reports={}, home=home, appdata=home / "appdata")
    assert target.read_bytes() == b"new editor changes"
    target.write_bytes(b"original profile")
    profile_rollout.install(manifest, reports={}, home=home, appdata=home / "appdata")
    assert target.read_bytes() == b"memory profile"
    assert json.loads((home / "local-ai/dodo-mem-neural/clients.active.json").read_text())[
        "enabled"
    ]
    target.write_bytes(b"later user changes")
    with pytest.raises(ValueError, match="changed"):
        profile_rollout.rollback(manifest, home=home, appdata=home / "appdata")
    assert target.read_bytes() == b"later user changes"
    target.write_bytes(b"memory profile")
    profile_rollout.rollback(manifest, home=home, appdata=home / "appdata")
    assert target.read_bytes() == b"original profile"
    assert not json.loads((home / "local-ai/dodo-mem-neural/clients.active.json").read_text())[
        "enabled"
    ]
