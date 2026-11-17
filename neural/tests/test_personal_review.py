"""Independent review checks for personal context relevance and provenance."""

from pathlib import Path
from types import SimpleNamespace

import pytest
from dodo_memory_desktop import personal_context
from dodo_memory_desktop.personal_context import PersonalProfile, ProfileSource

from dodo_mem_neural.catalogue_adapter import AuthorisedCatalogue
from dodo_mem_neural.retrieval import RetrievalEngine, SourceChanged


def test_personal_profile_rejects_remote_paths_without_accessing_them():
    with pytest.raises(ValueError, match="local|network|UNC"):
        PersonalProfile._validate(
            (ProfileSource(Path(r"\\synthetic-server\share\note.md"), "Coding"),)
        )


@pytest.mark.parametrize("path", (r"\\synthetic-server\share\note.md", r"\\?\C:\synthetic\note.md"))
def test_profile_source_read_rejects_network_and_device_paths_before_io(monkeypatch, path):
    def forbidden(*args, **kwargs):
        raise AssertionError("The rejected source must not reach file reading.")

    monkeypatch.setattr(personal_context, "read_bytes", forbidden)
    with pytest.raises(ValueError, match="local|authorised"):
        personal_context.read_profile_file(Path(path), 100)


@pytest.mark.parametrize(
    ("section", "question"),
    (("Coding", "implement python gradients"), ("Relationships", "reply to my friend")),
)
def test_approved_profile_does_not_displace_relevant_private_source(tmp_path, section, question):
    standing = tmp_path / "standing.md"
    coding = tmp_path / "coding.md"
    standing.write_text("Keep a clear and direct writing style.")
    coding.write_text("Use SI units and verify numerical gradients.")
    profile = SimpleNamespace(
        sources=lambda: (
            ProfileSource(standing, "Standing preferences"),
            ProfileSource(coding, section),
        )
    )
    catalogue = SimpleNamespace(search=lambda *a, **kw: [], node=lambda _: None)
    vault = SimpleNamespace(root=tmp_path / "vault", search=lambda *a, **kw: [])
    adapter = AuthorisedCatalogue(catalogue, vault, profile=profile)
    adapter.knowledge = SimpleNamespace(
        read=lambda *a, **kw: {
            "status": "found",
            "text": "Approved public preferences",
            "path": "profile.md",
        }
    )
    adapter._shared = lambda _: (vault.root / "profile.md", "shared-v1", {"text": "approved"})
    rows = adapter.search(question, scope="full_private")
    assert any(row["path"] == str(coding) for row in rows)
    assert all(
        not row["node_id"].startswith("profile-local:")
        for row in adapter.search("code", scope="shared")
    )


async def test_retrieval_preserves_historical_role_and_profile_basis():
    node = {"id": "historical", "path": "synthetic.md", "digest": "v1", "state": "indexed"}
    evidence = {
        "path": "synthetic.md",
        "digest": "v1",
        "locator": {"line": 1},
        "text": "An assistant once guessed that the learning rate was correct.",
        "role": "assistant",
        "timestamp": "2026-11-01T12:00:00Z",
        "historical": True,
        "profile": {"section": "Interests", "basis": "inferred"},
    }
    catalogue = SimpleNamespace(
        authorized=lambda *a: True,
        node=lambda _: node,
        search=lambda *a, **kw: [node],
        read_evidence=lambda *a, **kw: evidence,
        neighbours=lambda *a, **kw: [],
    )

    async def count(text, model):
        return len(text) // 4

    packet = await RetrievalEngine(catalogue, None, None, None, count).retrieve(
        "What was the learning rate?", "full_private", "local"
    )
    source = packet.sources[0]
    assert source["historical"] is True
    assert source["role"] == "assistant"
    assert source["timestamp"] == evidence["timestamp"]
    assert source["profile"]["basis"] == "inferred"


async def test_profile_basis_change_invalidates_a_saved_packet_before_dispatch():
    node = {"id": "profile", "path": "synthetic.md", "digest": "v1", "state": "indexed"}
    basis = "confirmed"

    def read(*args, **kwargs):
        return {
            "path": "synthetic.md",
            "digest": "v1",
            "locator": {"line": 1},
            "text": "Prefer numerical checks.",
            "profile": {"section": "Coding", "basis": basis},
        }

    catalogue = SimpleNamespace(
        authorized=lambda *a: True,
        node=lambda _: node,
        search=lambda *a, **kw: [node],
        read_evidence=read,
        neighbours=lambda *a, **kw: [],
    )

    async def count(text, model):
        return len(text) // 4

    engine = RetrievalEngine(catalogue, None, None, None, count)
    packet = await engine.retrieve("numerical checks", "full_private", "local")
    basis = "inferred"
    with pytest.raises(SourceChanged):
        engine.verify(packet, "full_private")
