from pathlib import Path

from dodo_mem_neural.catalogue_adapter import AuthorisedCatalogue


class Catalogue:
    def search(self, *args, **kwargs):
        return []

    def node(self, node_id):
        return None


class Vault:
    root = Path("C:/synthetic-vault")

    def search(self, *args, **kwargs):
        return []


def test_local_profile_sources_are_never_shared(tmp_path):
    source = tmp_path / "coding.md"
    source.write_text("Prefer focused executable checks and preserve units.")
    from dataclasses import dataclass

    @dataclass
    class ProfileSource:
        path: Path
        section: str = "Coding"
        basis: str = "confirmed"
        enabled: bool = True

    class Profile:
        enabled = True

        def sources(self):
            return (ProfileSource(source, enabled=self.enabled),)

    profile = Profile()
    adapter = AuthorisedCatalogue(Catalogue(), Vault(), profile=profile)
    rows = adapter.search("implement python code", scope="full_private")
    assert len(rows) == 1
    assert adapter.authorized(rows[0]["id"], "full_private")
    assert not adapter.authorized(rows[0]["id"], "shared")
    assert adapter.search("implement python code", scope="shared") == []
    profile.enabled = False
    assert not adapter.authorized(rows[0]["id"], "full_private")
