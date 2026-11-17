from pathlib import Path

from dodo_mem_neural.sqlite_history import discover_cline_databases


def test_client_databases_are_never_discovered_without_explicit_selection(tmp_path, monkeypatch):
    monkeypatch.delenv("DODO_MEM_CLINE_DATABASES", raising=False)
    assert discover_cline_databases() == []
    database = tmp_path / "synthetic.db"
    database.write_bytes(b"")
    monkeypatch.setenv("DODO_MEM_CLINE_DATABASES", f'["{database.as_posix()}"]')
    assert discover_cline_databases() == [database]


def test_declared_packages_all_resolve_from_this_source_checkout():
    import dodo_memory
    import dodo_memory_desktop

    import dodo_mem_neural

    root = Path(__file__).resolve().parents[2]
    assert Path(dodo_memory.__file__).resolve() == root / "dodo_memory/__init__.py"
    assert (
        Path(dodo_memory_desktop.__file__).resolve()
        == root / "desktop/src/dodo_memory_desktop/__init__.py"
    )
    assert (
        Path(dodo_mem_neural.__file__).resolve() == root / "neural/src/dodo_mem_neural/__init__.py"
    )
