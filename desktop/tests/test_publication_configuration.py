import pytest

from dodo_memory_desktop.personal_context import PersonalProfile


def test_missing_profile_never_selects_personal_sources(tmp_path):
    assert PersonalProfile(tmp_path / "profile.json").sources() == ()


def test_library_inventory_requires_explicit_local_roots(tmp_path, monkeypatch):
    from dodo_memory_desktop.whole_index_worker import discover_roots

    monkeypatch.delenv("DODO_MEM_LIBRARY_ROOTS", raising=False)
    assert discover_roots() == []
    monkeypatch.setenv("DODO_MEM_LIBRARY_ROOTS", f'["{tmp_path.as_posix()}"]')
    assert discover_roots() == [tmp_path]
    monkeypatch.setenv("DODO_MEM_LIBRARY_ROOTS", '["relative"]')
    with pytest.raises(ValueError, match="absolute local"):
        discover_roots()


def test_runtime_root_cannot_fall_back_to_installed_state(tmp_path, monkeypatch):
    from dodo_memory_desktop.configuration import runtime_path

    monkeypatch.delenv("DODO_MEM_RUNTIME_ROOT", raising=False)
    with pytest.raises(ValueError, match="DODO_MEM_RUNTIME_ROOT"):
        runtime_path("desktop")
    monkeypatch.setenv("DODO_MEM_RUNTIME_ROOT", str(tmp_path))
    assert runtime_path("desktop") == tmp_path / "desktop"


def test_runtime_root_rejects_a_redirect_before_opening_descendants(tmp_path, monkeypatch):
    from dodo_memory_desktop.configuration import runtime_path

    target = tmp_path / "target"
    target.mkdir()
    redirect = tmp_path / "redirect"
    try:
        redirect.symlink_to(target, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"Directory symlinks unavailable: {error}")
    monkeypatch.setenv("DODO_MEM_RUNTIME_ROOT", str(redirect / "missing"))
    with pytest.raises(ValueError, match="redirect"):
        runtime_path("desktop")


def test_runtime_component_cannot_redirect_outside_the_selected_root(tmp_path, monkeypatch):
    from dodo_memory_desktop.configuration import runtime_path

    runtime = tmp_path / "runtime"
    runtime.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    try:
        (runtime / "neural").symlink_to(outside, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"Directory symlinks unavailable: {error}")
    monkeypatch.setenv("DODO_MEM_RUNTIME_ROOT", str(runtime))
    with pytest.raises(ValueError, match="redirect"):
        runtime_path("neural")


def test_profile_reads_only_canonical_sources_inside_the_selected_root(tmp_path, monkeypatch):
    from dodo_memory_desktop.personal_context import read_profile_file

    selected = tmp_path / "selected"
    selected.mkdir()
    sibling = tmp_path / "selected-other"
    sibling.mkdir()
    allowed = selected / "allowed.md"
    allowed.write_text("Explicit synthetic source.")
    outside = sibling / "outside.md"
    outside.write_text("Synthetic private sibling.")
    monkeypatch.setenv("DODO_MEM_LIBRARY_ROOTS", f'["{selected.as_posix()}"]')
    assert read_profile_file(allowed, 100)["text"] == "Explicit synthetic source."
    for path in (outside, selected / ".." / "selected-other/outside.md"):
        with pytest.raises(ValueError):
            read_profile_file(path, 100)
    try:
        (selected / "redirect").symlink_to(sibling, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"Directory symlinks unavailable: {error}")
    with pytest.raises(ValueError, match="redirect"):
        read_profile_file(selected / "redirect/outside.md", 100)


def test_shared_draft_contains_no_automatic_personal_preferences(tmp_path, monkeypatch):
    from dodo_memory_desktop.personal_context import shared_draft

    monkeypatch.setenv("DODO_MEM_RUNTIME_ROOT", str(tmp_path))
    suggestions = tmp_path / "desktop/profile-suggestions"
    suggestions.mkdir(parents=True)
    (suggestions / "Interests.md").write_text("PRIVATE_SYNTHETIC_MATERIAL")
    result = shared_draft()
    assert "PRIVATE_SYNTHETIC_MATERIAL" not in result
    assert "No personal context is shared by default" in result
