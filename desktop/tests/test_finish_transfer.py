from dodo_memory_desktop.whole_index import Catalogue
from dodo_memory_desktop.whole_index_worker import Indexer
from tools.finish_transfer import app_qualified, catalogue_completion


def test_catalogue_gate_requires_published_inventory_and_graph(tmp_path):
    root = tmp_path / "files"
    root.mkdir()
    (root / "note.md").write_text("[Source](source.md)", encoding="utf-8")
    (root / "source.md").write_text("Source evidence.", encoding="utf-8")
    database = tmp_path / "index.sqlite3"
    index = Indexer(database)
    try:
        index.reconcile([root], extract_now=False)
        catalogue = Catalogue(database)
        assert catalogue_completion(catalogue) is None
        index.reconcile([root])
        receipt = catalogue_completion(catalogue)
        assert receipt["catalogue"]["content_pending"] == 0
        assert receipt["graph"]["containment"] == 2
        assert receipt["graph"]["bundled"] >= 1
    finally:
        index.close()


def test_shutdown_requires_native_qualification_and_rejects_slow_search():
    timing = {"samples": 12, "p95_ms": 33}
    receipt = {
        "qt_platform": "offscreen",
        "shortcut_verified": True,
        "same_application_at_start_and_end": True,
        "launcher_probe_verified": True,
        "live_updates": {
            "pass": True,
            "stale_source_denied": True,
            "created_ms": 1000,
            "changed_ms": 2000,
            "removed_ms": 3000,
        },
        "results": {
            "search": {
                "pass": True,
                "source_tasks": 12,
                "verified_source_tasks": 11,
                "filename_index_ready_during_both_passes": True,
                "first_pass": timing,
                "repeat_pass": dict(timing),
            },
            "graph": {"pass": True, "wheel_input_to_paint": timing, "cached_expansion": timing},
            "updates": {
                "pass": True,
                "created_ms": 1000,
                "changed_ms": 2000,
                "removed_ms": 3000,
                "stale_source_denied": True,
                "revoked_source_denied": True,
                "protected_source_denied": True,
                "reference_source_verified": True,
                "restart_source_verified": True,
            },
        },
    }
    assert not app_qualified(receipt)
    receipt["qt_platform"] = "windows"
    assert app_qualified(receipt)
    receipt["results"]["search"]["repeat_pass"]["p95_ms"] = 301
    assert not app_qualified(receipt)
