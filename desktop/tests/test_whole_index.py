import csv
import os
import shutil
import sqlite3
import subprocess
import sys
import time
from zipfile import ZIP_DEFLATED, ZipFile

import pytest

from dodo_memory_desktop import whole_index_worker
from dodo_memory_desktop.whole_index import Catalogue
from dodo_memory_desktop.whole_index_worker import Indexer


def test_changed_source_is_not_published_from_parallel_extraction(tmp_path, monkeypatch):
    root = tmp_path / "files"
    root.mkdir()
    source = root / "source.md"
    source.write_text("Old concurrent extraction evidence.", encoding="utf-8")
    database = tmp_path / "index.sqlite3"
    index = Indexer(database, extraction_workers=4)
    catalogue = Catalogue(database)
    parse = whole_index_worker.extract

    def change_during_parse(path, raw):
        if path == source:
            path.write_text("New version replaces the original evidence.", encoding="utf-8")
        return parse(path, raw)

    try:
        index.reconcile([root], extract_now=False)
        assert index.can_drain_published([root])
        with monkeypatch.context() as writing:
            writing.setattr(whole_index_worker, "extract", change_during_parse)
            assert index.extract_pending() == 1
        assert catalogue.resolve_path(source)["state"] == "failed"
        assert not catalogue.search("Old concurrent extraction evidence")
        assert not index.can_drain_published([root])

        index.reconcile([root])
        assert catalogue.search("New version replaces")
        assert catalogue.status()["message"] == "Local catalogue ready."
    finally:
        index.close()


def test_extraction_counts_separate_skipped_failed_and_extracted(tmp_path):
    root = tmp_path / "files"
    root.mkdir()
    (root / "note.md").write_text("A searchable note.", encoding="utf-8")
    (root / "opaque.bin").write_bytes(b"\0")
    (root / "broken.docx").write_bytes(b"Not an office archive.")
    database = tmp_path / "index.sqlite3"
    index = Indexer(database)
    try:
        index.reconcile([root])
    finally:
        index.close()

    catalogue = Catalogue(database)
    assert catalogue.extraction_counts() == {
        "extracted": 1,
        "skipped": 1,
        "failed": 1,
        "estimated": False,
    }
    assert catalogue.status()["indexed"] == 1
    assert catalogue.graph_status()["nodes"] == 4


def test_existing_root_refresh_publishes_work_before_it_finishes(tmp_path, monkeypatch):
    root = tmp_path / "files"
    root.mkdir()
    db = tmp_path / "index.sqlite3"
    catalogue = Catalogue(db)
    index = Indexer(db)
    try:
        index.reconcile([root])
        source = root / "note.md"
        source.write_text("Evidence published during the scan.", encoding="utf-8")
        monkeypatch.setattr(whole_index_worker, "INITIAL_INVENTORY_BATCH", 1)
        during_scan = []

        def checkpoint(completed):
            if not completed and catalogue.resolve_path(source):
                assert index.extract_pending() == 1
                assert catalogue.search("Evidence published during the scan")
                during_scan.append(catalogue.status()["inventory_pending"])

        index.reconcile([root], checkpoint=checkpoint)
    finally:
        index.close()
    assert during_scan == [1]
    assert catalogue.status()["message"] == "Local catalogue ready."


def test_watch_subtree_checkpoints_preserve_completed_root_and_drain(tmp_path, monkeypatch):
    root = tmp_path / "files"
    target = root / "watched"
    target.mkdir(parents=True)
    old = target / "old.md"
    old.write_text("Removed in a large watcher update.", encoding="utf-8")
    db = tmp_path / "index.sqlite3"
    catalogue = Catalogue(db)
    index = Indexer(db)
    try:
        index.reconcile([root])
        old.unlink()
        new = target / "new.md"
        new.write_text("Published before the folder refresh ends.", encoding="utf-8")
        monkeypatch.setattr(whole_index_worker, "INITIAL_INVENTORY_BATCH", 1)
        published = []

        def checkpoint(_scope):
            assert catalogue.status()["inventory_finished"] == 1
            assert catalogue.resolve_path(old) is None
            if index.extract_pending():
                assert catalogue.search("Published before the folder refresh ends")
                published.append(True)

        index.refresh_paths([target], [root], checkpoint=checkpoint)
        assert published == [True]
        assert catalogue.resolve_path(new)["state"] == "indexed"
    finally:
        index.close()


def test_watcher_prioritises_changed_file_over_historical_backlog(tmp_path):
    root = tmp_path / "files"
    root.mkdir()
    old = root / "historical.txt"
    old.write_text("Historical pending evidence.", encoding="utf-8")
    db = tmp_path / "index.sqlite3"
    catalogue = Catalogue(db)
    index = Indexer(db)
    try:
        index.reconcile([root], extract_now=False)
        fresh = root / "fresh.py"
        fresh.write_text('data = "Fresh watcher priority evidence"\n', encoding="utf-8")
        assert not index.watch_path_allowed(root / "AppData" / "Cache" / "churn.log", root)
        assert index.watch_path_allowed(root / ".git" / "refs" / "heads" / "main", root)

        index.refresh_paths([fresh], [root], checkpoint=lambda _scope: index.extract_watched())

        assert catalogue.search("Fresh watcher priority evidence")
        assert catalogue.resolve_path(old)["state"] == "pending"
    finally:
        index.close()


def test_paused_extraction_resumes_to_published_graph(tmp_path, monkeypatch):
    root = tmp_path / "files"
    root.mkdir()
    (root / "note.md").write_text("[Target](target.md)", encoding="utf-8")
    (root / "target.md").write_text("Verified destination.", encoding="utf-8")
    db = tmp_path / "index.sqlite3"
    catalogue = Catalogue(db)
    index = Indexer(db)
    try:
        with monkeypatch.context() as full:
            full.setattr(whole_index_worker, "MAX_INDEX_BYTES", 1)
            index.reconcile([root])
        paused = catalogue.status()
        assert paused["content_pending"] == 2
        assert paused["message"] == "Content extraction paused at the local disk budget."
        assert not catalogue.search("Verified destination")

        index.reconcile([root])
        ready = catalogue.status()
        assert ready["content_pending"] == 0
        assert ready["message"] == "Local catalogue ready."
        assert any(edge["kind"] == "reference" for edge in catalogue.projection()["edges"])
    finally:
        index.close()


def test_filename_budget_pause_blocks_ready_without_content_queue(tmp_path, monkeypatch):
    root = tmp_path / "files"
    root.mkdir()
    (root / "inventory-only.bin").write_bytes(b"\0")
    database = tmp_path / "index.sqlite3"
    catalogue = Catalogue(database)
    index = Indexer(database)
    try:
        with monkeypatch.context() as full:
            full.setattr(whole_index_worker, "MAX_INDEX_BYTES", 1)
            index.reconcile([root])
        assert catalogue.status()["content_pending"] == 0
        assert catalogue.status()["message"] == "Filename indexing paused at the local disk budget."

        index.reconcile([root])
        assert catalogue.status()["message"] == "Local catalogue ready."
        assert catalogue.search("inventory-only.bin")
    finally:
        index.close()


def test_inventory_links_and_version_bound_evidence(tmp_path):
    root = tmp_path / "files"
    nested = root / "nested"
    nested.mkdir(parents=True)
    (nested / "empty").mkdir()
    report = nested / "report.md"
    report.write_text("# Budget\n\nA unique local evidence phrase.\n", encoding="utf-8")
    (root / "notes.md").write_text("[report](nested/report.md)", encoding="utf-8")
    (root / ".env").write_text("AUTH_TOKEN=never-index-this-secret", encoding="utf-8")
    with ZipFile(root / "report.docx", "w", compression=ZIP_DEFLATED) as archive:
        archive.writestr("word/document.xml", '<doc><t>password="compressed-only-secret"</t></doc>')
    (root / "opaque.bin").write_bytes(b"\x01\x00\x02")
    database = tmp_path / "index.sqlite3"
    index = Indexer(database)
    index.reconcile([root])
    catalogue = Catalogue(database)
    assert catalogue.status()["indexed"] == 2
    overview = catalogue.projection()
    assert overview["total"] == 5
    assert any(link["kind"] == "reference" for link in overview["edges"])
    hit = catalogue.search("unique local evidence")[0]
    assert hit["path"] == str(report)
    evidence = catalogue.read_evidence(hit["id"])
    assert evidence["digest"] == hit["digest"]
    assert evidence["locator"]["kind"] == "lines"
    assert "unique local evidence" in evidence["text"]
    assert catalogue.resolve_path(nested / "empty")["kind"] == "folder"
    assert catalogue.resolve_path(root / "opaque.bin")["state"] == "metadata"
    credential = catalogue.resolve_path(root / ".env")
    assert credential["state"] == "protected"
    assert catalogue.read_evidence(credential["id"]) is None
    assert not catalogue.search("never-index-this-secret")
    assert catalogue.resolve_path(root / "report.docx")["state"] == "protected"
    assert not catalogue.search("compressed-only-secret")
    assert any(
        link["kind"] == "reference" and link["node"]["name"] == "notes.md"
        for link in catalogue.neighbours(hit["id"])
    )
    report.write_text("A replacement fact.\n", encoding="utf-8")
    assert catalogue.read_evidence(hit["id"]) is None
    index.reconcile([root])
    assert not catalogue.search("unique local evidence")
    assert catalogue.search("replacement fact")[0]["id"] == hit["id"]


def test_offline_root_is_retained_and_evidence_is_blocked(tmp_path):
    root = tmp_path / "drive"
    root.mkdir()
    (root / "note.md").write_text("searchable evidence", encoding="utf-8")
    database = tmp_path / "index.sqlite3"
    index = Indexer(database)
    index.reconcile([root])
    catalogue = Catalogue(database)
    hit = catalogue.search("searchable")[0]
    root.rename(tmp_path / "detached")
    index.reconcile([root])
    assert catalogue.roots()[0]["state"] == "offline"
    assert catalogue.node(hit["id"])["state"] == "offline"
    assert catalogue.read_evidence(hit["id"]) is None
    assert not catalogue.search("searchable")


def test_wikilinks_resolve_unique_names_and_titles_across_roots(tmp_path):
    notes, reports = tmp_path / "notes", tmp_path / "reports"
    notes.mkdir()
    reports.mkdir()
    source = notes / "source.md"
    source.write_text(
        "# Starting point\n"
        "[[calibration review#Meter records|review]]\n"
        "[[readme]]\n"
        "[[Shared]]\n"
        "[[./Shared#Decision]]\n",
        encoding="utf-8",
    )
    local_shared = notes / "Shared.md"
    local_shared.write_text("# Decision\nThe local note.", encoding="utf-8")
    (reports / "Shared.md").write_text("# Other note", encoding="utf-8")
    title_match = reports / "review-2026.markdown"
    title_match.write_text("# Calibration Review\n## Meter records", encoding="utf-8")
    filename_match = reports / "README.md"
    filename_match.write_text("Notes without a title.", encoding="utf-8")
    database = tmp_path / "catalogue.sqlite"
    index = Indexer(database)
    catalogue = Catalogue(database)
    try:
        index.reconcile([notes, reports])
        source_id = catalogue.resolve_path(source)["id"]

        def references():
            return {
                row["node"]["path"]: row
                for row in catalogue.neighbours(source_id)
                if row["kind"] == "reference"
            }

        links = references()
        assert set(links) == {str(title_match), str(filename_match), str(local_shared)}
        assert "on line 5" in links[str(local_shared)]["evidence"]
        link = links[str(title_match)]
        assert "Requested heading Meter records" in link["evidence"]
        evidence = catalogue.graph_evidence(link)
        assert evidence[0]["source"]["locator"]["start"] == 2
        assert evidence[0]["source"]["text"].startswith("[[calibration review")

        duplicate_title = notes / "conflicting-review.md"
        duplicate_title.write_text("# Calibration Review\nA different document.", encoding="utf-8")
        index.reconcile([notes, reports])
        assert set(references()) == {str(filename_match), str(local_shared)}
    finally:
        index.close()


def test_nested_scope_survives_parent_reconciliation_and_is_removed_when_revoked(
    tmp_path,
    monkeypatch,
):
    root = tmp_path / "profile"
    history = root / "history"
    history.mkdir(parents=True)
    session = history / "session.md"
    session.write_text("A separately scoped conversation.", encoding="utf-8")
    (root / "reference.md").write_text(f"[Conversation]({session.as_uri()})", encoding="utf-8")
    monkeypatch.setattr(whole_index_worker, "history_root_paths", lambda: (history,))
    database = tmp_path / "index.sqlite3"
    index = Indexer(database)
    catalogue = Catalogue(database)
    index.reconcile([root, history])
    hit = catalogue.search("separately scoped conversation")[0]
    assert any(edge["kind"] == "reference" for edge in catalogue.neighbours(hit["id"]))
    index._inventory(root, root)
    assert catalogue.read_evidence(hit["id"]) is not None
    index.reconcile([root])
    index.close()
    assert catalogue.read_evidence(hit["id"]) is None
    assert not catalogue.search("separately scoped conversation")


def test_initial_inventory_publishes_coherent_counts_before_finishing(tmp_path, monkeypatch):
    root = tmp_path / "files"
    first = root / "first"
    final = first / "nested"
    final.mkdir(parents=True)
    (final / "note.md").write_text("A local note.", encoding="utf-8")
    database = tmp_path / "index.sqlite3"
    catalogue = Catalogue(database)
    original = Indexer._directory_entries
    seen = []

    def observe(index, path, source_root):
        if path == final:
            seen.append(catalogue.roots()[0]["descendants"])
            assert catalogue.resolve_path(first)["descendants"] == 0
            assert catalogue.resolve_path(final) is None
        return original(index, path, source_root)

    monkeypatch.setattr(whole_index_worker, "INITIAL_INVENTORY_BATCH", 2)
    monkeypatch.setattr(Indexer, "_directory_entries", observe)
    index = Indexer(database)
    index.reconcile([root])
    index.close()
    assert seen == [1]
    assert catalogue.roots()[0]["descendants"] == 3


@pytest.mark.parametrize("legacy_catalogue", [False, True])
def test_interrupted_inventory_republishes_and_checks_other_roots(
    tmp_path, monkeypatch, legacy_catalogue
):
    root = tmp_path / "profile"
    desktop = root / "Desktop"
    desktop.mkdir(parents=True)
    gone = desktop / "old.md"
    gone.write_text("Deleted during interruption.", encoding="utf-8")
    slow = root / "zz-other-folder"
    slow.mkdir()
    (slow / "later.md").write_text("Checked after Desktop.", encoding="utf-8")
    drive = tmp_path / "data-drive"
    other = drive / "folder"
    other.mkdir(parents=True)
    (other / "data.md").write_text("Other root content.", encoding="utf-8")
    database = tmp_path / "index.sqlite3"
    catalogue = Catalogue(database)
    publish = Indexer._publish_initial_batch

    def interrupt(index, scope, folders, visited):
        publish(index, scope, folders, visited)
        if scope == drive:
            raise RuntimeError("Fixture interruption after commit")

    monkeypatch.setattr(whole_index_worker, "INITIAL_INVENTORY_BATCH", 3)
    with monkeypatch.context() as interrupted:
        interrupted.setattr(Indexer, "_publish_initial_batch", interrupt)
        index = Indexer(database)
        index.desktop = desktop
        with pytest.raises(RuntimeError, match="Fixture interruption"):
            index.reconcile([root, drive])
        index.close()

    assert catalogue.resolve_path(gone)
    assert catalogue.resolve_path(slow) is None
    assert catalogue.resolve_path(other)
    checkpoint = catalogue.status()
    assert checkpoint["inventory_pending"] == 2
    assert checkpoint["inventory_visited"] == 6
    assert checkpoint["content_pending"] == 2
    if legacy_catalogue:
        # Older workers wrote the same rows, but lacked a completion marker.
        db = sqlite3.connect(database)
        try:
            db.execute("DROP TABLE inventory_roots")
            db.commit()
        finally:
            db.close()
        assert catalogue.status()["inventory_pending"] == 0
    gone.unlink()
    new = desktop / "new.md"
    new.write_text("Recovered searchable fixture.", encoding="utf-8")

    resumed = []

    def observe(index, scope, folders, visited):
        publish(index, scope, folders, visited)
        if scope == root:
            status = catalogue.status()
            assert status["generation"] > checkpoint["generation"]
            assert status["inventory_pending"] == 2
            assert catalogue.resolve_path(gone) is None
            assert catalogue.resolve_path(new)
            assert catalogue.resolve_path(desktop)["descendants"] == 1
            resumed.append(scope)

    monkeypatch.setattr(Indexer, "_publish_initial_batch", observe)
    index = Indexer(database)
    index.desktop = desktop
    index.reconcile([root, drive])
    index.close()
    assert resumed == [root]
    assert catalogue.status()["inventory_finished"] == 2
    assert catalogue.status()["content_pending"] == 0
    assert catalogue.search("Recovered searchable fixture")


def test_pdf_parser_reads_real_page_and_reports_encryption(tmp_path):
    pypdf = pytest.importorskip("pypdf")
    generic = pypdf.generic
    root = tmp_path / "files"
    root.mkdir()
    writer = pypdf.PdfWriter()
    page = writer.add_blank_page(width=300, height=200)
    font = generic.DictionaryObject(
        {
            generic.NameObject("/Type"): generic.NameObject("/Font"),
            generic.NameObject("/Subtype"): generic.NameObject("/Type1"),
            generic.NameObject("/BaseFont"): generic.NameObject("/Helvetica"),
        }
    )
    page[generic.NameObject("/Resources")] = generic.DictionaryObject(
        {
            generic.NameObject("/Font"): generic.DictionaryObject(
                {generic.NameObject("/F1"): font}
            ),
        }
    )
    stream = generic.DecodedStreamObject()
    stream.set_data(b"BT /F1 12 Tf 20 20 Td (Printed proof text) Tj ET")
    page[generic.NameObject("/Contents")] = writer._add_object(stream.flate_encode())
    writer.write(root / "readable.pdf")
    writer.encrypt("test-document-password")
    writer.write(root / "encrypted.pdf")

    database = tmp_path / "index.sqlite3"
    index = Indexer(database)
    index.reconcile([root])
    index.close()
    catalogue = Catalogue(database)
    hit = catalogue.search("printed proof text")[0]
    assert catalogue.read_evidence(hit["id"])["locator"] == {"kind": "page", "page": 1}
    encrypted = catalogue.resolve_path(root / "encrypted.pdf")
    assert encrypted["state"] == "encrypted"
    assert encrypted["reason"] == "Document is encrypted."


@pytest.mark.skipif(not shutil.which("git"), reason="Git CLI not installed")
def test_git_merge_and_changed_files_are_actual_history(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()

    def git(*args):
        return subprocess.run(
            [
                "git",
                "-c",
                "user.name=Test",
                "-c",
                "user.email=239468521+diogommpires-glitch@users.noreply.github.com",
                *args,
            ],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    git("init", "-b", "main")
    (repo / "decision.md").write_text("The reviewed implementation decision.", encoding="utf-8")
    git("add", "decision.md")
    git("commit", "-m", "Record decision")
    git("switch", "-c", "topic")
    implementation = repo / "engine.md"
    implementation.write_text("[decision](decision.md)", encoding="utf-8")
    git("add", "engine.md")
    git("commit", "-m", "Implement verified join")
    revision = git("rev-parse", "HEAD")
    git("switch", "main")
    git("merge", "--no-ff", "-m", "Merge topic", "topic")
    merge = git("rev-parse", "HEAD")
    git("switch", "-c", "private-subject")
    git("commit", "--allow-empty", "-m", 'password="synthetic-git-credential"')
    protected_commit = git("rev-parse", "HEAD")
    git("switch", "main")

    database = tmp_path / "index.sqlite3"
    index = Indexer(database)
    index.reconcile([repo])
    catalogue = Catalogue(database)
    graph = catalogue.git_history(catalogue.roots()[0]["id"])
    commits = {node["digest"]: node for node in graph["nodes"]}
    assert len([edge for edge in graph["edges"] if edge["source"] == commits[merge]["id"]]) == 2
    assert any("main" in ref for ref in commits[merge]["refs"])
    assert "synthetic" not in commits[protected_commit]["name"]
    assert not catalogue.search("synthetic git credential")
    assert commits[revision]["changed_files"] == ["engine.md"]
    linked = catalogue.neighbours(commits[revision]["id"])
    assert any(
        edge["kind"] == "changed_file" and edge["node"]["path"] == str(implementation)
        for edge in linked
    )
    proof = catalogue.read_evidence(commits[revision]["id"])
    assert proof["locator"] == {"kind": "commit", "commit": revision}
    implementation.write_text("An unrelated working tree edit.", encoding="utf-8")
    index.refresh_paths([implementation], [repo])
    index.extract_pending()
    index.refresh_edges()
    assert any(
        edge["kind"] == "changed_file" for edge in catalogue.neighbours(commits[revision]["id"])
    )
    index.close()


@pytest.mark.skipif(os.name != "nt", reason="Native Windows watcher")
def test_worker_updates_renames_and_deletions_without_rescan(tmp_path):
    root = tmp_path / "local"
    root.mkdir()
    (root / "navigation.md").write_text("[Target](destination.md)", encoding="utf-8")
    (root / "destination.md").write_text("A final linked destination.", encoding="utf-8")
    database = tmp_path / "index.sqlite3"
    worker = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "dodo_memory_desktop.whole_index_worker",
            "--database",
            str(database),
            "--root",
            str(root),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    catalogue = Catalogue(database)

    def until(condition):
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline and worker.poll() is None:
            if condition():
                return
            time.sleep(0.05)
        pytest.fail("The watcher did not publish the expected change")

    try:
        until(lambda: catalogue.status()["message"] == "Local catalogue ready.")
        assert catalogue.status()["content_pending"] == 0
        assert any(edge["kind"] == "reference" for edge in catalogue.projection()["edges"])
        nested = root / "new"
        nested.mkdir()
        first = nested / "source.md"
        first.write_text("A unique native watcher token.", encoding="utf-8")
        until(lambda: catalogue.search("native watcher token"))
        original = first.stat()
        first.write_text("A unique native watcher alias.", encoding="utf-8")
        os.utime(first, ns=(original.st_atime_ns, original.st_mtime_ns))
        until(lambda: catalogue.search("native watcher alias"))
        assert not catalogue.search("native watcher token")
        second = nested / "renamed.md"
        first.rename(second)
        until(lambda: catalogue.resolve_path(first) is None and catalogue.resolve_path(second))
        until(
            lambda: (
                catalogue.search("native watcher alias")
                and catalogue.search("native watcher alias")[0]["path"] == str(second)
            )
        )
        identity = catalogue.resolve_path(second)["id"]
        _user, sid = next(
            csv.reader(
                subprocess.check_output(
                    ["whoami", "/user", "/fo", "csv", "/nh"],
                    text=True,
                ).splitlines()
            )
        )
        subprocess.run(
            ["icacls", str(second), "/deny", f"*{sid}:(RD)"], check=True, capture_output=True
        )
        try:
            until(lambda: catalogue.resolve_path(second)["state"] == "protected")
            assert catalogue.read_evidence(identity) is None
            assert not catalogue.search("native watcher alias")
        finally:
            subprocess.run(
                ["icacls", str(second), "/remove:d", f"*{sid}"], check=True, capture_output=True
            )
        until(lambda: catalogue.search("native watcher alias"))
        second.unlink()
        until(lambda: catalogue.resolve_path(second) is None)
        assert not catalogue.search("native watcher alias")
    finally:
        worker.terminate()
        worker.wait(timeout=8)
