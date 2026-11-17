"""Discover and refresh personal file evidence without publishing to the shared vault."""

import argparse
import hashlib
import json
import os
import shutil
import stat
import subprocess
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from urllib.parse import unquote, urlsplit
from xml.etree import ElementTree
from zipfile import BadZipFile

from .whole_index import (
    GRAPH_BLOCKED,
    MAX_SOURCE_BYTES,
    advance_name_index,
    default_database,
    entry_state,
    history_root_paths,
    history_source,
    metadata_only_runtime,
    path_id,
    path_key,
    protected,
    read_bytes,
    safe_path,
    writable_database,
)
from .whole_index_extract import (
    SUPPORTED,
    WIKI_DESTINATION,
    EncryptedDocument,
    account_bundle,
    category,
    extract,
    links_and_topics,
    normalise_note_title,
)
from .whole_index_git import history, repository, revision_id

SYSTEM = frozenset(
    {
        "windows",
        "program files",
        "program files (x86)",
        "programdata",
        "recovery",
        "system volume information",
        "$recycle.bin",
        "users",
        "documents and settings",
        "perflogs",
        "$windows.~bt",
        "windows.old",
        "$winreagent",
        "msocache",
        "boot",
        "onedrivetemp",
        "riot games",
        "temp",
    }
)
INTERNAL = frozenset(
    {
        "appdata",
        ".cache",
        "__pycache__",
        ".npm",
        ".vscode",
        ".cursor",
        ".local-ai-cache",
        ".incoming",
        ".pytest_cache",
        ".ruff_cache",
        ".mypy_cache",
    }
)
GENERATED = frozenset({"node_modules", ".venv", "site-packages", ".tox", ".git", "dist"})
PROJECT_MARKERS = frozenset(
    {
        ".git",
        "pyproject.toml",
        "package.json",
        "cargo.toml",
        "go.mod",
        "pom.xml",
        "cmakelists.txt",
    }
)
RESCAN_SECONDS = 300
RECONNECT_SECONDS = 10
DEBOUNCE_SECONDS = 2
LINK_REFRESH_SECONDS = 120
NAME_INDEX_SECONDS = 0.25
INITIAL_INVENTORY_BATCH = 4096
WATCH_EVENT_BATCH = 64
WATCH_EXTRACTION_BATCH = 16
WATCH_EXTRACTION_SECONDS = 0.5
WATCH_URGENT_LIMIT = 256
EARLY_NOTE_BATCH = 12
EARLY_NOTE_SECONDS = 0.5
EXTRACTION_BATCH = 128
EXTRACTION_WORKERS = 8
MIN_DISK_FREE = 20 * 1024**3
MAX_INDEX_BYTES = 12 * 1024**3
WAL_RETAIN_BYTES = 64 * 1024**2


def discover_roots() -> list[Path]:
    from .configuration import source_roots

    return list(source_roots("DODO_MEM_LIBRARY_ROOTS"))


def tree_bounds(path: Path) -> tuple[str, str]:
    prefix = path_key(path).rstrip(os.sep) + os.sep
    return prefix, prefix + "\U0010ffff"


def file_id(info):
    return f"{info.st_dev}:{info.st_ino}"


def sqlite_integer(value):
    # A non-numeric prefix prevents SQLite's INTEGER affinity rounding a large value to REAL.
    return value if -(1 << 63) <= value < 1 << 63 else f"int:{value}"


def file_version(info):
    return sqlite_integer(info.st_size), sqlite_integer(info.st_mtime_ns), file_id(info)


def file_state(path, info):
    if state := entry_state(info):
        return state
    if protected(path):
        return "protected"
    if not stat.S_ISREG(info.st_mode):
        return "metadata"
    if info.st_size > MAX_SOURCE_BYTES:
        return "oversized"
    if (path.suffix.casefold() not in SUPPORTED and not account_bundle(path)) or any(
        part.casefold() in GENERATED for part in path.parts[:-1]
    ):
        return "metadata"
    return "pending"


@dataclass(frozen=True, slots=True)
class FileEvidence:
    state: str
    digest: str | None = None
    passages: tuple[tuple[str, int, int, str], ...] = ()
    links: tuple[tuple[str, int], ...] = ()
    topics: tuple[tuple[str, int], ...] = ()
    titles: tuple[str, ...] = ()


class Indexer:
    def __init__(self, database: Path | None = None, *, extraction_workers=EXTRACTION_WORKERS):
        self.database = Path(database or default_database()).absolute()
        self.db = writable_database(self.database)
        self.db.execute(f"PRAGMA journal_size_limit={WAL_RETAIN_BYTES}")
        self.private_folder = default_database().parent
        self.database_files = {
            self.database,
            self.database.with_name(self.database.name + "-wal"),
            self.database.with_name(self.database.name + "-shm"),
        }
        self.history_roots = frozenset(history_root_paths())
        self.desktop = Path.home() / "Desktop"
        if os.name == "nt":
            from .whole_index_windows import desktop_directory

            self.desktop = desktop_directory() or self.desktop
        self.dirty_repos: set[str] = set()
        self.links_dirty = True
        self.extraction_paused = False
        self.extraction_workers = extraction_workers
        self.parsers = ThreadPoolExecutor(max_workers=extraction_workers)
        self.pdf_slots = threading.BoundedSemaphore(2)
        self.watched_files = {}

    def close(self):
        self.parsers.shutdown()
        self.db.close()

    def _publish(self, message, scanning=False):
        self.db.execute(
            """
            UPDATE catalogue SET generation=generation+1,scanning=?,message=?
        """,
            (int(scanning), message),
        )

    def reconcile(self, roots: list[Path], *, extract_now: bool = True, checkpoint=None):
        roots = list(dict.fromkeys(Path(root).absolute() for root in roots))
        configured = {path_id(root) for root in roots}
        with self.db:
            previous = self.db.execute("""
                SELECT id FROM nodes WHERE parent_id IS NULL AND kind='root'
                UNION SELECT id FROM inventory_roots
            """)
            retired = [(identity,) for (identity,) in previous if identity not in configured]
            self.db.executemany("DELETE FROM nodes WHERE root_id=?", retired)
            self.db.executemany("DELETE FROM inventory_roots WHERE id=?", retired)
            self.db.executemany(
                "INSERT OR IGNORE INTO inventory_roots (id,phase) VALUES (?,'pending')",
                ((identity,) for identity in configured),
            )
            self.links_dirty |= bool(retired)
            self._publish("Inventorying personal folders.", True)
        phases = dict(self.db.execute("SELECT id,phase FROM inventory_roots"))
        ordered = sorted(
            roots,
            key=lambda root: (root != Path("E:/"), not self.desktop.is_relative_to(root)),
        )
        pending = deque(
            (root, self._inventory_steps(root, root))
            for root in ordered
            if phases[path_id(root)] != "complete"
        )
        completed = set()

        def publish_early_links(token):
            if checkpoint is None:
                return
            checkpoint(completed)
            if token is not None:
                self.extract_pending(
                    EARLY_NOTE_BATCH,
                    EARLY_NOTE_SECONDS,
                    publish_references=True,
                    inventory_token=token,
                )

        # Separate roots take turns committing batches, so a large profile never hides
        # the beginning of the E: inventory. Completed roots retain an atomic rescan.
        while pending:
            root, step = pending.popleft()
            continuing, token = next(step, (False, None))
            if continuing:
                pending.append((root, step))
            else:
                completed.add(root)
            publish_early_links(token)
        for root in (root for root in ordered if phases[path_id(root)] == "complete"):
            for continuing, token in self._inventory_steps(
                root, root, publish_batches=checkpoint is not None
            ):
                if not continuing:
                    completed.add(root)
                publish_early_links(token)
        with self.db:
            self._publish("Extracting searchable evidence.", True)
        if extract_now:
            while self.extract_pending(EXTRACTION_BATCH):
                pass
            self.refresh_git()
            self.refresh_edges()
            names = self.advance_names()
            while names is False:
                names = self.advance_names()
            if names:
                self._publish_ready()

    def _inventory(self, target: Path, root: Path, *, changed: set[Path] | None = None):
        token = None
        for _, token in self._inventory_steps(target, root, changed=changed):
            pass
        return token

    def _inventory_steps(self, target: Path, root: Path, *, changed=None, publish_batches=False):
        token = time.time_ns()
        root_id = path_id(root)
        changed = changed or set()
        phase = self.db.execute(
            "SELECT phase FROM inventory_roots WHERE id=?", (root_id,)
        ).fetchone()
        initial = target == root and (
            publish_batches or phase is None or phase["phase"] != "complete"
        )
        incremental = initial or publish_batches
        rechecking = (
            incremental
            and self.db.execute("SELECT 1 FROM nodes WHERE id=?", (path_id(target),)).fetchone()
            is not None
        )
        if target == root and not initial:
            # An ordinary refresh stays atomic. An interruption makes its next attempt
            # publish progressively like an unfinished first scan.
            with self.db:
                self._inventory_phase(root_id, "pending")
                self._publish("Checking an existing library root.", True)
        try:
            safe_path(target.parent)
            info = target.lstat()
        except FileNotFoundError:
            with self.db:
                if target == root:
                    self._root_state(root, "offline", token)
                    self._inventory_phase(root_id, "offline")
                else:
                    self._remove_tree(target, root_id)
                    self._recount_ancestors(target.parent, root)
                    self._publish("Updating the local catalogue.", True)
            return
        except OSError:
            with self.db:
                if target == root:
                    self._root_state(root, "inaccessible", token)
                    self._inventory_phase(root_id, "offline")
                else:
                    self._inaccessible(target, root_id)
                    self._publish("Updating the local catalogue.", True)
            return
        with self.db:
            if target == root:
                self._root_state(root, "online", token)
            if initial:
                # Keep the coherent published rows, but recheck the file system after a
                # restart. Reusing a stale traversal cursor could miss changes while down.
                self._inventory_phase(root_id, "pending")
            parent = (
                None
                if root == target
                else self.db.execute(
                    "SELECT id,project_id FROM nodes WHERE path_key=?", (path_key(target.parent),)
                ).fetchone()
            )
            if target != root and parent is None:
                return
            scope = (parent["id"], parent["project_id"]) if parent else (None, None)
            pending = [(target, info, *scope)]
            finished = []
            recent_folders = set()
            batch_size = 0
            visited = 0
            while pending:
                path, info, parent_id, project_id = pending.pop()
                if self._skip(path, root):
                    continue
                identity = path_id(path)
                children = []
                state = entry_state(info)
                directory = stat.S_ISDIR(info.st_mode)
                kind = "root" if path == root else "folder" if directory else "file"
                if (
                    state is None
                    and directory
                    and protected(path)
                    and not metadata_only_runtime(path)
                    and root not in self.history_roots
                ):
                    state = "protected"
                if state is None and directory:
                    try:
                        children = self._directory_entries(path, root)
                        if rechecking and self._retire_missing_children(
                            path, identity, root_id, children
                        ):
                            recent_folders.add(path)
                        if any(
                            entry.name.casefold() in PROJECT_MARKERS for entry, _ in children
                        ) and not any(part.casefold() in GENERATED for part in path.parts):
                            project_id = identity
                            if path != root:
                                kind = "project"
                            if any(entry.name.casefold() == ".git" for entry, _ in children):
                                self.dirty_repos.add(identity)
                        state = "online"
                    except OSError:
                        state = "inaccessible"
                        self._inaccessible(path, root_id, token)
                if state is None:
                    state = file_state(path, info)
                self._entry(
                    path,
                    identity,
                    root_id,
                    parent_id,
                    project_id,
                    kind,
                    state,
                    info,
                    token,
                    force=path in changed,
                )
                if directory:
                    finished.append(path)
                if children and path != self.desktop and self.desktop.is_relative_to(path):
                    desktop_branch = path / self.desktop.relative_to(path).parts[0]
                    children.sort(key=lambda entry: entry[0] == desktop_branch)
                pending.extend(
                    (child, child_info, identity, project_id) for child, child_info in children
                )
                if incremental:
                    batch_size += 1
                    visited += 1
                    if path != root:
                        recent_folders.add(path.parent)
                    if batch_size >= INITIAL_INVENTORY_BATCH:
                        if target == root:
                            self._publish_initial_batch(root, recent_folders, visited)
                        else:
                            self._publish_initial_batch(root, recent_folders, visited, subtree=True)
                        recent_folders.clear()
                        batch_size = 0
                        yield True, token
            low, high = tree_bounds(target)
            removed = self.db.execute(
                """
                DELETE FROM nodes WHERE root_id=? AND path_key BETWEEN ? AND ?
                AND seen<>? AND kind<>'commit'
            """,
                (root_id, low, high, token),
            ).rowcount
            if removed:
                self._discard_orphan_commits()
            self.links_dirty |= bool(removed)
            for folder in reversed(finished):
                self._recount(folder)
            if target != root:
                self._recount_ancestors(target.parent, root)
            else:
                state = self.db.execute(
                    "SELECT state FROM nodes WHERE id=?", (root_id,)
                ).fetchone()[0]
                self._inventory_phase(
                    root_id, "complete" if state == "online" else "offline", visited
                )
            self._publish("Updating the local catalogue.", True)
        yield False, token

    def _inventory_phase(self, root_id, phase, visited=0):
        self.db.execute(
            """
            INSERT INTO inventory_roots VALUES (?,?,?)
            ON CONFLICT(id) DO UPDATE SET phase=excluded.phase,visited=excluded.visited
            """,
            (root_id, phase, visited),
        )

    def _publish_initial_batch(self, root, folders, visited, *, subtree=False):
        # Each checkpoint counts exactly the rows visible at that revision. The final
        # sweep retires unseen paths after an interrupted scan and fixes their counts.
        parents = set(folders)
        for folder in folders:
            parents.update(parent for parent in folder.parents if parent.is_relative_to(root))
        for folder in sorted(parents, key=lambda path: len(path.parts), reverse=True):
            self._recount(folder)
        if subtree:
            self._publish("Updating a catalogue folder.", True)
        else:
            self._inventory_phase(path_id(root), "pending", visited)
            self._publish("Inventorying personal folders.", True)
        self.db.commit()

    def _skip(self, path, root):
        if metadata_only_runtime(path):
            return False
        if path == root:
            return False
        if path in self.history_roots:
            return True
        if (
            root == Path("E:/")
            and path.parent == root
            and (path.name.casefold() in SYSTEM or path.name.startswith("$"))
        ):
            return True
        return (
            path.name.casefold() in INTERNAL
            or path in self.database_files
            or path.is_relative_to(self.private_folder)
        )

    def _directory_entries(self, path, root):
        if state := entry_state(path.lstat()):
            raise OSError(f"Directory became {state}")
        children = []
        with os.scandir(path) as entries:
            for entry in entries:
                child = Path(entry.path)
                if self._skip(child, root):
                    continue
                try:
                    info = entry.stat(follow_symlinks=False)
                    if os.name == "nt" and file_state(child, info) == "pending":
                        # Windows DirEntry.stat does not provide st_dev or st_ino.
                        info = child.lstat()
                    children.append((child, info))
                except FileNotFoundError:
                    continue
                except OSError:
                    self._inaccessible(child, path_id(root))
        return children

    def _root_state(self, root, state, token):
        identity = path_id(root)
        old = self.db.execute("SELECT state FROM nodes WHERE id=?", (identity,)).fetchone()
        if old is None or old["state"] != state:
            self.links_dirty = True
        self.db.execute(
            """
            INSERT INTO nodes (id,parent_id,root_id,path,path_key,name,kind,category,state,size,mtime_ns,
                file_id,seen)
            VALUES (?,NULL,?,?,?,?,'root','folder',?,0,0,'',?)
            ON CONFLICT(id) DO UPDATE SET state=excluded.state,seen=excluded.seen
        """,
            (identity, identity, str(root), path_key(root), root.name or str(root), state, token),
        )
        self._publish("Updating root availability.", True)

    def _entry(
        self, path, identity, root_id, parent_id, project_id, kind, state, info, token, *, force
    ):
        previous = self.db.execute(
            """
            SELECT state,size,mtime_ns,file_id,digest,descendants FROM nodes WHERE id=?
        """,
            (identity,),
        ).fetchone()
        digest = None
        version = file_version(info)
        if (
            not force
            and previous
            and state == "pending"
            and previous["state"] != "inaccessible"
            and version == (previous["size"], previous["mtime_ns"], previous["file_id"])
        ):
            state, digest = previous["state"], previous["digest"]
        elif previous and (previous["digest"] or previous["state"] == "indexed"):
            self._clear_evidence(identity)
        if (
            previous
            and previous["descendants"]
            and (kind == "file" or state in {"protected", "redirected", "placeholder"})
        ):
            self.db.execute(
                "DELETE FROM nodes WHERE root_id=? AND path_key BETWEEN ? AND ?",
                (root_id, *tree_bounds(path)),
            )
            self.db.execute("UPDATE nodes SET descendants=0 WHERE id=?", (identity,))
            self._discard_orphan_commits()
            self.links_dirty = True
        if previous is None or (previous["state"] in GRAPH_BLOCKED) != (state in GRAPH_BLOCKED):
            self.links_dirty = True
        self.db.execute(
            """
            INSERT INTO nodes (id,parent_id,root_id,project_id,path,path_key,name,kind,category,state,
                size,mtime_ns,file_id,digest,seen) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(id) DO UPDATE SET parent_id=excluded.parent_id,root_id=excluded.root_id,
                project_id=excluded.project_id,path=excluded.path,path_key=excluded.path_key,
                name=excluded.name,kind=excluded.kind,category=excluded.category,state=excluded.state,
                size=excluded.size,mtime_ns=excluded.mtime_ns,file_id=excluded.file_id,
                digest=excluded.digest,seen=excluded.seen
        """,
            (
                identity,
                parent_id,
                root_id,
                project_id,
                str(path),
                path_key(path),
                path.name or str(path),
                kind,
                "folder" if kind != "file" else category(path),
                state,
                *version,
                digest,
                token,
            ),
        )

    def _retire_missing_children(self, folder, identity, root_id, children):
        names = {path_key(path) for path, _ in children}
        removed = False
        for key, path in self.db.execute(
            "SELECT path_key,path FROM nodes WHERE parent_id=? AND root_id=?",
            (identity, root_id),
        ).fetchall():
            path = Path(path)
            if key not in names and path.parent == folder:
                self._remove_tree(path, root_id)
                removed = True
        return removed

    def _inaccessible(self, path, root_id, token=None):
        low, high = tree_bounds(path)
        condition = "root_id=? AND (path_key=? OR path_key BETWEEN ? AND ?)"
        args = (root_id, path_key(path), low, high)
        for (identity,) in self.db.execute(
            f"SELECT id FROM nodes WHERE {condition}", args
        ).fetchall():
            self._clear_evidence(identity)
        if token is None:
            self.db.execute(
                f"UPDATE nodes SET state='inaccessible',digest=NULL WHERE {condition}", args
            )
        else:
            self.db.execute(
                f"""
                UPDATE nodes SET state='inaccessible',digest=NULL,seen=? WHERE {condition}
            """,
                (token, *args),
            )

    def _clear_evidence(self, identity):
        for table in ("passages", "declared_links", "note_titles", "topics"):
            key = "node_id" if table == "passages" else "source"
            self.db.execute(f"DELETE FROM {table} WHERE {key}=?", (identity,))
        self.db.execute(
            """
            DELETE FROM edges WHERE kind IN ('reference','duplicate','topic')
            AND (source=? OR target=?)
        """,
            (identity, identity),
        )
        self.links_dirty = True

    def _remove_tree(self, path, root_id):
        self.db.execute(
            "DELETE FROM nodes WHERE root_id=? AND (path_key=? OR path_key BETWEEN ? AND ?)",
            (
                root_id,
                path_key(path),
                *tree_bounds(path),
            ),
        )
        self._discard_orphan_commits()
        self.links_dirty = True

    def _discard_orphan_commits(self):
        self.db.execute("""
            DELETE FROM nodes WHERE parent_id IS NULL AND kind='commit' AND NOT EXISTS (
                SELECT 1 FROM nodes project WHERE project.id=nodes.project_id
                AND project.kind IN ('project','root')
            )
        """)

    def _recount(self, folder):
        self.db.execute(
            """
            UPDATE nodes SET descendants=(
                SELECT coalesce(sum(1+descendants),0) FROM nodes WHERE parent_id=? AND kind<>'commit'
            ) WHERE id=?
        """,
            (path_id(folder), path_id(folder)),
        )

    def _recount_ancestors(self, folder, root):
        while folder.is_relative_to(root):
            self._recount(folder)
            if folder == root:
                break
            folder = folder.parent

    def watch_path_allowed(self, path, root):
        if path == root:
            return True
        relative = path.relative_to(root)
        if any(part.casefold() in INTERNAL for part in relative.parts) or self._skip(path, root):
            return False
        if self._skip(root / relative.parts[0], root):
            return False
        return (
            any(part.casefold() == ".git" for part in relative.parts)
            or root in self.history_roots
            or not protected(path.parent)
        )

    def refresh_paths(self, paths, roots, *, checkpoint=None):
        selected = {}
        changed = set()
        scopes = sorted(roots, key=lambda root: len(root.parts), reverse=True)
        for path in paths:
            for root in scopes:
                if not path.is_relative_to(root):
                    continue
                relative = path.relative_to(root)
                if not self.watch_path_allowed(path, root):
                    break
                # A Git ref change is evidence about the repository, not the .git directory.
                parts = tuple(part.casefold() for part in relative.parts)
                if ".git" in parts:
                    project = root.joinpath(*relative.parts[: parts.index(".git")])
                    self.dirty_repos.add(path_id(project))
                    break
                candidate = path
                while (
                    candidate != root
                    and self.db.execute(
                        "SELECT 1 FROM nodes WHERE path_key=?", (path_key(candidate.parent),)
                    ).fetchone()
                    is None
                ):
                    candidate = candidate.parent
                selected[(candidate, root)] = None
                changed.add(path)
                break
        if checkpoint is not None:
            direct = []
            for target, root in selected:
                if target in changed:
                    try:
                        if file_state(target, target.lstat()) == "pending":
                            direct.append((target, root))
                    except OSError:
                        continue
                if len(direct) >= WATCH_EXTRACTION_BATCH:
                    break
            for target, root in direct:
                self._inventory(target, root, changed=changed)
                selected.pop((target, root))
                changed.discard(target)
            if direct:
                self._prioritise_watched(target for target, _ in direct)
                checkpoint(None)
        parents = set()
        for target, root in sorted(selected, key=lambda pair: len(pair[0].parts)):
            if not any(parent in parents for parent in target.parents):
                for continuing, _ in self._inventory_steps(
                    target, root, changed=changed, publish_batches=checkpoint is not None
                ):
                    if continuing and checkpoint is not None:
                        self._prioritise_watched(changed)
                        checkpoint(target)
                parents.add(target)
        if checkpoint is not None:
            self._prioritise_watched(changed)

    def _prioritise_watched(self, paths):
        for path in paths:
            self.watched_files.setdefault(path_id(path), None)
        while len(self.watched_files) > WATCH_EVENT_BATCH * 8:
            self.watched_files.pop(next(iter(self.watched_files)))

    def extract_watched(self, *, allow_name_lookup=True):
        queued = tuple(islice(self.watched_files, WATCH_EXTRACTION_BATCH))
        if not queued:
            return 0
        marks = ",".join("?" for _ in queued)
        rows = self.db.execute(
            f"""
            SELECT n.id,n.path,n.mtime_ns,n.size,n.file_id FROM nodes n JOIN nodes r ON r.id=n.root_id
            WHERE n.id IN ({marks}) AND n.state='pending' AND r.state='online'
            """,
            queued,
        ).fetchall()
        selected = {row["id"]: row for row in rows}
        ordered = []
        for identity in queued:
            if identity in selected:
                ordered.append(selected[identity])
            else:
                self.watched_files.pop(identity)
        completed = self._extract_rows(
            ordered,
            WATCH_EXTRACTION_SECONDS,
            publish_references=True,
            allow_name_lookup=allow_name_lookup,
        )
        for row in ordered[:completed]:
            self.watched_files.pop(row["id"])
        return completed

    def extract_pending(
        self,
        limit=32,
        time_budget=1.5,
        *,
        publish_references=False,
        inventory_token=None,
        allow_name_lookup=True,
    ):
        if inventory_token is None:
            oldest = self.db.execute("""
                SELECT n.seen FROM nodes n JOIN nodes r ON r.id=n.root_id
                WHERE n.state='pending' AND r.state='online' ORDER BY n.seen LIMIT 1
            """).fetchone()
            if oldest is None:
                self.extraction_paused = False
                return 0
            seen = oldest["seen"]
            early_filter = ""
        else:
            seen = inventory_token
            early_filter = "AND n.category='note'"
        rows = self.db.execute(
            f"""
            SELECT n.id,n.path,n.mtime_ns,n.size,n.file_id FROM nodes n JOIN nodes r ON r.id=n.root_id
            WHERE n.state='pending' AND r.state='online' AND n.seen=? {early_filter}
            ORDER BY n.size LIMIT ?
        """,
            (seen, limit),
        ).fetchall()
        return self._extract_rows(
            rows,
            time_budget,
            publish_references=publish_references,
            allow_name_lookup=allow_name_lookup and inventory_token is None,
            inventory=inventory_token is not None,
        )

    def _extract_rows(
        self, rows, time_budget, *, publish_references, allow_name_lookup=True, inventory=False
    ):
        if not rows:
            return 0
        if not self._within_disk_budget():
            if not self.extraction_paused:
                with self.db:
                    self._publish("Content extraction paused at the local disk budget.")
            self.extraction_paused = True
            return 0
        self.extraction_paused = False
        deadline = time.monotonic() + time_budget
        outcomes = []
        for start in range(0, len(rows), self.extraction_workers):
            batch = rows[start : start + self.extraction_workers]
            outcomes.extend(self.parsers.map(self._prepare_extraction, batch))
            if time.monotonic() >= deadline:
                break
        processed = list(zip(rows[: len(outcomes)], outcomes, strict=True))
        with self.db:
            self._store_extraction(processed)
            if publish_references:
                self.refresh_references(
                    (row["id"] for row, evidence in processed if evidence.links),
                    allow_name_lookup=allow_name_lookup,
                )
            self._publish(
                "Inventorying personal folders." if inventory else "Extracting local evidence.",
                True,
            )
        return len(processed)

    def _within_disk_budget(self):
        index_bytes = (
            self.db.execute("PRAGMA page_count").fetchone()[0]
            * self.db.execute("PRAGMA page_size").fetchone()[0]
        )
        return index_bytes < MAX_INDEX_BYTES and (
            shutil.disk_usage(self.database.parent).free >= MIN_DISK_FREE
        )

    def advance_names(self) -> bool | None:
        if self.db.execute("SELECT ready FROM filename_index_progress").fetchone()[0]:
            return True
        if self._within_disk_budget():
            return advance_name_index(self.db, limit=32768)
        paused = "Filename indexing paused at the local disk budget."
        current = self.db.execute("SELECT message FROM catalogue").fetchone()[0]
        if not self.extraction_paused and current != paused:
            with self.db:
                self._publish(paused)
        return None

    def can_drain_published(self, roots):
        inventory = dict(self.db.execute("SELECT id,phase FROM inventory_roots"))
        return (
            inventory.keys() == {path_id(root) for root in roots}
            and "pending" not in inventory.values()
            and self._online_pending()
        )

    def _online_pending(self):
        return (
            self.db.execute("""
            SELECT 1 FROM nodes n JOIN nodes r ON r.id=n.root_id
            WHERE n.state='pending' AND r.state='online' LIMIT 1
        """).fetchone()
            is not None
        )

    def _publish_ready(self):
        if self.links_dirty or self.dirty_repos or self._online_pending():
            return
        if self.db.execute(
            "SELECT 1 FROM inventory_roots WHERE phase='pending' LIMIT 1"
        ).fetchone():
            return
        if not self.db.execute("SELECT ready FROM filename_index_progress").fetchone()[0]:
            return
        message = self.db.execute("SELECT message FROM catalogue").fetchone()[0]
        if message != "Local catalogue ready.":
            with self.db:
                self._publish("Local catalogue ready.")

    @staticmethod
    def _expected_version(row):
        return row["size"], row["mtime_ns"], row["file_id"]

    def _prepare_extraction(self, row):
        path = Path(row["path"])
        digest = None
        try:
            raw = read_bytes(path, internal=history_source(path) in {"codex", "claude"})
            if file_version(safe_path(path)) != self._expected_version(row):
                raise OSError("File changed during extraction")
            digest = hashlib.sha256(raw).hexdigest()
            suffix = path.suffix.casefold()
            with self.pdf_slots if suffix == ".pdf" else nullcontext():
                passages, text, _ = extract(path, raw)
            is_markdown = suffix in {".md", ".markdown"}
            refs, topics, title = links_and_topics(text) if is_markdown else ({}, {}, None)
            aliases = {normalise_note_title(path.stem), title} if is_markdown and passages else ()
            return FileEvidence(
                "indexed" if passages else "metadata",
                digest,
                tuple((kind, start, end, value) for kind, start, end, value in passages),
                tuple(refs.items()),
                tuple(topics.items()),
                tuple(alias for alias in aliases if alias),
            )
        except PermissionError:
            return FileEvidence("protected")
        except EncryptedDocument:
            return FileEvidence("encrypted")
        except FileNotFoundError:
            return FileEvidence("inaccessible")
        except (
            OSError,
            ValueError,
            BadZipFile,
            ElementTree.ParseError,
            subprocess.SubprocessError,
        ):
            return FileEvidence("failed", digest)

    def _store_extraction(self, processed):
        for index, (row, evidence) in enumerate(processed):
            if not evidence.digest:
                continue
            try:
                current = Path(row["path"]).lstat()
                if entry_state(current) or file_version(current) != self._expected_version(row):
                    raise OSError("File changed before evidence publication")
            except OSError:
                processed[index] = row, FileEvidence("failed")
        self.db.executemany(
            "INSERT INTO passages(node_id,kind,start,end,text) VALUES (?,?,?,?,?)",
            ((row["id"], *item) for row, evidence in processed for item in evidence.passages),
        )
        self.db.executemany(
            "INSERT INTO declared_links VALUES (?,?,?)",
            ((row["id"], *item) for row, evidence in processed for item in evidence.links),
        )
        self.db.executemany(
            "INSERT INTO topics VALUES (?,?,?)",
            ((row["id"], *item) for row, evidence in processed for item in evidence.topics),
        )
        self.db.executemany(
            "INSERT INTO note_titles VALUES (?,?)",
            ((row["id"], title) for row, evidence in processed for title in evidence.titles),
        )
        self.db.executemany(
            "UPDATE nodes SET state=?,digest=? WHERE id=?",
            ((evidence.state, evidence.digest, row["id"]) for row, evidence in processed),
        )
        self.links_dirty = True

    def refresh_edges(self):
        if not self.links_dirty:
            return
        with self.db:
            self.db.execute("DELETE FROM edges WHERE kind IN ('reference','duplicate','topic')")
            self.refresh_references()
            self.db.execute("""
                INSERT OR IGNORE INTO edges
                WITH groups AS (
                    SELECT n.digest,min(n.id) representative FROM nodes n JOIN nodes r ON r.id=n.root_id
                    WHERE n.digest IS NOT NULL AND n.kind='file' AND r.state='online'
                    GROUP BY n.digest HAVING count(*)>1
                )
                SELECT groups.representative,n.id,'duplicate','Identical SHA-256 of source bytes'
                FROM groups JOIN nodes n ON n.digest=groups.digest JOIN nodes r ON r.id=n.root_id
                WHERE n.id<>groups.representative AND n.kind='file' AND r.state='online'
            """)
            self.db.execute("""
                INSERT OR IGNORE INTO edges
                WITH common AS (
                    SELECT t.topic,min(t.source) representative,min(t.line) location
                    FROM topics t JOIN nodes n ON n.id=t.source JOIN nodes r ON r.id=n.root_id
                    WHERE n.state='indexed' AND r.state='online'
                    GROUP BY t.topic HAVING count(*) BETWEEN 2 AND 12
                )
                SELECT common.representative,t.source,'topic',
                       'Suggested from matching heading: '||t.topic||' (line '||t.line||')'
                FROM common JOIN topics t ON common.topic=t.topic
                JOIN nodes n ON n.id=t.source JOIN nodes r ON r.id=n.root_id
                WHERE t.source<>common.representative AND n.state='indexed' AND r.state='online'
            """)
            message = "Local links refreshed."
            if self.extraction_paused:
                message = "Content extraction paused at the local disk budget."
            self._publish(message, scanning=self._online_pending() and not self.extraction_paused)
        self.links_dirty = False

    def refresh_references(self, sources=None, *, allow_name_lookup=True):
        sources = None if sources is None else tuple(sources)
        if sources == ():
            return
        scope = ""
        if sources:
            scope = "AND d.source IN (" + ",".join("?" for _ in sources) + ")"
        rows = self.db.execute(
            f"""
            SELECT d.source,d.destination,d.line,n.path,n.project_id
            FROM declared_links d JOIN nodes n ON n.id=d.source JOIN nodes r ON r.id=n.root_id
            WHERE n.state='indexed' AND r.state='online' {scope}
            ORDER BY d.source,d.line,d.destination
            """,
            sources or (),
        )
        for row in rows:
            if (target := self._resolve_link(row, allow_name_lookup=allow_name_lookup)) and (
                target != row["source"]
            ):
                is_wikilink = row["destination"].startswith(WIKI_DESTINATION)
                syntax = "Wikilink" if is_wikilink else "Markdown link"
                evidence = f"{syntax} on line {row['line']}"
                heading = unquote(row["destination"].partition("#")[2])
                if heading.strip():
                    evidence += f" · Requested heading {heading.strip()[:120]}"
                self.db.execute(
                    "INSERT OR IGNORE INTO edges VALUES (?,?,?,?)",
                    (row["source"], target, "reference", evidence),
                )

    def _unique_wikilink(self, name):
        matches = self.db.execute(
            """
            WITH candidates AS (
                SELECT id FROM nodes WHERE kind='file' AND name COLLATE NOCASE IN (?,?,?)
                UNION SELECT source FROM note_titles WHERE title=?
            )
            SELECT n.id FROM candidates c JOIN nodes n ON n.id=c.id JOIN nodes r ON r.id=n.root_id
            WHERE r.state='online' AND n.state NOT IN (
                'offline','inaccessible','protected','redirected','placeholder'
            )
            LIMIT 2
            """,
            (name, name + ".md", name + ".markdown", normalise_note_title(name)),
        ).fetchall()
        return matches[0]["id"] if len(matches) == 1 else None

    def _resolve_link(self, row, *, allow_name_lookup=True):
        raw = row["destination"].partition("#")[0]
        if raw.startswith(WIKI_DESTINATION):
            raw = raw.removeprefix(WIKI_DESTINATION)
            if "/" not in raw and "\\" not in raw:
                # An unfinished inventory cannot establish uniqueness across the library.
                return self._unique_wikilink(unquote(raw)) if allow_name_lookup else None
        try:
            destination = urlsplit(raw)
        except ValueError:
            return None
        if destination.scheme == "file" and not destination.netloc:
            decoded = unquote(destination.path)
            if os.name == "nt" and len(decoded) >= 3 and decoded[0] == "/" and decoded[2] == ":":
                decoded = decoded[1:]
            relative = Path(decoded)
        elif not destination.scheme and not destination.netloc:
            relative = Path(unquote(destination.path))
        elif os.name == "nt" and Path(raw).is_absolute():
            relative = Path(unquote(raw))
        else:
            return None
        if not relative.parts or relative.drive.startswith(("\\", "/")):
            return None
        path = relative if relative.is_absolute() else Path(row["path"]).parent / relative
        targets = (path, path.with_suffix(".md")) if not path.suffix else (path,)
        for target in targets:
            match = self.db.execute(
                """
                SELECT n.id FROM nodes n JOIN nodes r ON r.id=n.root_id WHERE n.path_key=?
                AND n.state NOT IN ('protected','placeholder','redirected','inaccessible')
                AND r.state='online'
            """,
                (path_key(target),),
            ).fetchone()
            if match:
                return match["id"]
        if allow_name_lookup and len(relative.parts) == 1 and row["project_id"]:
            name = relative.name if relative.suffix else relative.name + ".md"
            matches = self.db.execute(
                """
                SELECT id FROM nodes WHERE project_id=? AND name=? COLLATE NOCASE AND state='indexed'
                LIMIT 2
            """,
                (row["project_id"], name),
            ).fetchall()
            if len(matches) == 1:
                return matches[0]["id"]
        return None

    def refresh_git(self, limit=None):
        identities = list(self.dirty_repos)[:limit]
        for identity in identities:
            self.dirty_repos.discard(identity)
            row = self.db.execute(
                "SELECT id,path,root_id FROM nodes WHERE id=?", (identity,)
            ).fetchone()
            if row is None:
                continue
            if not repository(Path(row["path"])):
                with self.db:
                    deleted = self.db.execute(
                        "DELETE FROM nodes WHERE kind='commit' AND project_id=?", (identity,)
                    ).rowcount
                    if deleted:
                        self._publish("Removed unavailable Git history.")
                continue
            try:
                records = history(Path(row["path"]))
            except (OSError, ValueError, subprocess.SubprocessError):
                continue
            commits = {
                record["commit"]: revision_id(identity, record["commit"]) for record in records
            }
            with self.db:
                self.db.execute(
                    "DELETE FROM nodes WHERE kind='commit' AND project_id=?", (identity,)
                )
                for record in records:
                    commit = record["commit"]
                    commit_id = commits[commit]
                    locator = f"{row['path']}#{commit}"
                    self.db.execute(
                        """
                        INSERT INTO nodes (id,parent_id,root_id,project_id,path,path_key,name,kind,
                            category,state,size,mtime_ns,file_id,digest)
                        VALUES (?,NULL,?,?,?,?,?,'commit','code','indexed',0,?,'',?)
                    """,
                        (
                            commit_id,
                            row["root_id"],
                            identity,
                            locator,
                            path_key(locator),
                            record["subject"] or commit[:12],
                            sqlite_integer(record["time_ns"]),
                            commit,
                        ),
                    )
                    self.db.execute(
                        "INSERT INTO passages(node_id,kind,start,end,text) VALUES (?,?,?,?,?)",
                        (commit_id, "commit", 1, 1, record["subject"] or commit),
                    )
                    changed = [
                        str(Path(name).relative_to(row["path"]))
                        for name in sorted(record["files"])[:300]
                    ]
                    self.db.execute(
                        "INSERT INTO commit_metadata VALUES (?,?,?)",
                        (
                            commit_id,
                            json.dumps(record["refs"]),
                            json.dumps(changed),
                        ),
                    )
                for record in records:
                    commit_id = commits[record["commit"]]
                    self.db.executemany(
                        "INSERT OR IGNORE INTO edges VALUES (?,?,?,?)",
                        (
                            (commit_id, commits[parent], "commit_parent", "Git commit parent")
                            for parent in record["parents"]
                            if parent in commits
                        ),
                    )
                    for file in record["files"]:
                        self.db.execute(
                            """
                            INSERT OR IGNORE INTO edges SELECT ?,id,'changed_file','Path in Git commit'
                            FROM nodes WHERE path_key=? AND root_id=?
                            AND state NOT IN ('protected','placeholder','redirected','inaccessible')
                        """,
                            (commit_id, file, row["root_id"]),
                        )
                self._publish("Git history refreshed.")


def serve(index, roots, *, automatic=False):
    changes = {}
    urgent_changes = {}
    watchers = {}
    scopes = ()
    lock = threading.Lock()
    next_names = 0.0
    names_ready = False

    def notify(path, watched):
        owner = next((root for root in scopes if path.is_relative_to(root)), None)
        if owner is None or not index.watch_path_allowed(path, owner):
            return
        if path == watched:
            path = owner  # The Windows watcher uses its directory as an overflow signal.
        urgent = (
            path != owner
            and path.is_relative_to(index.desktop)
            and (path.suffix.casefold() in SUPPORTED or account_bundle(path))
            and not any(part.casefold() in GENERATED for part in path.relative_to(owner).parts)
        )
        with lock:
            if len(changes) < 20000:
                changes.setdefault(path, None)
            else:
                changes.setdefault(owner, None)
            if urgent:
                urgent_changes.pop(path, None)
                urgent_changes[path] = None
                if len(urgent_changes) > WATCH_URGENT_LIMIT:
                    urgent_changes.pop(next(iter(urgent_changes)))

    def watch():
        nonlocal scopes
        scopes = tuple(sorted(roots, key=lambda root: len(root.parts), reverse=True))
        if os.name != "nt":
            return
        from .whole_index_windows import DirectoryWatch

        watched = list(roots)
        if any(index.desktop.is_relative_to(root) for root in roots) and index.desktop not in roots:
            watched.append(index.desktop)
        for path in watched:
            if path.is_dir() and (path not in watchers or not watchers[path].thread.is_alive()):
                watchers[path] = DirectoryWatch(path, lambda event, base=path: notify(event, base))
                watchers[path].start()

    def take_urgent(outside=None):
        with lock:
            paths = tuple(
                islice(
                    (
                        path
                        for path in reversed(urgent_changes)
                        if outside is None
                        or not (path.is_relative_to(outside) or outside.is_relative_to(path))
                    ),
                    WATCH_EXTRACTION_BATCH,
                )
            )
            for path in paths:
                urgent_changes.pop(path)
                changes.pop(path, None)
            return paths

    def service_names():
        nonlocal names_ready, next_names
        if not names_ready and time.monotonic() >= next_names:
            names_ready = index.advance_names()
            next_names = time.monotonic() + NAME_INDEX_SECONDS

    def extract_watched_only(_scope):
        index.extract_watched(allow_name_lookup=False)

    def drain_at_checkpoint(scope=None):
        if scope is not None and (urgent := take_urgent(outside=scope)):
            index.refresh_paths(urgent, roots, checkpoint=extract_watched_only)
        extract_watched_only(scope)
        index.extract_pending(
            EXTRACTION_BATCH,
            publish_references=True,
            allow_name_lookup=False,
        )
        service_names()

    def refresh_completed_paths(completed):
        drain_at_checkpoint()
        if not completed:
            return
        batch = []
        with lock:
            if any(root in changes for root in roots):
                return  # A watcher overflow requires a full pass after this inventory.
            for path in changes:
                owner = next((root for root in scopes if path.is_relative_to(root)), None)
                if owner in completed:
                    batch.append(path)
                    if len(batch) >= WATCH_EVENT_BATCH:
                        break
            for path in batch:
                changes.pop(path)
        if batch:
            index.refresh_paths(batch, roots, checkpoint=drain_at_checkpoint)

    watch()
    rescan_pending = index.can_drain_published(roots)
    if not rescan_pending:
        index.reconcile(roots, extract_now=False, checkpoint=refresh_completed_paths)
    available = {root for root in roots if root.is_dir()}
    rescanned = last_availability = time.monotonic()
    next_links = 0.0
    extracting = True
    while True:
        started = time.monotonic()
        if started - last_availability >= RECONNECT_SECONDS:
            current = {root for root in roots if root.is_dir()}
            if current != available:
                watch()
                for root in current ^ available:
                    for continuing, _ in index._inventory_steps(root, root, publish_batches=True):
                        if continuing:
                            drain_at_checkpoint(root)
                available = current
            last_availability = started
        urgent = take_urgent()
        with lock:
            if any(root in changes for root in roots):
                # Clear known work before a full recovery sweep.
                rescan_pending = True
                for root in roots:
                    changes.pop(root, None)
            batch = tuple(islice(changes, WATCH_EVENT_BATCH - len(urgent)))
            for path in (*urgent, *batch):
                changes.pop(path, None)
                urgent_changes.pop(path, None)
        batch = tuple(dict.fromkeys((*urgent, *batch)))
        routine = not batch and not extracting and started - rescanned >= RESCAN_SECONDS
        if routine or (rescan_pending and not extracting and not index.extraction_paused):
            if automatic:
                roots = discover_roots()
            watch()
            index.reconcile(roots, extract_now=False, checkpoint=refresh_completed_paths)
            rescan_pending = False
            rescanned = time.monotonic()
            available = {root for root in roots if root.is_dir()}
        elif batch:
            index.refresh_paths(batch, roots, checkpoint=drain_at_checkpoint)
        index.extract_watched()
        extracted = index.extract_pending(EXTRACTION_BATCH, publish_references=True)
        index.refresh_git(limit=1)
        extracting = bool(extracted or index.dirty_repos)
        if index.links_dirty and (started >= next_links or (not extracting and not rescan_pending)):
            full_started = time.monotonic()
            index.refresh_edges()
            finished = time.monotonic()
            # Keep costly full refreshes from monopolising the writer as the catalogue grows.
            next_links = finished + max(LINK_REFRESH_SECONDS, 8 * (finished - full_started))
        service_names()
        if not extracting and not rescan_pending and names_ready:
            index._publish_ready()
        delay = 0.1 if extracted else DEBOUNCE_SECONDS
        time.sleep(max(0.1, delay - (time.monotonic() - started)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=default_database())
    parser.add_argument("--root", action="append", type=Path)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    lock = nullcontext(True)
    if os.name == "nt":
        from .whole_index_windows import index_lock, lower_index_priority

        lock = index_lock(args.database)
    with lock as acquired:
        if not acquired:
            return
        if os.name == "nt":
            lower_index_priority()
        index = Indexer(args.database)
        try:
            roots = [Path(root).absolute() for root in (args.root or discover_roots())]
            if args.once:
                index.reconcile(roots)
            else:
                serve(index, roots, automatic=not args.root)
        finally:
            index.close()


if __name__ == "__main__":
    main()
