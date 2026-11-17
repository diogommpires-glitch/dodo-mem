"""Authorisation joins the private catalogue with the current shared vault policy."""

import hashlib
import re
import sqlite3
import stat
import time
from contextlib import closing, contextmanager
from copy import copy
from pathlib import Path

from .resources import ResourcePressure


class ProjectionBudget(ResourcePressure):
    pass


def local_sources():
    from dodo_memory.store import Vault
    from dodo_memory_desktop.configuration import runtime_path
    from dodo_memory_desktop.whole_index import Catalogue

    return Catalogue(), Vault(runtime_path("gateway"))


class AuthorisedCatalogue:
    def __init__(self, catalogue, vault, *, profile=None):
        self.catalogue, self.vault = catalogue, vault
        self.providers = []
        self.profile = profile
        self.profile_sources = {}
        self.profile_error = None
        self.knowledge = None
        if hasattr(vault, "db_path"):
            from dodo_memory.review import AcceptedKnowledge

            self.knowledge = AcceptedKnowledge(vault)
        if profile is not None:
            self._refresh_profile()

    def _refresh_profile(self):
        if self.profile is None:
            return
        self.profile_sources = {}
        try:
            from dodo_memory_desktop.personal_context import validate_profile_path
            from dodo_memory_desktop.whole_index import safe_path

            for source in self.profile.sources():
                if not source.enabled:
                    continue
                validate_profile_path(source.path)
                try:
                    info = safe_path(source.path)
                except FileNotFoundError:
                    continue
                if stat.S_ISREG(info.st_mode):
                    node_id = (
                        "profile-local:"
                        + hashlib.sha256(str(source.path).casefold().encode()).hexdigest()
                    )
                    self.profile_sources[node_id] = source
            self.profile_error = None
        except (OSError, ValueError):
            self.profile_sources = {}
            self.profile_error = "Personal profile configuration is unavailable or invalid."

    def _profile_read(self, node_id, limit=1000, start_line=1, query_terms=()):
        from dodo_memory_desktop.personal_context import read_profile_file

        source = self.profile_sources[node_id]
        result = read_profile_file(source.path, limit, start_line, query_terms=query_terms)
        if result:
            result.update(
                node_id=node_id, profile={"section": source.section, "basis": source.basis}
            )
        return result

    def revision(self):
        return self.catalogue.revision()

    def normalise_query(self, query):
        from dodo_memory_desktop.personal_context import contextual_request

        return contextual_request(query)[0]

    def _shared(self, relative):
        from dodo_memory_desktop.whole_index import read_bytes

        path = self.vault.resolve(relative)
        digest = hashlib.sha256(read_bytes(path)).hexdigest()
        note = self.vault.get(relative, scopes=("shared",), expected_sha256=digest)
        return path, digest, note

    def node(self, node_id):
        self._refresh_profile()
        if node_id in self.profile_sources:
            source = self.profile_sources[node_id]
            result = self._profile_read(node_id)
            return (
                {
                    **result,
                    "id": node_id,
                    "name": source.path.name,
                    "kind": "file",
                    "state": "indexed",
                }
                if result
                else None
            )
        for provider in self.providers:
            if provider.accepts(node_id):
                return provider.node(node_id)
        if node_id.startswith("shared:"):
            try:
                path, digest, _ = self._shared(node_id[7:])
            except (OSError, ValueError, PermissionError):
                return None
            return {
                "id": node_id,
                "path": str(path),
                "digest": digest,
                "state": "indexed",
                "kind": "file",
                "name": path.name,
            }
        return self.catalogue.node(node_id)

    def authorized(self, node_id, scope):
        if scope not in {"shared", "full_private"}:
            return False
        if node_id in self.profile_sources:
            return scope == "full_private" and self.node(node_id) is not None
        for provider in self.providers:
            if provider.accepts(node_id):
                return provider.authorized(node_id, scope)
        node = self.node(node_id)
        if not node or node.get("state") in {
            "offline",
            "inaccessible",
            "protected",
            "redirected",
            "placeholder",
            "deleted",
        }:
            return False
        path = Path(node["path"])
        if path.is_relative_to(self.vault.root):
            try:
                self.vault.get(
                    path.relative_to(self.vault.root).as_posix(),
                    scopes=("shared",) if scope == "shared" else ("shared", "private", "archive"),
                )
                return True
            except (OSError, ValueError, PermissionError):
                return False
        return scope == "full_private"

    def source_scope(self, node):
        path = Path(node["path"])
        if path.is_relative_to(self.vault.root):
            if not self.authorized(node["id"], "full_private"):
                return None
            if self.authorized(node["id"], "shared"):
                return "shared"
        return "private"

    @contextmanager
    def projection_reader(self, deadline):
        database = getattr(self.catalogue, "database", None)
        if database is None:
            yield self
            return
        from dodo_memory_desktop.whole_index import safe_path

        safe_path(database)
        with closing(
            sqlite3.connect(
                database.as_uri() + "?mode=ro", uri=True, timeout=0.05, isolation_level=None
            )
        ) as db:
            db.row_factory = sqlite3.Row
            db.execute("PRAGMA query_only=ON")
            db.set_progress_handler(lambda: int(time.monotonic() >= deadline), 1000)

            @contextmanager
            def read():
                if time.monotonic() >= deadline:
                    raise ProjectionBudget("Topic preview reached its validation budget.")
                try:
                    # Reuse the connection, not a transaction across physical source reads.
                    # The existing final node lookup therefore still sees a new revision.
                    yield db
                except sqlite3.OperationalError as error:
                    code = getattr(error, "sqlite_errorcode", 0) & 255
                    if code == sqlite3.SQLITE_INTERRUPT and time.monotonic() >= deadline:
                        raise ProjectionBudget(
                            "Topic preview reached its validation budget."
                        ) from error
                    if code in {sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED}:
                        raise ProjectionBudget(
                            "Topic preview is waiting for the source catalogue."
                        ) from error
                    raise

            reader = copy(self)
            reader.catalogue = copy(self.catalogue)
            reader.catalogue._read = read
            yield reader

    def _catalogue_lexical(self, query, limit):
        # Profile-only adapters have no private catalogue read capability.
        read = getattr(self.catalogue, "_read", None)
        terms = re.findall(r"\w+", query, flags=re.UNICODE)[:32]
        if read is None or not terms:
            return []
        from dodo_memory_desktop.whole_index import protected

        expression = " OR ".join('"' + term + '"' for term in terms)
        with read() as db:
            if db is None:
                return []
            rows = self.catalogue._effective(
                db,
                db.execute(
                    """WITH hits AS MATERIALIZED (
                        SELECT p.node_id,p.start,rank AS score FROM passage_search
                        JOIN passages p ON p.id=passage_search.rowid
                        JOIN nodes n ON n.id=p.node_id JOIN nodes r ON r.id=n.root_id
                        WHERE passage_search MATCH ? AND n.state='indexed' AND r.state='online'
                        ORDER BY rank LIMIT 1000
                    ), ranked AS (
                        SELECT *,row_number() OVER(PARTITION BY node_id ORDER BY score,start) position
                        FROM hits
                    ) SELECT n.*,h.start AS match_start FROM ranked h JOIN nodes n ON n.id=h.node_id
                    WHERE h.position=1 ORDER BY h.score,h.node_id LIMIT 64""",
                    (expression,),
                ),
            )
        return [
            row
            for row in rows
            if not (row["kind"] == "file" and protected(Path(row["path"])))
            and self.authorized(row["id"], "full_private")
        ][:limit]

    def search(self, query, *, limit=8, scope="shared"):
        self._refresh_profile()
        result = []
        if self.knowledge is not None:
            from dodo_memory.personal_profile import PROFILE_KEY, WITHDRAWN_PROFILE

            profile = self.knowledge.read(PROFILE_KEY, scopes=("shared",))
            if profile["status"] == "found" and profile["text"].strip() != WITHDRAWN_PROFILE:
                node = self.node("shared:" + profile["path"])
                if node:
                    result.append({**node, "node_id": node["id"], "start_line": 1})
        if scope == "full_private" and self.profile_sources:
            from dodo_memory_desktop.personal_context import profile_sections

            sections = profile_sections(query)
            ordered = sorted(
                self.profile_sources.items(),
                key=lambda item: item[1].section == "Standing preferences",
            )
            added_sections = set()
            for node_id, source in ordered:
                if source.section not in sections:
                    continue
                if source.section in added_sections:
                    continue
                evidence = self._profile_read(node_id, query_terms=query.casefold().split())
                node = self.node(node_id)
                if evidence and node:
                    result.append(
                        {**node, "node_id": node_id, "start_line": evidence["locator"]["start"]}
                    )
                    added_sections.add(source.section)
                if len(added_sections) == 2:
                    break
        for item in self.vault.search(
            query, scopes=("shared",), limit=min(limit, 20), mode="keyword"
        ):
            node_id = "shared:" + item["path"]
            node = self.node(node_id)
            if node:
                result.append({**node, "node_id": node_id, "start_line": 1})
        if scope == "full_private":
            result.extend(
                row
                for row in self.catalogue.search(query, limit=limit)
                if self.authorized(row["id"], scope)
            )
            if len(result) < limit:
                seen = {row.get("node_id", row.get("id")) for row in result}
                result.extend(
                    row for row in self._catalogue_lexical(query, limit) if row["id"] not in seen
                )
        return result[:limit]

    def read_evidence(self, node_id, *, limit=1000, start_line=1):
        self._refresh_profile()
        if node_id in self.profile_sources:
            return self._profile_read(node_id, limit, start_line)
        for provider in self.providers:
            if provider.accepts(node_id):
                return provider.read_evidence(node_id, limit=limit, start_line=start_line)
        if node_id.startswith("shared:"):
            try:
                path, digest, note = self._shared(node_id[7:])
            except (OSError, ValueError, PermissionError):
                return None
            lines = note["text"].splitlines()
            start = max(1, start_line)
            text = "\n".join(lines[start - 1 :])[:limit]
            return {
                "node_id": node_id,
                "path": str(path),
                "digest": digest,
                "locator": {"kind": "lines", "start": start, "end": start + text.count("\n")},
                "text": text,
            }
        return self.catalogue.read_evidence(node_id, limit=limit, start_line=start_line)

    def neighbours(self, node_id, *, limit=64, include_inferred=False):
        if node_id in self.profile_sources:
            return []
        if any(provider.accepts(node_id) for provider in self.providers):
            return []
        if node_id.startswith("shared:"):
            node = self.node(node_id)
            physical = self.catalogue.resolve_path(node["path"]) if node else None
            if physical is None:
                return []
            node_id = physical["id"]
        return self.catalogue.neighbours(node_id, limit=limit, include_inferred=include_inferred)
