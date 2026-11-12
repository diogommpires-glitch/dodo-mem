"""Shared desktop catalogue and one-hop provenance views."""

import json
import posixpath
import re
import time
from pathlib import PurePosixPath
from urllib.parse import unquote

from .chunks import CLOSING, FENCE, HEADING
from .evidence import _relative_note
from .investigation import _budget, _deadline, _documents, _inspect, _Limit, _policy

MAX_CANDIDATES = 1000
MAX_NOTE_BYTES = 1024**2
LINK = re.compile(
    r"(?<![!\\])\[\[([^\[\]\n]{1,2048})\]\]"
    r"|(?<![!\\])\[([^\[\]\n]{1,512})\]\(([^()\n]{1,2048})\)"
)
INLINE_CODE = re.compile(r"(?<!`)(`+)(?!`).*?(?<!`)\1(?!`)")


def _rows(vault, policy, deadline, *, after="", exact=None):
    vault.assert_retention_ready()
    with vault.connect() as db:
        _deadline(db, deadline)
        clause, arguments = _documents(db, policy)
        return [
            dict(row)
            for row in db.execute(
                f"""SELECT d.path, d.sha256 AS view_sha256, d.project, d.collection, d.accepted
                    FROM indexed_documents d WHERE {clause} AND d.path>?
                    AND (? IS NULL OR d.path=?)
                    ORDER BY d.path LIMIT ?""",
                (*arguments, after, exact, exact, MAX_CANDIDATES + 1),
            )
        ]


def _checked(vault, row, policy, budget, checked):
    path = row["path"]
    if path not in checked:
        try:
            target = vault.root / path
            if target.resolve(strict=True) != target or target.stat().st_size > MAX_NOTE_BYTES:
                checked[path] = None
            else:
                checked[path] = _inspect(vault, row, policy, budget)
        except (OSError, ValueError):
            checked[path] = None
    return checked[path]


def _summary(row, note):
    fields = note["_fields"]
    title = next(
        (line[2:] for line in note["text"].splitlines() if line.startswith("# ")),
        PurePosixPath(row["path"]).stem,
    )
    if row["accepted"]:
        title = fields["knowledge_key"]
    return {
        "path": row["path"],
        "title": title[:240],
        "project": row["project"],
        "collection": row["collection"],
        "kind": fields.get("kind", "note"),
        "sha256": row["view_sha256"],
    }


def browse(vault, *, project, collection, broad, limit, cursor):
    """Return hash-bound navigation metadata, never a note body."""
    policy = _policy(("shared",), project, collection, broad, limit, 1, 1024, 6)
    if limit > 100:
        raise ValueError("Desktop catalogue limit must be at most 100")
    after = _relative_note(cursor, policy["scopes"]) if cursor is not None else ""
    budget = _budget(16 * 1024**2, MAX_CANDIDATES, 6)
    rows = _rows(vault, policy, budget["deadline"], after=after)
    checked, items = {}, []
    projects, collections = set(), set()
    truncated = len(rows) > MAX_CANDIDATES
    try:
        for row in rows[:MAX_CANDIDATES]:
            note = _checked(vault, row, policy, budget, checked)
            if note is not None:
                items.append(_summary(row, note))
            if len(items) > limit:
                break
        navigation = {**policy, "project": None, "collection": None, "broad": True}
        facets = _rows(vault, navigation, budget["deadline"])
        truncated |= len(facets) > MAX_CANDIDATES
        for row in facets[:MAX_CANDIDATES]:
            project_label = row["project"]
            collection_label = row["collection"]
            project_match = broad or project_label in {None, project}
            add_project = project_label is not None and project_label not in projects
            add_collection = (
                project_match
                and collection_label is not None
                and collection_label not in collections
            )
            if (add_project or add_collection) and _checked(
                vault, row, navigation, budget, checked
            ) is not None:
                if add_project:
                    projects.add(project_label)
                if add_collection:
                    collections.add(collection_label)
    except _Limit:
        truncated = True
    more = len(items) > limit or (truncated and len(items) == limit)
    items = items[:limit]
    return {
        "items": items,
        "next_cursor": items[-1]["path"] if more and items else None,
        "projects": sorted(projects),
        "collections": sorted(collections),
        "truncated": truncated,
    }


def _document_lines(text):
    fence = None
    for line in text.splitlines():
        if fence:
            closing = CLOSING.match(line)
            if closing and set(closing[1]) == {fence[0]} and len(closing[1]) >= len(fence):
                fence = None
            continue
        opening = FENCE.match(line)
        if opening:
            fence = opening[1]
            continue
        yield line


def _links(text):
    heading = None
    for line in _document_lines(text):
        if HEADING.match(line):
            heading = line
        if len(line) > 8192 or ("]]" not in line and "](" not in line):
            continue
        for link in LINK.finditer(INLINE_CODE.sub("", line)):
            if link[1]:
                name = link[1].split("|", 1)[0]
                yield name, heading, True
            else:
                yield link[3], heading, False


def _headings(text):
    return [line for line in _document_lines(text) if HEADING.match(line)]


def _heading_key(heading):
    return re.sub(r"[^\w-]+", "-", heading.casefold()).strip("-")


def _indexed_titles(vault, policy, deadline):
    with vault.connect() as db:
        _deadline(db, deadline)
        clause, arguments = _documents(db, policy)
        return {
            row["path"]: row["title"]
            for row in db.execute(
                f"""SELECT d.path, MIN(c.title) AS title
                    FROM chunks c JOIN indexed_documents d ON d.path=c.path
                    WHERE {clause} GROUP BY d.path ORDER BY d.path LIMIT ?""",
                (*arguments, MAX_CANDIDATES),
            )
        }


class _Catalogue:
    """A bounded shared index view with hash verification on every returned note."""

    def __init__(self, vault):
        self.vault = vault
        self.policy = _policy(("shared",), None, None, True, MAX_CANDIDATES, 1, 1024, 8)
        self.budget = _budget(16 * 1024**2, MAX_CANDIDATES, 8)
        self.deadline = self.budget["deadline"]
        rows = _rows(vault, self.policy, self.deadline)
        self.truncated = len(rows) > MAX_CANDIDATES
        self.rows = {row["path"]: row for row in rows[:MAX_CANDIDATES]}
        self.titles = _indexed_titles(vault, self.policy, self.deadline)
        self.checked = {}
        self.names = {}
        for path in self.rows:
            for name in {PurePosixPath(path).stem, self.titles.get(path, "")} - {""}:
                self.names.setdefault(name.casefold(), set()).add(path)

    def inspect(self, path):
        if path not in self.rows:
            rows = _rows(self.vault, self.policy, self.deadline, exact=path)
            if not rows:
                return None
            self.rows[path] = rows[0]
        return _checked(self.vault, self.rows[path], self.policy, self.budget, self.checked)

    def choice(self, path, heading=None):
        inspected = self.inspect(path)
        if inspected is None:
            return None
        summary = _summary(self.rows[path], inspected)
        return {key: summary[key] for key in ("path", "title", "sha256")} | {"heading": heading}

    def choices(self, source, value, wiki):
        if wiki:
            value = value.split("|", 1)[0]
        target, _, fragment = value.partition("#")
        name = unquote(target.strip().strip("<>"))
        fragment = unquote(fragment).strip()
        if any(char in name for char in ":?\\") or name.startswith("//"):
            return []
        if not name and fragment:
            candidates = {source}
        elif not name:
            return []
        elif wiki and "/" not in name:
            if self.truncated:
                return []
            candidates = self.names.get(name.removesuffix(".md").casefold(), set())
        else:
            if wiki and not name.endswith(".md"):
                name += ".md"
            if not name.endswith(".md") or (name.startswith("/") and not wiki):
                return []
            relative = posixpath.normpath(posixpath.join(posixpath.dirname(source), name))
            if name.startswith("/"):
                candidates = {name[1:]}
            elif wiki and not name.startswith("."):
                candidates = {name, relative}
            else:
                candidates = {relative}
        found = []
        for path in sorted(candidates):
            try:
                path = _relative_note(path, self.policy["scopes"])
            except (PermissionError, ValueError):
                continue
            inspected = self.inspect(path)
            if inspected is None:
                continue
            heading = None
            if fragment:
                matches = [
                    line
                    for line in _headings(inspected["text"])
                    if _heading_key(HEADING.match(line)[1]) == _heading_key(fragment)
                ]
                if len(matches) != 1:
                    continue
                heading = matches[0]
            found.append(self.choice(path, heading))
        return found


def resolve_link(vault, path, target, *, wiki):
    source = _relative_note(path, ("shared",))
    if len(target) > 2048:
        raise ValueError("Note link target exceeds 2048 characters")
    catalogue = _Catalogue(vault)
    try:
        if catalogue.inspect(source) is None:
            raise ValueError("Source note is unavailable")
        choices = catalogue.choices(source, target, wiki)
    except _Limit as error:
        raise TimeoutError("Note link resolution reached its read limit") from error
    if len(choices) == 1:
        return {"status": "resolved", **choices[0], "choices": []}
    return {
        "status": "ambiguous" if choices else "unresolved",
        "path": None,
        "heading": None,
        "choices": choices[:30],
    }


def suggest_links(vault, query, limit):
    if not 1 <= limit <= 30 or len(query) > 256:
        raise ValueError("Link suggestions allow 1 to 30 results and a 256-character query")
    name, separator, fragment = query.casefold().partition("#")
    name, fragment = name.strip(), fragment.strip()
    catalogue = _Catalogue(vault)
    items = []
    try:
        for path in list(catalogue.rows):
            if (
                name
                and name not in path.casefold()
                and name not in catalogue.titles.get(path, "").casefold()
            ):
                continue
            inspected = catalogue.inspect(path)
            if inspected is None:
                continue
            headings = [None]
            if separator:
                headings = [
                    line
                    for line in _headings(inspected["text"])
                    if fragment in line.casefold() or fragment in _heading_key(line)
                ]
            for heading in headings:
                items.append(catalogue.choice(path, heading))
                if len(items) > limit:
                    break
            if len(items) > limit:
                break
    except _Limit:
        catalogue.truncated = True
    return {"items": items[:limit], "truncated": catalogue.truncated or len(items) > limit}


def card_refs(vault, paths):
    """Verify only the requested shared card identities, without returning their bodies."""
    if len(paths) > 50:
        raise ValueError("Card references allow at most 50 paths")
    policy = _policy(("shared",), None, None, True, 50, 1, 1024, 8)
    budget = _budget(16 * 1024**2, 50, 8)
    checked, items = {}, []
    for path in dict.fromkeys(paths):
        try:
            canonical = _relative_note(path, policy["scopes"])
        except (PermissionError, ValueError):
            continue
        rows = _rows(vault, policy, budget["deadline"], exact=canonical)
        if not rows:
            continue
        try:
            inspected = _checked(vault, rows[0], policy, budget, checked)
        except _Limit as error:
            raise TimeoutError("Card reference verification reached its read limit") from error
        if inspected is not None:
            summary = _summary(rows[0], inspected)
            items.append({key: summary[key] for key in ("path", "title", "sha256")})
    return {"items": items}


def _linking_rows(vault, policy, deadline):
    with vault.connect() as db:
        _deadline(db, deadline)
        clause, arguments = _documents(db, policy)
        return {
            row["path"]
            for row in db.execute(
                f"""SELECT DISTINCT d.path FROM chunks c
                    JOIN indexed_documents d ON d.path=c.path
                    WHERE {clause}
                    AND (instr(c.body,'[[')>0 OR instr(c.body,'](')>0
                    OR instr(c.heading,'[[')>0 OR instr(c.heading,'](')>0)
                    LIMIT ?""",
                (*arguments, MAX_CANDIDATES + 1),
            )
        }


def _edges(catalogue, policy, rows, limit, root=None):
    edges = {}

    def add(source, target, kind, source_heading, target_heading, sha256=None):
        if (
            target not in rows
            or source == target
            or (root is not None and root not in {source, target})
            or catalogue.inspect(target) is None
        ):
            return
        evidence_path, heading = (
            (target, target_heading) if kind == "evidence" else (source, source_heading)
        )
        digest = rows[evidence_path]["view_sha256"]
        if sha256 is not None and (
            digest != sha256 or heading not in catalogue.checked[evidence_path]["text"].splitlines()
        ):
            return
        key = source, target, kind
        edges.setdefault(
            key,
            {
                "source": source,
                "target": target,
                "kind": kind,
                "source_heading": source_heading,
                "target_heading": target_heading,
                "label": "Reviewed evidence" if kind == "evidence" else "Markdown link",
                "evidence": {"path": evidence_path, "heading": heading, "sha256": digest},
            },
        )

    linking = _linking_rows(catalogue.vault, policy, catalogue.deadline)
    ordered = list(rows) if root is None else [root, *(path for path in rows if path != root)]
    truncated = len(linking) > MAX_CANDIDATES
    try:
        for source in ordered:
            if time.monotonic() >= catalogue.deadline:
                raise _Limit("deadline")
            row = rows[source]
            if source != root and source not in linking and not row["accepted"]:
                continue
            note = catalogue.inspect(source)
            if note is None:
                continue
            for value, heading, wiki in _links(note["text"]):
                choices = catalogue.choices(source, value, wiki)
                if len(choices) == 1:
                    add(source, choices[0]["path"], "link", heading, choices[0]["heading"])
            if row["accepted"]:
                for citation in json.loads(note["_fields"]["review"])["evidence"]:
                    target = citation["path"]
                    add(source, target, "evidence", None, citation["heading"], citation["sha256"])
            if len(edges) > limit:
                truncated = True
                break
    except _Limit:
        truncated = True
    return list(edges.values())[:limit], truncated


def relationships(vault, path, *, project, collection, broad, limit):
    """Resolve explicit links and recorded evidence through the same shared visibility."""
    policy = _policy(("shared",), project, collection, broad, limit, 1, 1024, 8)
    if limit > 50:
        raise ValueError("Relationship limit must be at most 50")
    root = _relative_note(path, policy["scopes"])
    catalogue = _Catalogue(vault)
    selected = _rows(vault, policy, catalogue.deadline)
    rows = {row["path"]: row for row in selected[:MAX_CANDIDATES]}
    if root not in rows:
        matches = _rows(vault, policy, catalogue.deadline, exact=root)
        if not matches:
            raise ValueError("Note is unavailable in the requested catalogue")
        rows[root] = matches[0]
    if catalogue.inspect(root) is None:
        raise ValueError("Note is unavailable in the requested catalogue")
    visible, truncated = _edges(catalogue, policy, rows, limit, root)
    paths = dict.fromkeys([root, *(edge[key] for edge in visible for key in ("source", "target"))])
    return {
        "nodes": [_summary(rows[name], catalogue.checked[name]) for name in paths],
        "edges": visible,
        "count": len(visible),
        "truncated": truncated or catalogue.truncated or len(selected) > MAX_CANDIDATES,
    }


def snapshot(vault, *, project, collection, broad, node_limit, edge_limit):
    policy = _policy(("shared",), project, collection, broad, node_limit, 1, 1024, 8)
    if not 1 <= edge_limit <= 3000:
        raise ValueError("Graph snapshot edge limit must be from 1 to 3000")
    catalogue = _Catalogue(vault)
    selected = _rows(vault, policy, catalogue.deadline)
    rows, nodes = {}, []
    truncated = catalogue.truncated
    try:
        for row in selected:
            path = row["path"]
            note = catalogue.inspect(path)
            if note is None:
                continue
            if len(nodes) == node_limit:
                truncated = True
                break
            nodes.append(_summary(row, note))
            rows[path] = row
    except _Limit:
        truncated = True
    edges, edge_truncated = _edges(catalogue, policy, rows, edge_limit)
    return {
        "nodes": nodes,
        "edges": edges,
        "count": len(edges),
        "truncated": truncated or edge_truncated,
        "projects": sorted({item["project"] for item in nodes if item["project"] is not None}),
        "collections": sorted(
            {item["collection"] for item in nodes if item["collection"] is not None}
        ),
    }
