"""Bounded, revision-checked evidence for Ask Dodo."""

import json
import re
import stat
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

from .personal_context import (
    PROFILE_SUFFIXES,
    PersonalProfile,
    ProfileSource,
    contextual_request,
    profile_sections,
    read_profile_file,
)
from .whole_index import safe_path

MAX_SEEDS = 8
MAX_CANDIDATES = 40
MAX_HOPS = 2
MAX_EXCERPTS = 8
MAX_EVIDENCE_CHARS = 6000
MAX_QUESTION = 2000
MAX_PROFILE_EXCERPTS = 6
PROFILE_EXCERPT_CHARS = 800
EVIDENCE_EDGES = frozenset({"reference", "citation", "duplicate", "commit_parent", "changed_file"})
UNAVAILABLE_STATES = frozenset(
    {"offline", "inaccessible", "protected", "redirected", "placeholder"}
)
PROVIDERS = frozenset({"Codex", "Claude"})
QUESTION_WORDS = frozenset(
    [
        "about",
        "a",
        "an",
        "and",
        "are",
        "can",
        "could",
        "did",
        "do",
        "does",
        "for",
        "from",
        "how",
        "i",
        "in",
        "is",
        "it",
        "me",
        "my",
        "of",
        "on",
        "or",
        "please",
        "should",
        "show",
        "tell",
        "that",
        "the",
        "their",
        "them",
        "this",
        "to",
        "use",
        "was",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "will",
        "with",
        "would",
        "you",
        "your",
    ]
)


class AskError(RuntimeError):
    pass


def _json(value):
    # Keep invisible direction overrides visible in the exact handoff preview.
    return json.dumps(value, ensure_ascii=False, indent=2).translate(
        {code: f"\\u{code:04x}" for code in (*range(0x202A, 0x202F), *range(0x2066, 0x206A))}
    )


@dataclass(frozen=True)
class Link:
    source: str
    target: str
    source_name: str
    target_name: str
    source_digest: str | None
    target_digest: str | None
    kind: str
    evidence: str

    def describe(self):
        return {
            "from": self.source_name,
            "to": self.target_name,
            "kind": self.kind,
            "evidence": json.loads(self.evidence),
        }


@dataclass(frozen=True)
class Excerpt:
    node_id: str
    path: str
    digest: str
    locator: str
    text: str
    limit: int
    start_line: int
    trail: tuple[Link, ...]
    profile: ProfileSource | None = None

    def describe(self, number):
        description = {
            "cite": f"[{number}]",
            "path": self.path,
            "locator": json.loads(self.locator),
            "source_version": self.digest,
            "excerpt": self.text,
            "relationship_path": [link.describe() for link in self.trail],
        }
        if self.profile:
            description["profile"] = {"section": self.profile.section, "basis": self.profile.basis}
        return description


@dataclass(frozen=True)
class EvidenceSnapshot:
    description: dict
    sources: tuple[Excerpt, ...]
    connections: tuple[Link, ...]

    def verify(self, catalogue):
        revalidate(catalogue, self.sources, self.connections)


@dataclass(frozen=True)
class Context:
    node_id: str | None = None
    workspace: Path | None = None


WHOLE_LIBRARY = Context()


def _search_terms(question):
    words = list(
        dict.fromkeys(
            word
            for word in re.findall(r"[^\W_]+", question.casefold())
            if len(word) > 1 and word not in QUESTION_WORDS
        )
    )
    queries = [question]
    if words:
        queries.extend((" ".join(words[:4]), " ".join(words[:2]), *words[:5]))
    return tuple(dict.fromkeys(queries))


def _context_nodes(catalogue, context):
    selected = catalogue.node(context.node_id) if context.node_id else None
    workspace = None
    if context.workspace and context.workspace.is_absolute():
        for path in (context.workspace, *context.workspace.parents):
            workspace = catalogue.resolve_path(path)
            if workspace:
                break
    for node in (selected, workspace):
        if node and node.get("state") not in UNAVAILABLE_STATES:
            scope = node.get("project_id") or (
                node["id"]
                if node.get("kind") in {"root", "folder", "directory", "project"}
                else node.get("parent_id")
            )
            return node, scope
    return None, None


class EvidenceSession:
    """The model can read only the candidates reached by this question."""

    def __init__(self, catalogue, question, context=WHOLE_LIBRARY, *, personal=False, profile=None):
        question = question.strip()
        if not question or len(question) > MAX_QUESTION:
            raise AskError(f"Enter a question of up to {MAX_QUESTION:,} characters.")
        self.catalogue = catalogue
        self.question, requested, overview = contextual_request(question)
        self.personalized = personal and requested
        self.profile_overview = self.personalized and overview
        self.selected, self.scope = _context_nodes(catalogue, context)
        self.candidates = {}
        self.excerpts = []
        self.seed_links = []
        self.unavailable = 0
        self.truncated = False
        self.status = catalogue.status(summary=True)
        if self.personalized:
            self._read_profile(profile if profile is not None else PersonalProfile(), overview)
        if not self.profile_overview:
            self._walk()

    def _read_profile(self, profile, overview):
        sections = profile_sections(self.question, overview=overview, in_project=bool(self.scope))
        try:
            sources = [
                source
                for source in profile.sources()
                if source.enabled and source.section in sections
            ]
        except (OSError, ValueError):
            self.unavailable += 1
            sources = []
        if (
            self.scope
            and "Coding" in sections
            and (project := self.catalogue.node(self.scope))
            and project.get("path")
        ):
            rules = Path(project["path"]) / "AGENTS.md"
            if rules.is_file():
                sources.insert(0, ProfileSource(rules, "Projects"))
        # Read one source per selected section before adding further sources from that section.
        first_in_section = {}
        for source in sources:
            first_in_section.setdefault(source.section, source)
        sources = list(dict.fromkeys((*first_in_section.values(), *sources)))
        words = tuple(
            word
            for word in re.findall(r"[^\W_]+", self.question.casefold())
            if len(word) > 2 and word not in QUESTION_WORDS
        )
        seen = set()
        for source in sources:
            if len(self.excerpts) >= MAX_PROFILE_EXCERPTS:
                self.truncated = True
                break
            path = source.path
            start = 1
            try:
                is_directory = stat.S_ISDIR(safe_path(path).st_mode)
            except (OSError, ValueError):
                self.unavailable += 1
                continue
            if is_directory:
                folder = self.catalogue.resolve_path(path)
                matches = (
                    self.catalogue.search(self.question, limit=8, scope=folder["id"])
                    if folder
                    else ()
                )
                found = next(
                    (
                        match
                        for match in matches
                        if match.get("path")
                        and Path(match["path"]).is_relative_to(path)
                        and Path(match["path"]).suffix.casefold() in PROFILE_SUFFIXES
                        and match.get("state") not in UNAVAILABLE_STATES
                    ),
                    None,
                )
                if not found:
                    self.unavailable += 1
                    continue
                path = Path(found["path"])
                start = found.get("match_start") or 1
            if path in seen:
                continue
            terms = words if not is_directory and not overview else ()
            relationship = source.section == "Relationships"
            evidence = read_profile_file(
                path,
                PROFILE_EXCERPT_CHARS,
                start,
                relationship_terms=terms if relationship else (),
                query_terms=() if relationship else terms,
            )
            if not evidence:
                self.unavailable += 1
                continue
            seen.add(path)
            self.excerpts.append(
                Excerpt(
                    str(path),
                    evidence["path"],
                    evidence["digest"],
                    _json(evidence["locator"]),
                    evidence["text"],
                    PROFILE_EXCERPT_CHARS,
                    evidence["locator"]["start"],
                    (),
                    source,
                )
            )

    def _walk(self):
        seeds = []
        if self.selected:
            seeds.append(self.selected)
        for scope in dict.fromkeys((self.scope, None)):
            for query in _search_terms(self.question):
                for node in self.catalogue.search(query, limit=MAX_SEEDS, scope=scope):
                    if node.get("state") in UNAVAILABLE_STATES:
                        continue
                    existing = next((item for item in seeds if item["id"] == node["id"]), None)
                    if existing is None:
                        seeds.append(node)
                    elif node.get("match_start") and not existing.get("match_start"):
                        seeds[seeds.index(existing)] = node
                    if len(seeds) == MAX_SEEDS:
                        break
                if len(seeds) == MAX_SEEDS:
                    break
            if len(seeds) == MAX_SEEDS:
                break
        queue = deque((node, ()) for node in seeds)
        seed_ids = {node["id"] for node in seeds}
        self.candidates.update((node["id"], (node, ())) for node in seeds)
        while queue:
            node, trail = queue.popleft()
            if len(trail) == MAX_HOPS:
                continue
            rows = self.catalogue.neighbours(
                node["id"], limit=MAX_CANDIDATES, include_inferred=False
            )
            if len(rows) == MAX_CANDIDATES:
                self.truncated = True
            for row in rows:
                linked = row["node"]
                if row["kind"] not in EVIDENCE_EDGES or linked.get("state") in UNAVAILABLE_STATES:
                    continue
                other_seed = not trail and linked["id"] in seed_ids and node["id"] < linked["id"]
                if linked["id"] in self.candidates and not other_seed:
                    continue
                link = Link(
                    node["id"],
                    linked["id"],
                    node["name"],
                    linked["name"],
                    node.get("digest"),
                    linked.get("digest"),
                    row["kind"],
                    _json(row.get("evidence")),
                )
                if other_seed:
                    self.seed_links.append(link)
                    continue
                if len(self.candidates) == MAX_CANDIDATES:
                    self.truncated = True
                    continue
                route = (*trail, link)
                self.candidates[linked["id"]] = (linked, route)
                queue.append((linked, route))

    def fill(self, limit=MAX_EXCERPTS):
        # Alternate seeds and linked sources so a full seed list does not drown out graph evidence.
        ordered = [key for key, (_, trail) in self.candidates.items() if not trail]
        linked = [key for key, (_, trail) in self.candidates.items() if trail]
        keys = []
        for index in range(max(len(ordered), len(linked))):
            keys.extend(group[index] for group in (ordered, linked) if index < len(group))
        for key in keys:
            if len(self.excerpts) >= limit:
                break
            self.read(key, self.candidates[key][0].get("match_start") or 1)

    def read(self, node_id, start_line=1):
        if (
            node_id not in self.candidates
            or type(start_line) is not int
            or not 1 <= start_line <= 1_000_000
        ):
            return None
        node, trail = self.candidates[node_id]
        for index, item in enumerate(self.excerpts, start=1):
            if item.profile and item.path == node.get("path"):
                return item.describe(index)
        for index, item in enumerate(self.excerpts, start=1):
            if item.node_id == node_id and item.start_line == start_line:
                return item.describe(index)
        remaining = MAX_EVIDENCE_CHARS - sum(len(item.text) for item in self.excerpts)
        limit = min(1000, remaining)
        if not limit or len(self.excerpts) >= MAX_EXCERPTS:
            self.truncated = True
            return None
        evidence = self.catalogue.read_evidence(node_id, limit=limit, start_line=start_line)
        if not evidence or not evidence.get("text") or not evidence.get("digest"):
            self.unavailable += 1
            return None
        excerpt = Excerpt(
            node_id,
            evidence["path"],
            evidence["digest"],
            _json(evidence["locator"]),
            evidence["text"][:limit],
            limit,
            start_line,
            trail,
        )
        self.excerpts.append(excerpt)
        return excerpt.describe(len(self.excerpts))

    def describe(self):
        description = {
            "question": self.question,
            "coverage": {
                "index_generation": self.status.get("generation"),
                "index_scanning": bool(self.status.get("scanning")),
                "candidates": len(self.candidates),
                "unavailable_evidence_reads": self.unavailable,
                "bounded_results": self.truncated,
            },
            "sources": [
                source.describe(index) for index, source in enumerate(self.excerpts, start=1)
            ],
            "connections": [link.describe() for link in self.connections()],
        }
        if self.personalized:
            description["coverage"]["personal_sources"] = sum(
                item.profile is not None for item in self.excerpts
            )
        return description

    def connections(self):
        evidence_ids = {item.node_id for item in self.excerpts}
        for item in self.excerpts:
            evidence_ids.update(link.source for link in item.trail)
        return tuple(
            link
            for link in self.seed_links
            if link.source in evidence_ids and link.target in evidence_ids
        )

    def verify(self):
        revalidate(self.catalogue, self.excerpts, self.connections())

    def snapshot(self):
        return EvidenceSnapshot(self.describe(), tuple(self.excerpts), self.connections())


def revalidate(catalogue, sources, connections=()):
    """Every shown or exported excerpt and its path must still have current evidence."""
    for source in sources:
        if source.profile:
            current = read_profile_file(Path(source.path), source.limit, source.start_line)
        else:
            current = catalogue.read_evidence(
                source.node_id, limit=source.limit, start_line=source.start_line
            )
        if (
            not current
            or current["path"] != source.path
            or current["digest"] != source.digest
            or _json(current["locator"]) != source.locator
            or current["text"][: source.limit] != source.text
        ):
            raise AskError("A source changed or became unavailable. Run the question again.")
    checked_links = {}
    checked_versions = {source.node_id: source.digest for source in sources}
    links = {*connections, *(link for item in sources for link in item.trail)}
    for link in links:
        for node_id, name, digest in (
            (link.source, link.source_name, link.source_digest),
            (link.target, link.target_name, link.target_digest),
        ):
            node = catalogue.node(node_id)
            if digest and node_id not in checked_versions:
                verified = catalogue.read_evidence(node_id, limit=1)
                checked_versions[node_id] = verified["digest"] if verified else None
            if (
                not node
                or node.get("state") in UNAVAILABLE_STATES
                or node["name"] != name
                or node.get("digest") != digest
                or checked_versions.get(node_id) != digest
            ):
                raise AskError("A relationship source changed. Run the question again.")
        if link.source not in checked_links:
            checked_links[link.source] = catalogue.neighbours(
                link.source, limit=MAX_CANDIDATES, include_inferred=False
            )
        if not any(
            row["node"]["id"] == link.target
            and row["kind"] == link.kind
            and row["node"]["name"] == link.target_name
            and _json(row.get("evidence")) == link.evidence
            for row in checked_links[link.source]
        ):
            raise AskError("A relationship changed. Run the question again.")


@dataclass(frozen=True)
class EvidencePacket:
    provider: str
    preview: str
    snapshot: EvidenceSnapshot


class Handoffs:
    """One pending preview per panel. Approval consumes it before any I/O."""

    def __init__(self, catalogue, send):
        self.catalogue = catalogue
        self.send = send
        self._lock = Lock()
        self._pending = None

    def prepare(self, provider, question, context=WHOLE_LIBRARY):
        if provider not in PROVIDERS:
            raise AskError("Select Codex or Claude for a hosted request.")
        session = EvidenceSession(self.catalogue, question, context)
        session.fill()
        if not session.excerpts:
            raise AskError("There are no verified excerpts to share for this question.")
        snapshot = session.snapshot()
        snapshot.verify(self.catalogue)
        packet = EvidencePacket(provider, _json(snapshot.description), snapshot)
        with self._lock:
            self._pending = packet
        return packet

    def submit(self, packet, *, approved):
        with self._lock:
            if packet is None or packet is not self._pending:
                raise AskError("That evidence approval expired. Prepare a new preview.")
            self._pending = None
        if approved is not True:
            return None
        packet.snapshot.verify(self.catalogue)
        answer = self.send(packet.provider, packet.preview)
        packet.snapshot.verify(self.catalogue)
        return answer
