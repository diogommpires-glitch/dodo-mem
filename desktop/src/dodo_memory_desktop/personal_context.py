"""Private, user-selected sources for personal context."""

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile

from .configuration import configured_path, source_roots
from .whole_index import default_database, read_bytes

SECTIONS = (
    "Standing preferences",
    "Writing",
    "Coding",
    "Projects",
    "Workflow",
    "Interests",
    "Relationships",
)
BASES = ("confirmed", "inferred", "observed")
COMMAND = r"contextuali[sz]e\s+with\s+mem"
COMMAND_PREFIX = re.compile(rf"^{COMMAND}(?=$|[\s,.!:;?–—])[\s,.!:;?–—]*", re.IGNORECASE)
COMMAND_SUFFIX = re.compile(rf"(?<!\w){COMMAND}[\s.!?]*$", re.IGNORECASE)
PROFILE_SUFFIXES = frozenset({".md", ".markdown", ".txt", ".rst", ".org"})
PROFILE_BYTES = 256 * 1024


def validate_profile_path(path):
    path = configured_path(str(path))
    if not any(path.is_relative_to(root) for root in source_roots("DODO_MEM_LIBRARY_ROOTS")):
        raise ValueError("Personal context must stay inside explicitly configured library roots.")
    return path


SECTION_TERMS = {
    "Writing": r"draft|writ\w*|reword|rewrit\w*|edit|email|message|caption|voice|tone|style|text",
    "Coding": r"cod\w*|bug|debug|implement\w*|refactor\w*|tests?|repo\w*|python|javascript|api|build",
    "Projects": r"project\w*|workspace|repo\w*|priorit\w*|plan\w*|roadmap",
    "Workflow": r"workflow|process|plan\w*|review\w*|tests?|priorit\w*|schedule|routine|work",
    "Interests": r"recommend\w*|suggest\w*|hobb\w*|interests?|like\w*|save\w*|instagram|ideas?|enjoy",
    "Relationships": r"friend\w*|family|relationship\w*|people|talk|reply",
}


@dataclass(frozen=True)
class ProfileSource:
    path: Path
    section: str
    basis: str = "confirmed"
    enabled: bool = True


class PersonalProfile:
    def __init__(self, path: Path | None = None):
        self.path = (
            path if path is not None else default_database().with_name("personal-profile.json")
        )

    def sources(self) -> tuple[ProfileSource, ...]:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return ()
        try:
            sources = tuple(
                ProfileSource(Path(item["path"]), item["section"], item["basis"], item["enabled"])
                for item in data["sources"]
            )
        except (KeyError, TypeError) as error:
            raise ValueError("The personal profile has invalid source records.") from error
        return self._validate(sources)

    @staticmethod
    def _validate(sources):
        sources = tuple(dict.fromkeys(sources))
        if len(sources) > 32:
            raise ValueError("Choose up to 32 personal context sources.")
        for source in sources:
            validate_profile_path(source.path)
            if source.section not in SECTIONS or source.basis not in BASES:
                raise ValueError("Choose a known section and evidence basis.")
            if type(source.enabled) is not bool:
                raise ValueError("Choose whether each source is enabled.")
        return sources

    def save(self, sources):
        sources = self._validate(sources)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "sources": [
                {"path": str(s.path), "section": s.section, "basis": s.basis, "enabled": s.enabled}
                for s in sources
            ]
        }
        with NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=self.path.parent, delete=False
        ) as staged:
            temporary = Path(staged.name)
            json.dump(payload, staged, ensure_ascii=False, indent=2)
        try:
            os.replace(temporary, self.path)
        finally:
            temporary.unlink(missing_ok=True)


def contextual_request(question):
    clean, count = COMMAND_PREFIX.subn("", question.strip(), count=1)
    if not count:
        clean, count = COMMAND_SUFFIX.subn("", question.strip(), count=1)
    if not count:
        return question, False, False
    clean = clean.strip(" ,:;–—").strip()
    overview = not clean.strip(".!?")
    if overview:
        clean = "Summarise my personal memory and distinguish preferences from inferred interests."
    return clean, True, overview


def profile_sections(question, *, overview=False, in_project=False):
    matched = {
        name
        for name, words in SECTION_TERMS.items()
        if re.search(rf"\b(?:{words})\b", question, re.IGNORECASE)
    }
    if in_project:
        matched.add("Projects")
    if overview:
        return SECTIONS
    return tuple(name for name in SECTIONS if name in matched or name == "Standing preferences")


def read_profile_file(path, limit, start_line=1, *, relationship_terms=(), query_terms=()):
    path = validate_profile_path(path)
    if path.suffix.casefold() not in PROFILE_SUFFIXES:
        return None
    try:
        data = read_bytes(path, cap=PROFILE_BYTES)
        lines = data.decode("utf-8-sig").splitlines()
    except (OSError, ValueError):
        return None
    if relationship_terms:
        matches = (
            (
                sum(
                    term in set(re.findall(r"[^\W_]+", line.split("|", 2)[1].casefold()))
                    for term in relationship_terms
                ),
                index,
            )
            for index, line in enumerate(lines, 1)
            if line.startswith("|") and line.count("|") >= 3
        )
        score, line = max(matches, key=lambda item: (item[0], -item[1]), default=(0, 1))
        if score:
            start_line = line
    elif query_terms:
        terms = set(query_terms)
        matches = (
            (len(terms.intersection(re.findall(r"[^\W_]+", line.casefold()))), -index)
            for index, line in enumerate(lines, 1)
        )
        score, negative_line = max(matches, default=(0, -1))
        if score:
            start_line = -negative_line
    parts = lines[start_line - 1 : start_line + 23]
    excerpt = "\n".join(parts)[:limit]
    if not excerpt.strip():
        return None
    return {
        "path": str(path),
        "digest": hashlib.sha256(data).hexdigest(),
        "locator": {"kind": "lines", "start": start_line, "end": start_line + excerpt.count("\n")},
        "text": excerpt,
    }


def shared_draft():
    return (
        "# Shared context draft\n\n"
        "No personal context is shared by default. Add only text you have reviewed "
        "and approved for shared access.\n"
    )
