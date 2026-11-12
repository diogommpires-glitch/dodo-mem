"""Open authorised, hash-bound retained evidence with bounded location metadata."""

import hashlib
import io
import json
import math
import os
import re
import stat
from pathlib import PurePosixPath

from .repositories import CODE_SUFFIXES, decode_code
from .store import scope_for

MAX_SOURCE_BYTES = 100 * 1024**2
MAX_ARTIFACT_BYTES = 256 * 1024**2
MAX_LOCATION_ITEMS = 200
MAX_PREVIEW_CHARS = 12000
NATIVE_PARSERS = {"codex-jsonl-v1", "claude-jsonl-v1"}


def _relative_note(path, scopes):
    if not isinstance(path, str):
        raise TypeError("Source note must be a vault-relative Markdown path")
    value = path.replace("\\", "/")
    relative = PurePosixPath(value)
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or ":" in value
        or relative.as_posix() != value
        or relative.suffix != ".md"
        or any(part.endswith((".", " ")) for part in relative.parts)
        or any(ord(character) < 32 for character in value)
    ):
        raise ValueError("Source note must be a canonical vault-relative Markdown path")
    # Denied scope must not trigger note lookup, existence checks or registry IO.
    if scope_for(PurePosixPath(relative.parts[0].casefold(), *relative.parts[1:])) not in scopes:
        raise PermissionError("Source note is outside the granted scopes")
    return value


def _unlinked(path):
    if (
        any(
            part.is_symlink() or getattr(part, "is_junction", lambda: False)()
            for part in (path, *path.parents)
        )
        or path.resolve(strict=True) != path
    ):
        raise ValueError("Retained evidence paths must not contain links or redirections")


def _signature(value):
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns


def _verified_bytes(path, digest, limit, expected_size=None):
    _unlinked(path)
    before = path.stat()
    if not stat.S_ISREG(before.st_mode) or not 0 <= before.st_size <= limit:
        raise ValueError("Retained evidence exceeds its bounded file size limit")
    with path.open("rb") as stream:
        descriptor = os.fstat(stream.fileno())
        if _signature(before) != _signature(descriptor):
            raise ValueError("Retained evidence changed before reading")
        content = stream.read(limit + 1)
        if (
            len(content) > limit
            or _signature(descriptor) != _signature(os.fstat(stream.fileno()))
            or _signature(before) != _signature(path.stat())
        ):
            raise ValueError("Retained evidence changed during bounded reading")
    _unlinked(path)
    if hashlib.sha256(content).hexdigest() != digest:
        raise ValueError("Retained evidence does not match its recorded source hash")
    if expected_size is not None and len(content) != expected_size:
        raise ValueError("Retained evidence differs from its registered byte count")
    return content


def _binding(vault, note, scopes, digest):
    keys = ("source_id", "source_version", "parser", "record_id")
    if not any(key in note for key in keys):
        return None
    if (
        not all(isinstance(note.get(key), str) for key in keys)
        or not re.fullmatch(r"[0-9a-f]{32}", note["source_id"])
        or note["source_version"] != digest
    ):
        raise ValueError("Managed source metadata does not identify a retained source version")
    with vault.connect() as db:
        if not db.execute("SELECT 1 FROM sqlite_master WHERE name='ingest_jobs'").fetchone():
            raise ValueError("Managed source has no registered ingestion job")
        revoked = vault.source_revocation_sql(db, "s.id", "j.sha256")
        row = db.execute(
            f"""SELECT j.id,j.source_id,j.sha256,j.parser,j.path,j.status,
            s.scope,s.kind,v.original,v.bytes,v.suffix,v.repository_provenance
            FROM ingest_jobs j JOIN sources s ON s.id=j.source_id
            JOIN source_versions v ON v.source_id=j.source_id AND v.sha256=j.sha256
            WHERE j.id=?{revoked}""",
            (note["record_id"],),
        ).fetchone()
    if row is None or row["scope"] not in scopes:
        raise PermissionError("Source version is unavailable in the granted scopes")
    job = dict(row)
    if (
        job["status"] != "indexed"
        or job["path"] != note["path"]
        or any(job[key] != note[key] for key in ("source_id", "parser"))
        or job["sha256"] != digest
        or job["scope"] != scope_for(PurePosixPath(note["path"]))
        or job["original"] != f"originals/{digest}{job['suffix']}"
    ):
        raise ValueError("Managed source differs from its registered original or parser version")
    if job["suffix"] in {".json", ".jsonl"} and (
        job["kind"] != "conversation"
        or job["scope"] != "private"
        or (job["suffix"] == ".jsonl" and job["parser"] not in NATIVE_PARSERS)
        or (job["suffix"] == ".json" and job["parser"] != "conversation-json-v1")
    ):
        raise ValueError("Conversation originals require their registered private parser")
    if job["suffix"] in CODE_SUFFIXES and (
        job["kind"] != "code" or job["parser"] != "code-text-v1"
    ):
        raise ValueError("Code originals require captured repository provenance")
    return job


def _artifact(vault, job, name):
    relative = f"derived/{job['source_id']}/{job['sha256']}/{job['parser']}/{name}"
    with vault.connect() as db:
        row = db.execute(
            """SELECT sha256,bytes FROM source_artifacts
            WHERE path=? AND source_id=? AND source_version=? AND parser=? AND job_id=?""",
            (relative, job["source_id"], job["sha256"], job["parser"], job["id"]),
        ).fetchone()
    if row is None:
        raise ValueError(
            "Message locators require the registered parser artifact; reparse explicitly"
        )
    return json.loads(
        _verified_bytes(vault.home / relative, row["sha256"], MAX_ARTIFACT_BYTES, row["bytes"])
    )


def _range(value, total, context):
    if (
        not isinstance(value, (list, tuple))
        or len(value) != 2
        or any(type(number) is not int for number in value)
        or not 1 <= value[0] <= value[1] <= total
    ):
        raise ValueError(f"Location range must be inclusive integers between 1 and {total}")
    start, end = max(1, value[0] - context), min(total, value[1] + context)
    if end - start + 1 > MAX_LOCATION_ITEMS:
        raise ValueError("Location and context must span at most 200 items")
    return {
        "start": value[0],
        "end": value[1],
        "context_start": start,
        "context_end": end,
        "total": total,
    }


def _messages(vault, job, content, selected, context, conversation, branch):
    if job is None:
        raise ValueError("Message locators require a registered conversation parser")
    parser = job["parser"]
    if parser in NATIVE_PARSERS:
        if conversation is not None or branch is not None:
            raise ValueError(
                "Native message ranges use source order; conversation/branch selectors do not apply"
            )
        normal = _artifact(vault, job, "transcript.json")
        if (
            normal.get("parser") != parser
            or normal.get("source_sha256") != job["sha256"]
            or normal.get("scope") != "private"
        ):
            raise ValueError("Native message artifact does not match its source version")
        records = [
            record
            for record in normal["records"]
            if record["role"] in {"user", "assistant", "tool"}
        ]
        locator = {"kind": "messages", **_range(selected, len(records), context)}
        source_lines = content.count(b"\n") + bool(content and not content.endswith(b"\n"))
        items = []
        for position in range(locator["context_start"], locator["context_end"] + 1):
            record = records[position - 1]
            line = record["line"]
            reference = f"sha256:{job['sha256']}#L{line}"
            if (
                type(line) is not int
                or not 1 <= line <= source_lines
                or record["source_ref"] != reference
            ):
                raise ValueError("Native message artifact has an invalid original-line binding")
            items.append(
                {
                    "message": position,
                    "line": line,
                    "source_ref": reference,
                    "id": record["native_id"],
                }
            )
        locator["items"] = items
        return locator
    if parser != "conversation-json-v1":
        raise ValueError("Message locators require a registered conversation parser")
    if conversation is None or branch is None:
        raise ValueError("Export message ranges require explicit conversation and branch IDs")
    if not isinstance(conversation, str) or not isinstance(branch, str):
        raise TypeError("Conversation and branch IDs must be strings")
    normal = _artifact(vault, job, "conversation.json")
    matches = [
        (index, item)
        for index, item in enumerate(normal["conversations"])
        if item["id"] == conversation
    ]
    if len(matches) != 1:
        raise ValueError("Conversation ID is not present in the retained export")
    index, item = matches[0]
    branches = [value for value in item["branches"] if value["leaf"] == branch]
    if len(branches) != 1:
        raise ValueError("Requested branch ID is not present in the retained conversation")
    nodes = [node for node in branches[0]["nodes"] if node in item["messages"]]
    locator = {
        "kind": "messages",
        "conversation": conversation,
        "branch": branch,
        **_range(selected, len(nodes), context),
    }
    prefix = f"/{index}" if content.removeprefix(b"\xef\xbb\xbf").lstrip().startswith(b"[") else ""
    items = []
    for position in range(locator["context_start"], locator["context_end"] + 1):
        node = nodes[position - 1]
        if item["export_kind"] == "chatgpt_mapping":
            escaped = node.replace("~", "~0").replace("/", "~1")
            pointer = f"{prefix}/mapping/{escaped}/message"
        elif item["export_kind"] == "claude_messages":
            pointer = f"{prefix}/chat_messages/{list(item['nodes']).index(node)}"
        else:
            raise ValueError("Unrecognised retained conversation export kind")
        items.append(
            {
                "message": position,
                "id": item["messages"][node]["id"],
                "node": node,
                "json_pointer": pointer,
            }
        )
    locator["items"] = items
    return locator


def _pdf_region(pdf_page, bbox):
    def finite(values):
        return all(type(value) in {int, float} and math.isfinite(value) for value in values)

    if not isinstance(bbox, (tuple, list)) or len(bbox) != 4 or not finite(bbox):
        raise ValueError("PDF region needs four finite numeric coordinates")
    box = [float(value) for value in pdf_page.mediabox]
    scale = pdf_page.get("/UserUnit", 1)
    if (
        not finite(box)
        or not isinstance(scale, (int, float))
        or isinstance(scale, bool)
        or not math.isfinite(scale)
        or scale <= 0
    ):
        raise ValueError("PDF page has unknown or invalid coordinate units")
    left, bottom, right, top = bbox
    if not box[0] <= left < right <= box[2] or not box[1] <= bottom < top <= box[3]:
        raise ValueError("PDF region must be inside the actual page box")
    rotation = pdf_page.get("/Rotate", 0)
    if (
        not isinstance(rotation, (int, float))
        or isinstance(rotation, bool)
        or not math.isfinite(rotation)
        or rotation % 90 != 0
    ):
        raise ValueError("PDF page has an invalid rotation")
    return {
        "rect": [float(value) for value in bbox],
        "coordinate_space": "pdf_user_space_bottom_left",
        "unit": "pdf_user_unit",
        "user_unit_scale_points": float(scale),
        "page_box": box,
        "rotation": rotation,
    }


def source_location(
    vault,
    path,
    scopes=("shared",),
    page=None,
    *,
    lines=None,
    messages=None,
    conversation=None,
    branch=None,
    context=0,
    bbox=None,
    preview=False,
):
    path = _relative_note(path, scopes)
    if type(preview) is not bool:
        raise ValueError("Source preview must be a boolean")
    if type(context) is not int or not 0 <= context <= 50:
        raise ValueError("Location context must be an integer from 0 to 50")
    if sum(value is not None for value in (page, lines, messages)) > 1:
        raise ValueError("Select one page, line range or message range")
    if bbox is not None and page is None:
        raise ValueError("A PDF region requires an explicit page")
    if context and lines is None and messages is None:
        raise ValueError("Context requires a line or message range")
    if (conversation is not None or branch is not None) and messages is None:
        raise ValueError("Conversation and branch selectors require a message range")
    note = vault.get(path, scopes)
    source = note["source"]
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", source):
        raise ValueError("This note is curated or historical; follow its recorded source reference")
    digest = source.removeprefix("sha256:")
    job = _binding(vault, note, scopes, digest)
    originals = vault.home / "originals"
    _unlinked(originals)
    if job is None:
        files = [
            candidate
            for candidate in originals.glob(digest + ".*")
            if candidate.suffix in {".md", ".txt", ".pdf"}
        ]
        if len(files) != 1:
            raise ValueError("Expected exactly one retained original for this imported source")
        original = files[0]
    else:
        if job["suffix"] not in CODE_SUFFIXES | {".md", ".txt", ".pdf", ".json", ".jsonl"}:
            raise ValueError("Retained original format is unsupported")
        original = vault.home / job["original"]
    content = _verified_bytes(original, digest, MAX_SOURCE_BYTES, job["bytes"] if job else None)
    result = {"note": note["path"], "sha256": digest, "original": str(original)}
    if job is not None:
        result.update(
            source_id=job["source_id"], source_version=job["sha256"], parser=job["parser"]
        )
        if job["parser"] == "code-text-v1":
            provenance = json.loads(job["repository_provenance"] or "null")
            if not isinstance(provenance, dict) or provenance.get("content_sha256") != digest:
                raise ValueError("Code original has no matching captured repository provenance")
            result["repository"] = provenance
    preview_text = None
    preview_reason = None
    if original.suffix == ".pdf":
        from pypdf import PdfReader

        pdf = PdfReader(io.BytesIO(content))
        pages = len(pdf.pages)
        result["pages"] = pages
        if page is not None and (type(page) is not int or not 1 <= page <= pages):
            raise ValueError(f"PDF page must be between 1 and {pages}, got {page!r}")
        if lines is not None or messages is not None:
            raise ValueError("PDF originals require page or region locators")
        if bbox is not None:
            result["region"] = _pdf_region(pdf.pages[page - 1], bbox)
        if preview:
            if bbox is not None:
                preview_reason = "Region text is unavailable. Open the full page to read its text."
            elif pages:
                page = page if page is not None else 1
                preview_text = pdf.pages[page - 1].extract_text() or None
                if preview_text is None:
                    preview_reason = "This page contains no extractable text."
    elif page is not None:
        raise ValueError("Page locators apply only to retained PDFs")
    if page is not None:
        result["page"] = page
    text_preview = preview and original.suffix not in {".pdf", ".json", ".jsonl"}
    if lines is not None or text_preview:
        text = (
            decode_code(content, original.suffix)
            if original.suffix in CODE_SUFFIXES
            else content.decode("utf-8-sig")
        )
        if lines is not None:
            total = (
                content.count(b"\n") + bool(content and not content.endswith(b"\n"))
                if original.suffix == ".jsonl"
                else len(text.splitlines())
            )
            result["locator"] = {"kind": "lines", **_range(lines, total, context)}
            if text_preview:
                selected = text.splitlines(keepends=True)[lines[0] - 1 : lines[1]]
                preview_text = "".join(selected[:-1]) + selected[-1].splitlines()[0]
        elif text_preview:
            preview_text = text
    if messages is not None:
        result["locator"] = _messages(vault, job, content, messages, context, conversation, branch)
    if preview:
        result["preview_text"] = (
            preview_text[:MAX_PREVIEW_CHARS] if preview_text is not None else None
        )
        result["preview_truncated"] = (
            preview_text is not None and len(preview_text) > MAX_PREVIEW_CHARS
        )
        if preview_text is None:
            result["preview_reason"] = preview_reason or "This source has no desktop text preview."
    # Recheck the canonical view and live revocation gate after reading retained evidence.
    if vault.get(path, scopes) != note:
        raise ValueError("Source note changed during source opening; retry against the new version")
    return result
