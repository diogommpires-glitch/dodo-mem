import hashlib
import json
import re
import tempfile
from pathlib import Path

from .store import metadata, now


def capture_checkpoint(vault, session_id, event_id, text, writer, repository=None, commit=None):
    """Retain one immutable client event; caller must hold the single-writer gate and lock."""
    for name, value in [("session", session_id), ("event", event_id)]:
        if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", value):
            raise ValueError(
                f"{name} ID must contain 1 to 128 letters, digits, underscores or hyphens"
            )
    if not isinstance(text, str) or not text.strip() or len(text) > 100_000:
        raise ValueError(
            "Checkpoint text must contain 1 to 100000 characters with non-whitespace text"
        )
    for name, value, limit in [
        ("writer", writer, 160),
        ("repository", repository, 2048),
        ("commit", commit, 128),
    ]:
        if value is None and name != "writer":
            continue
        if (
            not isinstance(value, str)
            or not 1 <= len(value) <= limit
            or value.splitlines() != [value]
            or any(ord(character) < 32 for character in value)
        ):
            raise ValueError(f"Checkpoint {name} must be a single line of 1 to {limit} characters")
    payload = {
        "session_id": session_id,
        "event_id": event_id,
        "text": text,
        "writer": writer,
        "repository": repository,
        "commit": commit,
    }
    request_hash = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    identity = hashlib.sha256(f"{session_id}\0{event_id}".encode()).hexdigest()
    relative = f"episodes/{identity}.md"
    path = vault.resolve(relative)
    source_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    fields = {
        "kind": "episode",
        "source": f"checkpoint:{session_id}/{event_id}",
        "status": "historical",
        "session_id": session_id,
        "event_id": event_id,
        "writer": writer,
        "source_sha256": source_hash,
        "request_sha256": request_hash,
        "verification": "unverified_client_checkpoint",
    }
    fields.update(
        {
            name: value
            for name, value in [("repository", repository), ("commit", commit)]
            if value is not None
        }
    )
    provenance = (
        f"Client: {writer}\n\nSession: {session_id}\n\nEvent: {event_id}\n\n"
        f"Source SHA-256: {source_hash}\n\n"
    )
    if repository is not None:
        provenance += f"Repository: {repository}\n\n"
    if commit is not None:
        provenance += f"Commit: {commit}\n\n"
    body = (
        "# Session checkpoint\n\n"
        "This is a client-supplied historical checkpoint, not reviewed memory. "
        "Claimed actions and validation have not been independently verified.\n\n"
        f"## Provenance\n\n{provenance}"
        f"## Reported event\n\n{text}\n"
    )
    with vault.connect() as db:
        db.execute("BEGIN IMMEDIATE")
        exists = path.exists()
        if exists:
            previous, previous_body = metadata(path.read_bytes().decode("utf-8"))
            if previous_body != body or any(
                previous.get(key) != value for key, value in fields.items()
            ):
                raise ValueError(
                    "Checkpoint event already exists with different content or provenance"
                )
        from .resources import admit_storage

        admit_storage(
            vault.home,
            len(body.encode("utf-8")) * 8 + 4 * 1024**2,
            additional_entries=6,
            temporary_bytes=len(body.encode("utf-8")) * 2,
            temporary_entries=4,
        )
        if not exists:
            path.parent.mkdir(parents=True, exist_ok=True)
            fields["recorded"] = now()
            document = "---\n" + "".join(f"{key}: {value}\n" for key, value in fields.items())
            with tempfile.TemporaryDirectory(prefix=".capture-", dir=path.parent) as temporary:
                stage = Path(temporary) / "event.md"
                stage.write_bytes((document + "---\n" + body).encode("utf-8"))
                stage.replace(path)
        vault.upsert_document(relative, db=db)
    return {
        "path": relative,
        "source_sha256": source_hash,
        "status": "already_captured" if exists else "captured",
    }
