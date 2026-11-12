"""Owner-reviewed working knowledge with immutable Markdown versions."""

import hashlib
import json
import re
from datetime import UTC, datetime
from uuid import uuid4

from .personal_profile import PROFILE_KEY
from .source_registry import atomic_bytes, single_line
from .store import metadata, now, scope_for

SCOPES = ("shared", "private")
ASSERTION_TYPES = {
    "user_statement",
    "assistant_claim",
    "source_claim",
    "verified_observation",
    "model_inference",
    "unknown",
}


class RevisionConflict(ValueError):
    """An owner review was based on a different accepted revision."""


def _timestamp(value, name):
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(f"{name} must be an explicit ISO timestamp or null")
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise ValueError(f"{name} must specify its timezone; none is inferred")
    return parsed.astimezone(UTC).isoformat()


def _grants(scopes):
    scopes = tuple(scopes)
    if not scopes or any(scope not in SCOPES for scope in scopes):
        raise PermissionError("Working knowledge requires explicit shared/private scope grants")
    return scopes


def _document(info, text, *, proposal):
    title = "Proposed working knowledge" if proposal else "Accepted working knowledge version"
    fields = {
        "kind": "knowledge_proposal" if proposal else "accepted_knowledge",
        "status": "proposed" if proposal else "historical",
        "source": f"review:{info['proposal_id']}",
        "recorded": info["recorded_at"],
        "review": json.dumps(info, ensure_ascii=False, sort_keys=True),
    }
    if not proposal:
        fields.update(
            {"knowledge_key": info["entry_key"], "knowledge_revision": str(info["revision"])}
        )
    head = "---\n" + "".join(f"{name}: {value}\n" for name, value in fields.items())
    return (head + f"---\n# {title}\n\n{text}\n").encode("utf-8")


class AcceptedKnowledge:
    """The gateway supplies authenticated roles and holds the shared mutation gate."""

    def __init__(self, vault):
        self.vault = vault
        with vault.connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS accepted_knowledge_proposals (
                    id TEXT PRIMARY KEY, entry_key TEXT NOT NULL, scope TEXT NOT NULL,
                    path TEXT NOT NULL, sha256 TEXT NOT NULL, expected_revision INTEGER NOT NULL,
                    status TEXT NOT NULL, recorded_at TEXT NOT NULL, accepted_revision INTEGER
                );
                CREATE TABLE IF NOT EXISTS accepted_knowledge_versions (
                    entry_key TEXT NOT NULL, revision INTEGER NOT NULL, scope TEXT NOT NULL,
                    path TEXT NOT NULL UNIQUE, sha256 TEXT NOT NULL, recorded_at TEXT NOT NULL,
                    valid_from TEXT, valid_to TEXT, proposal_id TEXT NOT NULL UNIQUE,
                    PRIMARY KEY(entry_key,revision)
                );
                CREATE TABLE IF NOT EXISTS accepted_knowledge_heads (
                    entry_key TEXT PRIMARY KEY, scope TEXT NOT NULL, revision INTEGER NOT NULL
                );
            """)

    def _read(self, row, *, proposal=False):
        self.vault.assert_retention_ready()
        with self.vault.connect() as db:
            if (
                db.execute("SELECT 1 FROM sqlite_master WHERE name='purge_denials'").fetchone()
                and db.execute(
                    "SELECT 1 FROM purge_denials WHERE kind='knowledge' AND value=?",
                    (row["entry_key"],),
                ).fetchone()
            ):
                raise ValueError("Working knowledge is unavailable")
        path = self.vault.resolve(row["path"])
        content = path.read_bytes()
        if hashlib.sha256(content).hexdigest() != row["sha256"]:
            raise ValueError("Canonical working knowledge failed its SHA-256 integrity check")
        fields, body = metadata(content.decode("utf-8"))
        info = json.loads(fields["review"])
        title = "Proposed working knowledge" if proposal else "Accepted working knowledge version"
        prefix = f"# {title}\n\n"
        if not body.startswith(prefix):
            raise ValueError("Canonical working knowledge has an unexpected document structure")
        return {
            **info,
            "text": body.removeprefix(prefix).removesuffix("\n"),
            "path": row["path"],
            "sha256": row["sha256"],
        }

    def _evidence(self, references, scope, scopes):
        if not isinstance(references, list) or not 1 <= len(references) <= 20:
            raise ValueError("Working knowledge requires 1 to 20 exact evidence references")
        checked = []
        for reference in references:
            if not isinstance(reference, dict) or set(reference) != {"path", "sha256", "heading"}:
                raise ValueError("Evidence references require path, sha256 and exact heading")
            path = self.vault.resolve(reference["path"])
            relative = path.relative_to(self.vault.root)
            if scope == "shared" and scope_for(relative) == "private":
                raise PermissionError(
                    "Shared working knowledge cannot disclose private source evidence"
                )
            note = self.vault.get(relative.as_posix(), scopes)
            if (
                not isinstance(reference["sha256"], str)
                or not re.fullmatch(r"[0-9a-f]{64}", reference["sha256"])
                or hashlib.sha256(path.read_bytes()).hexdigest() != reference["sha256"]
            ):
                raise ValueError("Evidence source bytes do not match the supplied SHA-256")
            heading = single_line(reference["heading"], "evidence heading", 512)
            if not heading.startswith("#") or heading not in note["text"].splitlines():
                raise ValueError("Evidence heading does not identify an exact source heading")
            checked.append(
                {"path": relative.as_posix(), "sha256": reference["sha256"], "heading": heading}
            )
        return checked

    def propose(
        self,
        entry_key,
        text,
        *,
        evidence,
        expected_revision,
        writer,
        scope="shared",
        assertion_type="model_inference",
        observed_at=None,
        valid_from=None,
        valid_to=None,
        scopes=("shared",),
        quantities=None,
    ):
        scopes = _grants(scopes)
        self.vault.assert_retention_ready()
        single_line(entry_key, "working knowledge key", 160)
        single_line(writer, "proposal writer", 160)
        if scope not in scopes or scope not in SCOPES:
            raise PermissionError("Working knowledge scope is not granted")
        if type(expected_revision) is not int or expected_revision < 0:
            raise ValueError("expected_revision must be an explicit non-negative integer")
        if not isinstance(text, str) or not text.strip() or len(text) > 100000:
            raise ValueError("Working knowledge text must contain 1 to 100000 characters")
        if assertion_type not in ASSERTION_TYPES:
            raise ValueError("Working knowledge assertion type is unsupported")
        observed_at = _timestamp(observed_at, "observed_at")
        valid_from = _timestamp(valid_from, "valid_from")
        valid_to = _timestamp(valid_to, "valid_to")
        if valid_from is not None and valid_to is not None and valid_to <= valid_from:
            raise ValueError("valid_to must be later than valid_from; intervals are [from,to)")
        if quantities is not None:
            from .quantities import validate_quantities

            quantities = validate_quantities(quantities)
        if entry_key == PROFILE_KEY:
            if (
                scope != "shared"
                or assertion_type != "user_statement"
                or evidence != []
                or any(
                    value is not None for value in (observed_at, valid_from, valid_to, quantities)
                )
            ):
                raise ValueError(
                    "Personal profiles require an exact shared user statement without evidence or dates"
                )
        else:
            evidence = self._evidence(evidence, scope, scopes)
        identity = uuid4().hex
        info = {
            "schema_version": 1,
            "proposal_id": identity,
            "entry_key": entry_key,
            "scope": scope,
            "expected_revision": expected_revision,
            "writer": writer,
            "assertion_type": assertion_type,
            "observed_at": observed_at,
            "valid_from": valid_from,
            "valid_to": valid_to,
            "recorded_at": now(),
            "evidence": evidence,
            "authority": "unreviewed_proposal",
        }
        if quantities is not None:
            info.update(schema_version=2, quantities=quantities)
        private_draft = scope == "private" or entry_key == PROFILE_KEY
        relative = f"{'private/inbox' if private_draft else 'inbox'}/knowledge-{identity}.md"
        content = _document(info, text, proposal=True)
        with self.vault.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            if (
                db.execute("SELECT 1 FROM sqlite_master WHERE name='purge_denials'").fetchone()
                and db.execute(
                    "SELECT 1 FROM purge_denials WHERE kind='knowledge' AND value=?", (entry_key,)
                ).fetchone()
            ):
                raise ValueError("Working knowledge key is unavailable")
            from .resources import admit_storage

            admit_storage(
                self.vault.home,
                len(content) * 8 + 4 * 1024**2,
                additional_entries=6,
                temporary_bytes=len(content),
                temporary_entries=4,
            )
            atomic_bytes(self.vault.root / relative, content)
            db.execute(
                "INSERT INTO accepted_knowledge_proposals VALUES (?,?,?,?,?,?,?,?,NULL)",
                (
                    identity,
                    entry_key,
                    scope,
                    relative,
                    hashlib.sha256(content).hexdigest(),
                    expected_revision,
                    "pending_review",
                    info["recorded_at"],
                ),
            )
        return {
            "proposal_id": identity,
            "path": relative,
            "status": "pending_review",
            "expected_revision": expected_revision,
        }

    def get_proposal(self, proposal_id, *, scopes=("shared",), include_personal_profile=False):
        scopes = _grants(scopes)
        placeholders = ",".join("?" for _ in scopes)
        with self.vault.connect() as db:
            revoked = self._revocation_sql(db)
            row = db.execute(
                f"SELECT * FROM accepted_knowledge_proposals WHERE id=? AND scope IN ({placeholders}){revoked}",
                (proposal_id, *scopes),
            ).fetchone()
        if row is None or (row["entry_key"] == PROFILE_KEY and not include_personal_profile):
            raise ValueError("Proposal is unavailable for these scope grants")
        return {
            **self._read(row, proposal=True),
            "status": row["status"],
            "accepted_revision": row["accepted_revision"],
        }

    def list_proposals(
        self,
        *,
        state="pending_review",
        scopes=("shared",),
        limit=20,
        include_personal_profile=False,
    ):
        scopes = _grants(scopes)
        if not isinstance(state, str) or state not in {"pending_review", "accepted"}:
            raise ValueError("Proposal state must be pending_review or accepted")
        if type(limit) is not int or not 1 <= limit <= 100:
            raise ValueError("proposal list limit must be between 1 and 100")
        placeholders = ",".join("?" for _ in scopes)
        with self.vault.connect() as db:
            revoked = self._revocation_sql(db)
            return [
                dict(row)
                for row in db.execute(
                    f"SELECT id AS proposal_id,entry_key,scope,path,sha256,expected_revision,status,recorded_at,accepted_revision FROM accepted_knowledge_proposals WHERE status=? AND scope IN ({placeholders}){revoked} AND (? OR entry_key<>?) ORDER BY recorded_at,id LIMIT ?",
                    (state, *scopes, include_personal_profile, PROFILE_KEY, limit),
                )
            ]

    def accept(
        self,
        proposal_id,
        *,
        actor_role,
        reviewed_by,
        review_note,
        expected_revision,
        scopes=("shared",),
    ):
        if actor_role != "owner":
            raise PermissionError("Only an authenticated owner may review working knowledge")
        scopes = _grants(scopes)
        single_line(reviewed_by, "owner reviewer", 160)
        single_line(review_note, "owner review note", 2048)
        if type(expected_revision) is not int or expected_revision < 0:
            raise ValueError("expected_revision must be an explicit non-negative integer")
        placeholders = ",".join("?" for _ in scopes)
        with self.vault.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                f"SELECT * FROM accepted_knowledge_proposals WHERE id=? AND scope IN ({placeholders})",
                (proposal_id, *scopes),
            ).fetchone()
            if row is None:
                raise ValueError("Proposal is unavailable for these scope grants")
            proposal = self._read(row, proposal=True)
            if expected_revision != proposal["expected_revision"]:
                raise RevisionConflict("Owner review must use the proposal's expected revision")
            fingerprint = hashlib.sha256(
                json.dumps(
                    [row["sha256"], expected_revision, reviewed_by, review_note], ensure_ascii=False
                ).encode()
            ).hexdigest()
            if row["status"] == "accepted":
                accepted = db.execute(
                    "SELECT * FROM accepted_knowledge_versions WHERE proposal_id=?", (proposal_id,)
                ).fetchone()
                result = self._read(accepted)
                if result["acceptance_fingerprint"] != fingerprint:
                    raise RevisionConflict("This proposal already has a different owner review")
                return result
            head = db.execute(
                "SELECT * FROM accepted_knowledge_heads WHERE entry_key=?", (proposal["entry_key"],)
            ).fetchone()
            if head is not None and (
                head["scope"] not in scopes or head["scope"] != proposal["scope"]
            ):
                raise PermissionError("Working knowledge entry is unavailable for this scope")
            current = head["revision"] if head else 0
            if current != expected_revision:
                raise RevisionConflict(
                    f"Expected revision {expected_revision}, current revision is {current}"
                )
            personal_profile = proposal["entry_key"] == PROFILE_KEY
            if not personal_profile:
                self._evidence(proposal["evidence"], proposal["scope"], scopes)
            revision = current + 1
            key_hash = hashlib.sha256(proposal["entry_key"].encode()).hexdigest()
            # Hosted callers resolve only the current profile through its fixed virtual note.
            private_version = proposal["scope"] == "private" or personal_profile
            prefix = "private/knowledge" if private_version else "knowledge"
            relative = f"{prefix}/{key_hash}/{revision}.md"
            target = self.vault.root / relative
            info = {
                key: value
                for key, value in proposal.items()
                if key not in {"text", "path", "sha256"}
            }
            info.update(
                {
                    "revision": revision,
                    "recorded_at": now(),
                    "reviewed_by": reviewed_by,
                    "review_note": review_note,
                    "acceptance_fingerprint": fingerprint,
                    "supersedes": None
                    if not current
                    else {"entry_key": proposal["entry_key"], "revision": current},
                    "derived_from": {"proposal_id": proposal_id, "evidence": proposal["evidence"]},
                    "authority": "owner_reviewed_working_knowledge",
                }
            )
            content = _document(info, proposal["text"], proposal=False)
            exists = target.exists()
            if exists:
                if target.is_symlink() or target.resolve() != target.absolute():
                    raise ValueError("Canonical working knowledge cannot contain redirects")
                existing = target.read_bytes()
                previous, _ = metadata(existing.decode("utf-8"))
                prior_info = json.loads(previous["review"])
                if prior_info.get("acceptance_fingerprint") != fingerprint:
                    raise RevisionConflict(
                        "An unregistered canonical version belongs to another owner review"
                    )
                recorded_at = _timestamp(prior_info.get("recorded_at"), "orphan recorded_at")
                if recorded_at is None or not proposal["recorded_at"] <= recorded_at <= now():
                    raise ValueError("Orphan recording time is outside its owner-review interval")
                info["recorded_at"] = recorded_at
                content = _document(info, proposal["text"], proposal=False)
                if existing != content:
                    raise ValueError(
                        "Canonical orphan differs from the exact owner-reviewed proposal"
                    )
            from .resources import admit_storage

            admit_storage(
                self.vault.home,
                len(content) * 8 + 8 * 1024**2,
                additional_entries=8,
                temporary_bytes=len(content),
                temporary_entries=4,
                require_memory=True,
            )
            if not exists:
                atomic_bytes(target, content)
            digest = hashlib.sha256(content).hexdigest()
            db.execute(
                "INSERT INTO accepted_knowledge_versions VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    proposal["entry_key"],
                    revision,
                    proposal["scope"],
                    relative,
                    digest,
                    info["recorded_at"],
                    info["valid_from"],
                    info["valid_to"],
                    proposal_id,
                ),
            )
            db.execute(
                "INSERT INTO accepted_knowledge_heads VALUES (?,?,?) ON CONFLICT(entry_key) DO UPDATE SET revision=excluded.revision",
                (proposal["entry_key"], proposal["scope"], revision),
            )
            db.execute(
                "UPDATE accepted_knowledge_proposals SET status='accepted',accepted_revision=? WHERE id=?",
                (revision, proposal_id),
            )
            self.vault.upsert_document(relative, db=db)
            accepted = db.execute(
                "SELECT * FROM accepted_knowledge_versions WHERE proposal_id=?", (proposal_id,)
            ).fetchone()
            return self._read(accepted)

    def read(self, entry_key, *, as_of=None, scopes=("shared",)):
        scopes = _grants(scopes)
        at = _timestamp(as_of, "as_of") if as_of is not None else now()
        placeholders = ",".join("?" for _ in scopes)
        with self.vault.connect() as db:
            revoked = self._revocation_sql(db)
            rows = db.execute(
                f"SELECT * FROM accepted_knowledge_versions WHERE entry_key=? AND scope IN ({placeholders}){revoked} ORDER BY revision DESC",
                (entry_key, *scopes),
            )
            for row in rows:
                if row["valid_from"] is not None and row["valid_from"] > at:
                    continue
                if row["valid_to"] is not None and row["valid_to"] <= at:
                    return {"status": "not_found"}
                if row["valid_from"] is None and as_of is not None:
                    return {
                        "status": "unknown_validity",
                        "reason": "An accepted revision has no known validity start",
                    }
                return {**self._read(row), "status": "found", "as_of": as_of}
        return {"status": "not_found"}

    def history(self, entry_key, *, scopes=("shared",), limit=20):
        scopes = _grants(scopes)
        if type(limit) is not int or not 1 <= limit <= 100:
            raise ValueError("history limit must be between 1 and 100")
        placeholders = ",".join("?" for _ in scopes)
        with self.vault.connect() as db:
            revoked = self._revocation_sql(db)
            rows = db.execute(
                f"SELECT * FROM accepted_knowledge_versions WHERE entry_key=? AND scope IN ({placeholders}){revoked} ORDER BY revision DESC LIMIT ?",
                (entry_key, *scopes, limit + 1),
            ).fetchall()
        return {
            "records": [self._read(row) for row in rows[:limit]],
            "truncated": len(rows) > limit,
        }

    def list_entries(self, *, scopes=("shared",), limit=20):
        scopes = _grants(scopes)
        if type(limit) is not int or not 1 <= limit <= 100:
            raise ValueError("list limit must be between 1 and 100")
        placeholders = ",".join("?" for _ in scopes)
        with self.vault.connect() as db:
            revoked = self._revocation_sql(db)
            return [
                dict(row)
                for row in db.execute(
                    f"SELECT entry_key,scope,revision FROM accepted_knowledge_heads WHERE scope IN ({placeholders}){revoked} ORDER BY entry_key LIMIT ?",
                    (*scopes, limit),
                )
            ]

    def _revocation_sql(self, db):
        self.vault.assert_retention_ready()
        if not db.execute("SELECT 1 FROM sqlite_master WHERE name='purge_denials'").fetchone():
            return ""
        return " AND entry_key NOT IN (SELECT value FROM purge_denials WHERE kind='knowledge')"
