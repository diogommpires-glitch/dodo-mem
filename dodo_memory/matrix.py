"""Owner-approved Matrix runs backed by the existing Hybrid execution engine."""

import hashlib
import json
import os
import re
import subprocess
import threading
import time
from pathlib import Path
from uuid import uuid4

from .parsing import _owned_process
from .store import defer_busy_claim, now

ROUTING_INSTRUCTION = (
    ' When the user asks to "connect to agent matrix", call connect_agent_matrix. '
    "That phrase alone prepares the connection and starts no work. For a concrete task, "
    "matrix_submit prepares a bounded request; dispatch waits for local owner approval. "
    "Use the returned run ID with matrix_status; never invent model access or worker replies."
)
ACTIVE = {"running", "pause_requested", "stop_requested"}
ENGINE = Path.home() / "Desktop/coding_repos/local-ai-hybrid"


class CancelledBeforeDispatch(InterruptedError):
    pass


def _session(value):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{32}", value):
        raise ValueError("session_id must be the 32-character ID returned by connect_agent_matrix")
    return value


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _write(path, value):
    if path.parent.resolve(strict=True) != path.parent or path.is_symlink():
        raise ValueError("Matrix artifacts cannot use redirected paths")
    temporary = path.with_suffix(".new")
    if temporary.is_symlink():
        raise ValueError("Matrix artifact staging path is redirected")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(_json(value))
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


class AgentMatrix:
    def __init__(self, vault):
        self.vault = vault
        self.stopping = threading.Event()
        self.directory = vault.state / "agent-matrix"
        with vault.connect() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS matrix_runs (
                id TEXT PRIMARY KEY, owner TEXT NOT NULL, request_id TEXT NOT NULL,
                state TEXT NOT NULL, request TEXT NOT NULL, request_sha256 TEXT NOT NULL,
                result TEXT, error TEXT, created TEXT NOT NULL, updated TEXT NOT NULL,
                UNIQUE(owner,request_id))""")
            if "session_id" not in {row[1] for row in db.execute("PRAGMA table_info(matrix_runs)")}:
                db.execute("ALTER TABLE matrix_runs ADD COLUMN session_id TEXT NOT NULL DEFAULT ''")

    def describe(self):
        return {
            "state": "ready",
            "session_id": uuid4().hex,
            "requires_local_approval": True,
            "account_access": "Unverified until a live provider response",
            "context_scope": "shared",
            "modes": ["plan", "work"],
            "manager": "Highest-ranked eligible model and qualified effort",
            "workers": "Bounded Low / Medium / High specialists; Qwen is the sole local writer",
            "native_ultra": "Unqualified: nested agents are not globally ledger-bound",
            "budget": "Dispatch admission only; no guaranteed provider-side spending ceiling",
            "cross_client_join": "Not enabled; run IDs do not grant access",
        }

    def submit(
        self,
        owner,
        request_id,
        question,
        *,
        session_id,
        mode="plan",
        project=None,
        providers=None,
        token_budget=128000,
        allow_uncertain_usage=False,
        budget_seconds=28800,
    ):
        session_id = _session(session_id)
        if not isinstance(request_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", request_id):
            raise ValueError("request_id must contain 1 to 128 letters, numbers, _ or -")
        if not isinstance(question, str) or not question.strip() or len(question) > 8000:
            raise ValueError("question must contain 1 to 8000 characters")
        if mode not in {"plan", "work"}:
            raise ValueError("mode must be plan or work")
        providers = ["codex", "claude"] if providers is None else providers
        if (
            not isinstance(providers, list)
            or not 1 <= len(providers) <= 2
            or any(p not in {"codex", "claude"} for p in providers)
            or len(set(providers)) != len(providers)
        ):
            raise ValueError(
                "providers must name one or both of codex and claude, without duplicates"
            )
        if type(token_budget) is not int or not 1 <= token_budget <= 1000000:
            raise ValueError("token_budget must be an integer from 1 to 1000000")
        if type(allow_uncertain_usage) is not bool:
            raise TypeError("allow_uncertain_usage must be a boolean")
        if type(budget_seconds) is not int or not 1 <= budget_seconds <= 28800:
            raise ValueError("budget_seconds must be an integer from 1 to 28800")
        if project is not None:
            if not isinstance(project, str):
                raise TypeError("project must be an absolute Git root")
            path = Path(project)
            if not path.is_absolute() or not path.is_dir() or path.resolve() != path:
                raise ValueError("project must be a real absolute Git root without redirects")
            probe = subprocess.run(
                ["git", "rev-parse", "--show-toplevel"],
                cwd=path,
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
            if probe.returncode or Path(probe.stdout.strip()).resolve() != path:
                raise ValueError("project must identify the Git root")
            project = str(path)
        if mode == "work" and project is None:
            raise ValueError("Work mode requires a project Git root")
        request = _json(
            {
                "question": question,
                "mode": mode,
                "project": project,
                "providers": providers,
                "token_budget": token_budget,
                "allow_uncertain_usage": allow_uncertain_usage,
                "budget_seconds": budget_seconds,
            }
        )
        digest = hashlib.sha256(request.encode()).hexdigest()
        request_id = session_id + ":" + request_id
        with self.vault.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT id,request_sha256 FROM matrix_runs WHERE owner=? AND request_id=?",
                (owner, request_id),
            ).fetchone()
            if row:
                if row["request_sha256"] != digest:
                    raise ValueError("request_id already identifies a different request")
                run_id = row["id"]
            else:
                run_id = uuid4().hex
                db.execute(
                    "INSERT INTO matrix_runs VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        run_id,
                        owner,
                        request_id,
                        "awaiting_approval",
                        request,
                        digest,
                        None,
                        None,
                        now(),
                        now(),
                        session_id,
                    ),
                )
        return self.get(run_id, owner, session_id=session_id)

    def get(self, run_id, owner, *, session_id=None, admin=False):
        if not isinstance(run_id, str) or not re.fullmatch(r"[0-9a-f]{32}", run_id):
            raise ValueError("run_id must be 32 lowercase hexadecimal characters")
        with self.vault.connect() as db:
            row = db.execute("SELECT * FROM matrix_runs WHERE id=?", (run_id,)).fetchone()
        if (
            row is None
            or not admin
            and (row["owner"] != owner or row["session_id"] != _session(session_id))
        ):
            raise PermissionError("Matrix run does not exist for this client")
        return {
            "run_id": row["id"],
            "session_id": row["session_id"],
            "state": row["state"],
            "request": json.loads(row["request"]),
            "request_sha256": row["request_sha256"],
            "result": json.loads(row["result"]) if row["result"] else None,
            "error": row["error"],
            "created": row["created"],
            "updated": row["updated"],
        }

    def list(self, owner, *, session_id=None, admin=False, limit=20):
        if type(limit) is not int or not 1 <= limit <= 100:
            raise ValueError("limit must be an integer from 1 to 100")
        if not admin:
            _session(session_id)
        with self.vault.connect() as db:
            ids = db.execute(
                "SELECT id FROM matrix_runs WHERE (? OR (owner=? AND session_id=?)) ORDER BY created DESC LIMIT ?",
                (int(admin), owner, session_id, limit),
            ).fetchall()
        return {
            "runs": [self.get(row["id"], owner, session_id=session_id, admin=admin) for row in ids]
        }

    def approve(self, run_id, expected_request_sha256):
        with self.vault.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM matrix_runs WHERE id=?", (run_id,)).fetchone()
            if row is None or row["request_sha256"] != expected_request_sha256:
                raise ValueError("Matrix request changed or no longer exists")
            if row["state"] != "awaiting_approval":
                raise ValueError("Only a request awaiting approval can be approved")
            db.execute(
                "UPDATE matrix_runs SET state='queued',updated=? WHERE id=?", (now(), run_id)
            )
        return self.get(run_id, row["owner"], admin=True)

    def control(self, run_id, action, owner, *, session_id=None, admin=False):
        self.get(run_id, owner, session_id=session_id, admin=admin)
        with self.vault.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            state = db.execute("SELECT state FROM matrix_runs WHERE id=?", (run_id,)).fetchone()[0]
            if action == "pause" and state == "queued":
                target = "paused"
            elif action == "pause" and state == "running":
                target = "pause_requested"
            elif action == "resume" and state == "paused":
                target = "queued"
            elif action == "stop" and state in ACTIVE:
                target = "stop_requested"
            elif action == "stop" and state in {"queued", "paused", "awaiting_approval"}:
                target = "stopped"
            else:
                raise ValueError(
                    f"Cannot {action!r} Matrix run in {state}; approval or reconciliation may be required"
                )
            db.execute(
                "UPDATE matrix_runs SET state=?,updated=? WHERE id=?", (target, now(), run_id)
            )
        return self.get(run_id, owner, session_id=session_id, admin=admin)

    def recover(self):
        with self.vault.connect() as db:
            db.execute(
                "UPDATE matrix_runs SET state='interrupted',error=?,updated=? "
                "WHERE state IN ('running','pause_requested','stop_requested')",
                (
                    "Gateway stopped during execution; usage and side effects need reconciliation; not replayed",
                    now(),
                ),
            )

    def _claim(self):
        with defer_busy_claim(), self.vault.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT * FROM matrix_runs WHERE state='queued' ORDER BY created LIMIT 1"
            ).fetchone()
            if row is None:
                return None
            db.execute(
                "UPDATE matrix_runs SET state='running',updated=? WHERE id=?", (now(), row["id"])
            )
            return dict(row)
        return None

    def shared_context(self, query):
        packet = []
        remaining = 6000
        for row in self.vault.search(
            query[:2000], scopes=("shared",), mode="keyword", broad=True, limit=8
        ):
            raw = (self.vault.root / row["path"]).read_bytes()
            digest = hashlib.sha256(raw).hexdigest()
            source = self.vault.get(row["path"], scopes=("shared",), expected_sha256=digest)
            text = row.get("body", row.get("text", ""))[: min(1200, remaining)]
            if not text:
                continue
            if text not in source["text"]:
                raise ValueError("Matrix evidence changed after search; refresh the source index")
            packet.append(
                {
                    "path": source["path"],
                    "heading": row.get("heading", "")[:256],
                    "text": text,
                    "sha256": digest,
                    "scope": "shared",
                }
            )
            remaining -= len(text)
            if remaining <= 0:
                break
        self.recheck_context(packet)
        return packet

    def recheck_context(self, packet):
        for excerpt in packet:
            self.vault.get(excerpt["path"], scopes=("shared",), expected_sha256=excerpt["sha256"])

    def run_next(self):
        job = self._claim()
        if job is None:
            return False
        run_id = job["id"]
        try:
            self.directory.mkdir(exist_ok=True)
            directory = self.directory / run_id
            directory.mkdir(exist_ok=True)
            if directory.resolve(strict=True) != directory:
                raise ValueError("Matrix run directory is redirected")
            request_path = directory / "request.json"
            if request_path.exists():
                if request_path.is_symlink() or request_path.stat().st_size > 131072:
                    raise ValueError("Invalid Matrix request artifact")
                request = json.loads(request_path.read_text(encoding="utf-8"))
                expected = {k: v for k, v in request.items() if k not in {"run_id", "context"}}
                if _json(expected) != job["request"] or request.get("run_id") != run_id:
                    raise ValueError("Stored Matrix request differs from the approved manifest")
            else:
                request = {**json.loads(job["request"]), "run_id": run_id}
                request["context"] = self.shared_context(request["question"])
                _write(request_path, request)
            self.recheck_context(request["context"])
            python = ENGINE / ".venv/Scripts/python.exe"
            if not python.is_file() or python.resolve() != python:
                raise ValueError("The installed Hybrid Matrix runtime is unavailable or redirected")
            control = directory / "control.json"
            result = directory / ("result-" + uuid4().hex + ".json")
            gate = directory / "start.gate"
            gate.unlink(missing_ok=True)

            def allow_start():
                state = self.get(run_id, job["owner"], admin=True)["state"]
                if self.stopping.is_set() or state == "stop_requested":
                    raise CancelledBeforeDispatch("stopped")
                if state == "pause_requested":
                    raise CancelledBeforeDispatch("paused")
                if state != "running":
                    raise RuntimeError("Matrix run is no longer admitted for execution")

            allow_start()
            _write(control, {"state": "running"})
            command = [
                str(python),
                "-m",
                "local_ai_hybrid.matrix",
                "--request",
                str(request_path),
                "--result",
                str(result),
                "--control",
                str(control),
            ]
            with (
                (directory / "process.log").open("ab") as log,
                _owned_process(
                    command, dict(os.environ), log, gate, before_start=allow_start
                ) as process,
            ):
                cancel_at = None
                previous = None
                while process.poll() is None:
                    state = self.get(run_id, job["owner"], admin=True)["state"]
                    desired = (
                        "stop"
                        if self.stopping.is_set() or state == "stop_requested"
                        else ("pause" if state == "pause_requested" else "running")
                    )
                    if desired != previous:
                        _write(control, {"state": desired})
                        previous = desired
                    if desired != "running":
                        cancel_at = time.monotonic() if cancel_at is None else cancel_at
                        if time.monotonic() - cancel_at > 15:
                            raise RuntimeError(
                                "Matrix cancellation was not confirmed; execution is interrupted"
                            )
                    time.sleep(0.2)
                if process.returncode and not result.exists():
                    raise RuntimeError(f"Hybrid Matrix exited with code {process.returncode}")
            if result.is_symlink() or result.stat().st_size > 2 * 1024**2:
                raise ValueError("Matrix result is redirected or exceeds 2 MiB")
            output = json.loads(result.read_text(encoding="utf-8"))
            if not isinstance(output, dict) or output.get("state") not in {
                "completed",
                "paused",
                "stopped",
                "blocked",
                "uncertain",
            }:
                raise ValueError("Hybrid Matrix returned an invalid terminal state")
            state = output["state"]
            error = output.get("error")
        except CancelledBeforeDispatch as exc:
            state, output, error = str(exc), None, None
        except (OSError, ValueError, KeyError, RuntimeError, subprocess.SubprocessError) as exc:
            state, output, error = "interrupted", None, f"{type(exc).__name__}: {exc}"
        with self.vault.connect() as db:
            db.execute(
                "UPDATE matrix_runs SET state=?,result=?,error=?,updated=? WHERE id=?",
                (state, _json(output) if output else None, error, now(), run_id),
            )
        return True
