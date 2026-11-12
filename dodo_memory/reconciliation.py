"""Owner-only live-code checks and explicit refresh of selected retained code."""

import hashlib
import json
import subprocess
import time
from pathlib import Path

from .evidence import _relative_note, source_location
from .lifecycle import _git
from .repositories import (
    _Cancelled,
    _linked,
    _path_exclusion,
    _read_code,
    _relative,
    _signature,
    capture_repository,
)
from .store import now

MAX_CHECK_SECONDS = 15
MAX_LIVE_BYTES = 1024**2


def _git_before(root, deadline, *arguments, optional=False):
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("Live code check budget exhausted")
    result = _git(root, *arguments, optional=optional, timeout=min(5, remaining))
    if time.monotonic() >= deadline:
        raise TimeoutError("Live code check budget exhausted")
    return result


def _repository(root, deadline):
    if _linked(root) or root.resolve(strict=True) != root:
        raise ValueError("Approved repository root contains a redirection")
    actual = _git_before(root, deadline, "rev-parse", "--show-toplevel").decode("utf-8").strip()
    if Path(actual).resolve() != root:
        raise ValueError("Live inspection requires the exact configured Git root")
    if _git_before(root, deadline, "rev-parse", "--show-superproject-working-tree").strip():
        raise ValueError("Live inspection excludes submodule roots")
    return root


def _snapshot(root, deadline):
    head = _git_before(root, deadline, "rev-parse", "--verify", "--quiet", "HEAD", optional=True)
    branch = _git_before(
        root, deadline, "symbolic-ref", "--quiet", "--short", "HEAD", optional=True
    )
    status = _git_before(root, deadline, "status", "--porcelain=v1", "-z", "--untracked-files=all")
    index = _git_before(root, deadline, "ls-files", "--stage", "-z")
    return {
        "repository": str(root),
        "head": head.decode("ascii").strip() if head else None,
        "branch": branch.decode("utf-8").strip() if branch else None,
        "dirty": bool(status),
        "status_sha256": hashlib.sha256(status).hexdigest(),
        "index_sha256": hashlib.sha256(index).hexdigest(),
    }


def _observe(root, relative, deadline):
    _repository(root, deadline)
    before = _snapshot(root, deadline)
    exclusion = _path_exclusion(root, relative, ())
    if exclusion:
        return None, exclusion, None
    ignored = _git_before(
        root, deadline, "check-ignore", "--no-index", "-q", "--", relative.as_posix(), optional=True
    )
    if ignored is not None:
        return None, "git_ignored", None
    captured, exclusion, _ = _read_code(
        root / relative,
        MAX_LIVE_BYTES,
        MAX_LIVE_BYTES,
        lambda: time.monotonic() >= deadline,
    )
    if exclusion:
        return None, exclusion, None
    content, signature = captured
    observed = _snapshot(root, deadline)
    if before != observed or _signature((root / relative).stat()) != signature:
        return None, "changed_during_check", None
    return (
        {
            **observed,
            "sha256": hashlib.sha256(content).hexdigest(),
            "bytes": len(content),
            "observed_at": now(),
        },
        None,
        signature,
    )


class CodeReconciler:
    def __init__(self, vault, registry):
        # Construction performs no IO; the method must authorise its caller first.
        self.vault = vault
        self.registry = registry

    def _source(self, path, scopes, admin):
        if admin is not True:
            raise PermissionError("Live code reconciliation requires the owner's local credential")
        path = _relative_note(path, scopes)
        if self.registry.vault is not self.vault:
            raise ValueError("Reconciliation registry must use the same vault authority")
        location = source_location(self.vault, path, scopes)
        if location.get("parser") != "code-text-v1":
            raise ValueError("Live code reconciliation requires a registered code source")
        context = location["repository"]
        root = Path(context["repository"])
        # Match the configured name before stat/resolve/Git can touch an unapproved path.
        if not root.is_absolute() or root not in self.registry.repository_roots:
            raise PermissionError("Recorded repository is not an exact configured root")
        relative = _relative(context["relative_path"])
        return location, root, relative

    def _matching_job(self, source_id, digest, scopes):
        with self.vault.connect() as db:
            revoked = self.vault.source_revocation_sql(db, "s.id", "j.sha256")
            placeholders = ",".join("?" for _ in scopes)
            row = db.execute(
                f"""SELECT j.id,j.path,j.status,v.repository_provenance FROM ingest_jobs j
                JOIN sources s ON s.id=j.source_id
                JOIN source_versions v ON v.source_id=j.source_id AND v.sha256=j.sha256
                WHERE j.source_id=? AND j.sha256=? AND j.parser='code-text-v1'
                AND s.scope IN ({placeholders}){revoked}""",
                (source_id, digest, *scopes),
            ).fetchone()
        return dict(row) if row is not None else None

    def _finish(self, result, requested, scopes):
        source_location(self.vault, requested["note"], scopes)
        selected = result["selected_path"]
        if selected is not None and selected != requested["note"]:
            source_location(self.vault, selected, scopes)
        if "job_id" in result:
            self.registry.job(result["job_id"], scopes)
        return result

    def check(self, path, scopes=("shared",), *, admin=False, latest=True):
        """Re-read live bytes; latest means their exact retained version, not newest insertion."""
        if type(latest) is not bool:
            raise TypeError("latest must be a boolean")
        deadline = time.monotonic() + MAX_CHECK_SECONDS
        requested, root, relative = self._source(path, scopes, admin)
        result = {
            "requested_path": requested["note"],
            "selected_path": requested["note"],
            "source_id": requested["source_id"],
            "source_version": requested["source_version"],
            "selection": "latest_live_bytes" if latest else "requested_historical",
            "status": "unknown",
            "current": False,
            "requires_refresh": False,
            "capture_context": requested["repository"],
            "observed_now": None,
            "context_drift": [],
            "uncertainty": [],
            "reasons": [],
        }
        try:
            observed, exclusion, signature = _observe(root, relative, deadline)
        except (OSError, ValueError, subprocess.TimeoutExpired, _Cancelled) as error:
            result["reasons"] = [
                "check_budget_exhausted"
                if isinstance(error, (TimeoutError, subprocess.TimeoutExpired, _Cancelled))
                else "live_source_unavailable"
            ]
            return self._finish(result, requested, scopes)
        if exclusion:
            result["status"] = (
                "deleted"
                if exclusion == "missing_or_non_regular" and not (root / relative).exists()
                else "unknown"
            )
            result["reasons"] = [exclusion]
            return self._finish(result, requested, scopes)
        result["observed_now"] = observed
        selected = requested
        if latest:
            job = self._matching_job(requested["source_id"], observed["sha256"], scopes)
            if job is None:
                result.update(
                    status="stale", requires_refresh=True, reasons=["live_bytes_not_retained"]
                )
            elif job["status"] != "indexed":
                pending_context = json.loads(job["repository_provenance"] or "null")
                if (
                    not isinstance(pending_context, dict)
                    or pending_context.get("content_sha256") != observed["sha256"]
                ):
                    raise ValueError("Pending code view has no matching captured source context")
                result["capture_context"] = pending_context
                result.update(
                    status="pending_latest"
                    if job["status"] in {"queued", "running"}
                    else "unknown",
                    selected_path=None,
                    source_version=observed["sha256"],
                    job_id=job["id"],
                    job_status=job["status"],
                    reasons=["matching_view_not_indexed"],
                )
            else:
                selected = source_location(self.vault, job["path"], scopes)
                if (
                    selected["source_id"] != requested["source_id"]
                    or selected["source_version"] != observed["sha256"]
                    or selected["repository"]["repository"] != requested["repository"]["repository"]
                    or selected["repository"]["relative_path"] != relative.as_posix()
                ):
                    raise ValueError(
                        "Latest retained view differs from the verified live source identity"
                    )
                result.update(
                    selected_path=selected["note"],
                    source_version=selected["source_version"],
                    capture_context=selected["repository"],
                    status="current",
                    current=True,
                )
        elif observed["sha256"] == requested["source_version"]:
            result.update(status="current", current=True)
        else:
            result.update(
                status="stale", requires_refresh=True, reasons=["historical_bytes_differ"]
            )
        captured = result["capture_context"]
        for key in ("head", "branch", "dirty", "status_sha256", "index_sha256"):
            if key not in captured:
                result["uncertainty"].append(f"captured_{key}_not_recorded")
            elif captured[key] != observed[key]:
                result["context_drift"].append(key)
        # The live file/Git state may change while validating a selected retained view.
        try:
            if (
                _linked(root / relative)
                or _signature((root / relative).stat()) != signature
                or _snapshot(root, deadline)
                != {
                    key: observed[key]
                    for key in (
                        "repository",
                        "head",
                        "branch",
                        "dirty",
                        "status_sha256",
                        "index_sha256",
                    )
                }
            ):
                result.update(status="unknown", current=False, reasons=["changed_during_check"])
        except (OSError, ValueError, subprocess.TimeoutExpired) as error:
            result.update(
                status="unknown",
                current=False,
                reasons=[
                    "check_budget_exhausted"
                    if isinstance(error, (TimeoutError, subprocess.TimeoutExpired))
                    else "live_source_unavailable"
                ],
            )
        return self._finish(result, requested, scopes)

    def refresh(self, path, scopes=("shared",), *, admin=False):
        """Explicitly capture the recorded file's current stable bytes; never run an index worker."""
        location, root, relative = self._source(path, scopes, admin)
        checked = self.check(path, scopes, admin=admin)
        if checked["observed_now"] is None or checked["status"] in {"unknown", "deleted"}:
            raise ValueError("Live source is unavailable or uncertain; refresh was not queued")
        source = self.registry.source(location["source_id"], scopes)
        result = capture_repository(
            self.registry,
            root,
            paths=[relative.as_posix()],
            scope=source["scope"],
            collection=source["collection"],
            owner=source["owner"],
            max_files=1,
            max_bytes=MAX_LIVE_BYTES,
            max_file_bytes=MAX_LIVE_BYTES,
        )
        if result["status"] != "queued" or len(result["jobs"]) != 1:
            raise ValueError("Live source did not produce exactly one refresh job")
        job = result["jobs"][0]
        if job["source_id"] != location["source_id"]:
            raise ValueError("Refreshed file differs from the original logical source identity")
        return {
            "status": "already_indexed" if job["status"] == "indexed" else job["status"],
            "job_id": job["id"],
            "source_id": job["source_id"],
            "source_version": job["sha256"],
            "job_status": job["status"],
            "path": job["path"],
        }
