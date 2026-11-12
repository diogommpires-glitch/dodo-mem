"""Owner-issued offline evaluation snapshots; private data never becomes a shared corpus."""

import csv
import hashlib
import json
import os
import re
import sqlite3
import stat
import subprocess
from contextlib import closing
from pathlib import Path
from uuid import uuid4

from .backends import verify_engine_corpus
from .evaluation import digest
from .source_registry import atomic_bytes
from .store import writer_lock

VIEW_LIMIT = 16 * 1024**2
TOTAL_VIEW_LIMIT = 128 * 1024**2


def _job_root(home, destination):
    base = Path(home).absolute() / ".state/private-evaluation"
    path = Path(destination).absolute()
    if (
        path == base
        or not path.is_relative_to(base)
        or path.resolve() != path
        or base.resolve() != base
    ):
        raise ValueError(
            "Private evaluation jobs must stay in their unlinked owner state directory"
        )
    if any(
        (parent / "building.json").exists() or (parent / "receipt.json").exists()
        for parent in path.parents
        if parent.is_relative_to(base) and parent != base
    ):
        raise ValueError(
            "Independent owner evaluation jobs cannot be nested inside an existing job"
        )
    return base, path


def _sid():
    result = subprocess.run(
        ["whoami.exe", "/user", "/fo", "csv", "/nh"],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    rows = list(csv.reader(result.stdout.splitlines()))
    sid = rows[0][-1].strip() if len(rows) == 1 else ""
    if not re.fullmatch(r"S-1-[0-9-]+", sid):
        raise PermissionError("Cannot establish the current Windows owner SID")
    return sid


def _permissions(path, *, create=False):
    path = Path(path)
    if path.resolve() != path:
        raise PermissionError("Private evaluation permissions cannot follow links")
    if os.name != "nt":
        if create:
            path.chmod(0o700)
        info = path.stat()
        expected_modes = {0o700} if path.is_dir() else {0o400, 0o600}
        if info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) not in expected_modes:
            raise PermissionError(
                "Private evaluation directory requires owner-only 0700 permissions"
            )
        return {"owner_only": True, "kind": "posix-0700"}
    sid = _sid()
    if create:
        subprocess.run(
            [
                "icacls.exe",
                str(path),
                "/inheritance:r",
                "/grant:r",
                "*" + sid + ":(OI)(CI)F",
                "/remove:g",
                "*S-1-5-18",
                "*S-1-5-32-544",
                "*S-1-3-4",
                "*S-1-1-0",
                "*S-1-5-32-545",
                "*S-1-5-11",
            ],
            check=True,
            capture_output=True,
            timeout=15,
        )
    scratch = path if path.is_dir() else path.parent
    for parent in (path, *path.parents):
        if (parent / "building.json").is_file() or (parent / "receipt.json").is_file():
            scratch = parent / ".state"
            if not scratch.is_dir() or scratch.resolve() != scratch:
                raise PermissionError("Private ACL metadata directory is unavailable or redirected")
            break
    check = scratch / (".owner-acl-" + uuid4().hex)
    try:
        subprocess.run(
            ["icacls.exe", str(path), "/save", str(check), "/q"],
            check=True,
            capture_output=True,
            timeout=15,
        )
        descriptor = check.read_bytes().decode("utf-16")
    finally:
        check.unlink(missing_ok=True)
    aces = [value.split(";") for value in re.findall(r"\(([^()]*)\)", descriptor)]
    if not aces or any(
        len(ace) != 6
        or ace[0] != "A"
        or ace[-1] != sid
        or (path.is_dir() and "ID" in ace[1])
        or ace[2] != "FA"
        for ace in aces
    ):
        raise PermissionError("Private evaluation ACL is not exclusively owned by the current user")
    return {
        "owner_only": True,
        "kind": "windows-owner-acl",
        "owner_sid_sha256": hashlib.sha256(sid.encode()).hexdigest(),
    }


def _directory(path):
    if not path.exists():
        path.mkdir(parents=True, mode=0o700)
        return _permissions(path, create=True)
    return _permissions(path)


def _bounded_bytes(path, limit):
    path = Path(path).absolute()
    if path.resolve() != path:
        raise ValueError("Private retained data cannot redirect through links")
    before = path.stat()
    with path.open("rb") as stream:
        data = stream.read(limit + 1)
        after = os.fstat(stream.fileno())
    if len(data) > limit:
        raise ValueError("Private retained artifact exceeds its bounded-read limit")
    current = path.stat()
    identities = [
        (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns)
        for value in [before, after, current]
    ]
    if len(set(identities)) != 1 or path.resolve() != path:
        raise ValueError("Private retained data changed during capture")
    return data


def _original_hash(path, expected_bytes):
    if path.resolve() != path or expected_bytes > 100 * 1024**2:
        raise ValueError("Private original is outside its retained bounded source contract")
    total, checksum = 0, hashlib.sha256()
    with path.open("rb") as stream:
        before = os.fstat(stream.fileno())
        while total <= expected_bytes:
            block = stream.read(min(1024**2, expected_bytes + 1 - total))
            if not block:
                break
            total += len(block)
            checksum.update(block)
        after = os.fstat(stream.fileno())
    if total != expected_bytes or (before.st_size, before.st_mtime_ns, before.st_ino) != (
        after.st_size,
        after.st_mtime_ns,
        after.st_ino,
    ):
        raise ValueError("Private original changed or exceeds its registered byte count")
    return checksum.hexdigest()


def _receipt(path, manifest_sha, count):
    return {
        "job_id": path.name,
        "scope": "private",
        "sources": count,
        "manifest_sha256": manifest_sha,
        "receipt": str(path / "receipt.json"),
    }


def freeze_private_evaluation(vault, registry, requests, destination, *, principal_scopes):
    """Caller passes server-bound principal scopes, never request-supplied permission metadata."""
    if not {"admin", "private"} <= set(principal_scopes):
        raise PermissionError("Private evaluation export requires the gateway owner before any IO")
    base, destination = _job_root(vault.home, destination)
    if not 1 <= len(requests) <= 60 or len({row["job_id"] for row in requests}) != len(requests):
        raise ValueError("Private export requires 1 to 60 distinct explicitly approved jobs")
    requests = sorted(requests, key=lambda row: row["job_id"])
    jobs, families = [], {}
    for request in requests:
        job = registry.job(request["job_id"], scopes=("private",))
        if job["status"] != "indexed" or job["scope"] != "private" or job["cancel_requested"]:
            raise PermissionError("Only indexed private source jobs may enter an owner evaluation")
        if (job["source_id"], job["sha256"], job["view_sha256"]) != (
            request["source_id"],
            request["source_version"],
            request["view_sha256"],
        ):
            raise ValueError(
                "Private export source/version/view identity differs from the approved job"
            )
        family = request["family_id"]
        if not isinstance(family, str) or not family.strip():
            raise ValueError("Private export requires an approved source family")
        if job["source_id"] in families and families[job["source_id"]] != family:
            raise ValueError("Stable private source revisions cannot have different families")
        families[job["source_id"]] = family
        relative = Path(job["path"])
        if relative.is_absolute() or ".." in relative.parts or relative.parts[0] != "private":
            raise PermissionError("Private view paths must retain their private collection prefix")
        jobs.append((job, family))
    _directory(base)
    permissions = _directory(destination)
    _directory(destination / ".state")
    with writer_lock(destination):
        if os.name != "nt":
            (destination / ".state/writer.lock").chmod(0o600)
        building = destination / "building.json"
        request_sha = digest(requests)
        source_versions = [
            {
                "source_id": job["source_id"],
                "source_version": job["sha256"],
                "ingest_job_id": job["id"],
                "view_sha256": job["view_sha256"],
            }
            for job, _ in jobs
        ]
        retention = {
            "retention_unit": "whole-owner-job",
            "source_ids": sorted(families),
            "source_versions": source_versions,
        }
        building_payload = {
            "schema_version": 1,
            "kind": "owner-private-evaluation-building",
            "scope": "private",
            "job_id": destination.name,
            "requests_sha256": request_sha,
            **retention,
        }
        building_payload["sha256"] = digest(building_payload)
        if building.exists() and json.loads(_bounded_bytes(building, 1024**2)) != building_payload:
            raise ValueError("Private evaluation job belongs to a different approved allowlist")
        if (destination / "receipt.json").exists():
            manifest = verify_private_receipt(vault.home, destination)
            if manifest["requests_sha256"] != request_sha:
                raise ValueError("Completed private evaluation job cannot be replaced")
            return _receipt(destination, digest(manifest), len(manifest["documents"]))
        atomic_bytes(building, json.dumps(building_payload, sort_keys=True).encode())
        documents, artifacts, total = [], [], 0
        corpus = destination / "corpus"
        _directory(corpus)
        for job, family in jobs:
            original = (vault.home / job["original"]).absolute()
            if (
                not original.is_relative_to(vault.home / "originals")
                or _original_hash(original, job["bytes"]) != job["sha256"]
            ):
                raise ValueError("Private original source hash mismatch")
            view = vault.resolve(job["path"])
            content = _bounded_bytes(view, VIEW_LIMIT)
            total += len(content)
            if (
                total > TOTAL_VIEW_LIMIT
                or hashlib.sha256(content).hexdigest() != job["view_sha256"]
            ):
                raise ValueError("Private normalized view hash or total byte budget mismatch")
            target = corpus / job["path"]
            for parent in reversed(list(target.parent.parents)):
                if parent.is_relative_to(corpus):
                    _directory(parent)
            _directory(target.parent)
            atomic_bytes(target, content)
            if os.name != "nt":
                target.chmod(0o600)
            documents.append(
                {
                    "path": job["path"],
                    "source_id": job["source_id"],
                    "source_version": job["sha256"],
                    "sha256": job["view_sha256"],
                    "bytes": len(content),
                    "family_id": family,
                    "scope": "private",
                    "ingest_job_id": job["id"],
                    "parser": job["parser"],
                    "original_reference": job["original"],
                    "kind": job["kind"],
                }
            )
            derived = vault.home / "derived" / job["source_id"] / job["sha256"] / job["parser"]
            for name in ["document.json", "locators.json", "parse-metadata.json"]:
                source = derived / name
                if not source.exists():
                    continue
                data = _bounded_bytes(source, VIEW_LIMIT)
                relative = (
                    Path("provenance") / job["source_id"] / job["sha256"] / job["parser"] / name
                )
                output = destination / relative
                for parent in reversed(list(output.parent.parents)):
                    if parent.is_relative_to(destination):
                        _directory(parent)
                _directory(output.parent)
                atomic_bytes(output, data)
                if os.name != "nt":
                    output.chmod(0o600)
                artifacts.append(
                    {
                        "path": relative.as_posix(),
                        "sha256": hashlib.sha256(data).hexdigest(),
                        "bytes": len(data),
                        "source_id": job["source_id"],
                        "source_version": job["sha256"],
                        "scope": "private",
                    }
                )
        manifest = {
            "schema_version": 1,
            "kind": "owner-private-evaluation",
            "job_id": destination.name,
            "scope": "private",
            "requests_sha256": request_sha,
            **retention,
            "permissions": permissions,
            "documents": documents,
            "artifacts": artifacts,
        }
        encoded = json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2).encode()
        atomic_bytes(destination / "manifest.json", encoded)
        receipt = _receipt(destination, digest(manifest), len(documents))
        atomic_bytes(destination / "receipt.json", json.dumps(receipt, sort_keys=True).encode())
        verify_private_receipt(vault.home, destination)
        return receipt


def verify_private_receipt(home, destination):
    """Offline worker verifies owner protection and exact private snapshot bytes."""
    _, destination = _job_root(home, destination)
    _permissions(destination)
    _permissions(destination / "receipt.json")
    _permissions(destination / "manifest.json")
    receipt = json.loads(_bounded_bytes(destination / "receipt.json", 1024**2))
    manifest = json.loads(_bounded_bytes(destination / "manifest.json", 4 * 1024**2))
    if (
        manifest.get("scope") != "private"
        or manifest.get("kind") != "owner-private-evaluation"
        or digest(manifest) != receipt["manifest_sha256"]
        or receipt["job_id"] != destination.name
    ):
        raise ValueError("Private evaluation manifest hash/identity mismatch")
    assert_private_sources_current(home, manifest)
    verify_engine_corpus(destination / "corpus", manifest)
    provenance = destination / "provenance"
    if provenance.exists() or manifest["artifacts"]:
        verify_engine_corpus(
            provenance,
            {
                "documents": [
                    {"path": Path(entry["path"]).relative_to("provenance").as_posix()}
                    for entry in manifest["artifacts"]
                ]
            },
        )
    for entry in manifest["documents"] + manifest["artifacts"]:
        if entry["scope"] != "private":
            raise PermissionError("Private snapshot entries cannot be reclassified as shared")
        root = destination / "corpus" if entry in manifest["documents"] else destination
        path = (root / entry["path"]).absolute()
        if not path.is_relative_to(root) or path.resolve() != path:
            raise PermissionError("Private snapshot entry escaped its protected job")
        _permissions(path)
        data = _bounded_bytes(path, VIEW_LIMIT)
        if len(data) != entry["bytes"] or hashlib.sha256(data).hexdigest() != entry["sha256"]:
            raise ValueError("Private evaluation snapshot hash mismatch")
    return manifest


def assert_private_sources_current(home, manifest):
    """Read-only source/denial check; creating an evaluation is never import authority."""
    database = Path(home).absolute() / ".state/memory.sqlite"
    if database.resolve(strict=True) != database:
        raise PermissionError("Private source authority database is redirected")
    with closing(sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)) as db:
        db.execute("BEGIN")
        tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        planned = (
            [
                json.loads(row[0])
                for row in db.execute("SELECT plan_json FROM purge_events WHERE state='planned'")
            ]
            if "purge_events" in tables
            else []
        )
        for document in manifest["documents"]:
            row = db.execute(
                "SELECT j.source_id,j.sha256,j.view_sha256,j.path,j.status,j.cancel_requested,s.scope FROM ingest_jobs j JOIN sources s ON s.id=j.source_id JOIN source_versions v ON v.source_id=j.source_id AND v.sha256=j.sha256 WHERE j.id=?",
                (document["ingest_job_id"],),
            ).fetchone()
            expected = (
                document["source_id"],
                document["source_version"],
                document["sha256"],
                document["path"],
                "indexed",
                0,
                "private",
            )
            if row != expected:
                raise PermissionError("Private frozen source is no longer current and authorized")
            if (
                "purge_denials" in tables
                and db.execute(
                    "SELECT 1 FROM purge_denials WHERE (kind='source' AND value=?) OR (kind='path' AND value=?)",
                    (document["source_id"], document["path"]),
                ).fetchone()
            ):
                raise PermissionError("Private frozen source is revoked by a current purge denial")
            if any(
                document["source_id"] in plan.get("source_ids", [])
                or any(
                    version["source_id"] == document["source_id"]
                    and version["sha256"] == document["source_version"]
                    for version in plan.get("versions", [])
                )
                for plan in planned
            ):
                raise PermissionError("Private frozen source is part of a planned purge")
