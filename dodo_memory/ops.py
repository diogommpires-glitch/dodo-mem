import hashlib
import json
import re
import shutil
import sqlite3
import tempfile
from contextlib import closing, contextmanager
from pathlib import Path, PurePosixPath

from .store import Vault, now, read_retention_config, writer_lock


def _require_empty(destination):
    if destination.is_symlink() or destination.resolve() != destination:
        raise ValueError("Backup/restore destination cannot contain redirects")
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise ValueError(f"Destination must be an empty directory, got {destination}")


@contextmanager
def _staging(destination):
    _require_empty(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".dodo-backup-", dir=destination.parent) as temporary:
        stage = Path(temporary) / "payload"
        stage.mkdir()
        yield stage
        _require_empty(destination)
        if destination.exists():
            destination.rmdir()
        stage.replace(destination)


def _inventory(root):
    files = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink() or path.resolve() != path:
            raise ValueError(f"Backup paths must not be symlinks: {path}")
        if path.is_file() and path != root / "manifest.json":
            with path.open("rb") as stream:
                checksum = hashlib.file_digest(stream, "sha256").hexdigest()
            files[path.relative_to(root).as_posix()] = {
                "bytes": path.stat().st_size,
                "sha256": checksum,
            }
    return files


def snapshot_backup(vault, destination):
    """Snapshot canonical files and SQLite; caller must hold the single-writer gate and lock."""
    destination = Path(destination).absolute()
    vault.assert_retention_ready()
    if destination.is_relative_to(vault.home):
        raise ValueError("Backup destination must be outside the memory home")
    from .resources import MIN_FREE_BYTES, ResourceRefused, admit_storage

    admission = admit_storage(vault.home, 0, additional_entries=0)
    volume = destination.parent
    while not volume.exists():
        volume = volume.parent
    required = MIN_FREE_BYTES + admission["owned_accounted_bytes"] * 2 + 8 * 1024**2
    free = shutil.disk_usage(volume).free
    if free < required:
        raise ResourceRefused(
            "backup_destination_headroom", free_bytes=free, required_bytes=required
        )
    with _staging(destination) as stage:
        for folder in ["vault", "originals", "generated", "derived", ".state/native-capture"]:
            root = vault.home / folder
            if not root.exists():
                continue
            for path in [root, *root.rglob("*")]:
                if path.is_symlink() or path.resolve() != path:
                    raise ValueError(f"Backup paths must not be symlinks: {path}")
                if path.is_file():
                    target = stage / path.relative_to(vault.home)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(path, target)
        (stage / ".state").mkdir(exist_ok=True)
        with (
            vault.connect() as source_db,
            closing(sqlite3.connect(stage / ".state/memory.sqlite")) as target_db,
        ):
            source_db.backup(target_db)
        files = _inventory(stage)
        manifest = {
            "format": 1,
            "created": now(),
            "memory_home": str(vault.home),
            "retention_required": vault.retention is not None,
            "files": files,
        }
        (stage / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {"directory": str(destination), "files": len(files), "format": 1}


def initialize_retention(home, *, actor_role, scopes):
    """Explicit offline owner maintenance; construction never creates an empty authority."""
    if actor_role != "owner" or set(scopes) != {"shared", "private"}:
        raise PermissionError("Retention initialization requires the authenticated private owner")
    home = Path(home).resolve()
    if read_retention_config(home) is None:
        raise ValueError("Owner must first configure retention.local.json with external paths")
    with writer_lock(home):
        vault = Vault(home, _retention_setup=True)
        return vault.retention.initialize_authority(actor_role=actor_role, scopes=scopes)


def restore_backup(source, destination, *, current_home=None, authority=None, backup_roots=None):
    """Verify a snapshot and restore into an empty home; never copy provider configuration."""
    source = Path(source).absolute()
    if source.is_symlink() or source.resolve(strict=True) != source:
        raise ValueError("Backup source cannot contain redirects")
    destination = Path(destination).absolute()
    _require_empty(destination)
    if destination.is_relative_to(source):
        raise ValueError("Restore destination must be outside the backup")
    manifest = json.loads((source / "manifest.json").read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("format") != 1:
        raise ValueError("Expected backup manifest format 1")
    files = manifest.get("files")
    if not isinstance(files, dict) or ".state/memory.sqlite" not in files:
        raise ValueError("Backup manifest must inventory files and the SQLite snapshot")
    for name, record in files.items():
        path = PurePosixPath(name)
        if (
            path.is_absolute()
            or path.as_posix() != name
            or ".." in path.parts
            or "\\" in name
            or ":" in name
            or not (
                (
                    len(path.parts) >= 2
                    and path.parts[0] in {"vault", "originals", "generated", "derived"}
                )
                or (len(path.parts) >= 3 and path.parts[:2] == (".state", "native-capture"))
                or name == ".state/memory.sqlite"
            )
        ):
            raise ValueError(f"Invalid backup inventory path: {name!r}")
        if (
            not isinstance(record, dict)
            or set(record) != {"bytes", "sha256"}
            or type(record["bytes"]) is not int
            or record["bytes"] < 0
            or not isinstance(record["sha256"], str)
            or not re.fullmatch(r"[0-9a-f]{64}", record["sha256"])
        ):
            raise ValueError(f"Invalid backup inventory record: {name!r}")
    if _inventory(source) != files:
        raise ValueError("Backup integrity check failed")
    from .retention import PurgeBlocked, assert_restore_allowed, load_authority

    context = current_home if current_home is not None else manifest.get("memory_home")
    if context is None and authority is None:
        raise PurgeBlocked(
            "Legacy restore requires explicit current_home or current external authority"
        )
    context = Path(context).absolute() if context is not None else None
    config = read_retention_config(context) if context is not None else None
    historical_home = manifest.get("memory_home")
    historical_config = (
        read_retention_config(historical_home) if historical_home is not None else None
    )
    if historical_config is not None:
        if config is not None and config != historical_config:
            raise PurgeBlocked(
                "Backup home and requested current home disagree on retention authority"
            )
        config = historical_config
    contexts = {
        path
        for path in (
            context,
            Path(historical_home).absolute() if historical_home is not None else None,
        )
        if path is not None
    }
    for memory_home in contexts:
        live_database = memory_home / ".state/memory.sqlite"
        if live_database.is_file():
            with closing(sqlite3.connect(live_database.as_uri() + "?mode=ro", uri=True)) as db:
                if (
                    db.execute("SELECT 1 FROM sqlite_master WHERE name='purge_events'").fetchone()
                    and db.execute(
                        "SELECT 1 FROM purge_events WHERE state!='complete' LIMIT 1"
                    ).fetchone()
                ):
                    raise PurgeBlocked(
                        "Current memory has an unfinished purge; resume it before restore activation"
                    )
    with closing(
        sqlite3.connect(
            (source / ".state/memory.sqlite").as_uri() + "?mode=ro&immutable=1", uri=True
        )
    ) as db:
        governed = (
            db.execute("SELECT 1 FROM sqlite_master WHERE name='retention_meta'").fetchone()
            is not None
        )
        registered_authority = (
            db.execute("SELECT authority FROM retention_meta WHERE id=1").fetchone()[0]
            if governed
            else None
        )
    if config is not None:
        if (
            authority is not None
            and Path(authority).absolute() != Path(config["authority"]).absolute()
        ):
            raise PurgeBlocked("Restore cannot replace the configured current authority")
        authority, backup_roots = config["authority"], config["backup_roots"]
    if (governed or manifest.get("retention_required")) and authority is None:
        raise PurgeBlocked(
            "Governed backup requires the current external authority even at a fresh destination"
        )
    if context is not None and config is None and (context / ".state/memory.sqlite").is_file():
        with closing(
            sqlite3.connect((context / ".state/memory.sqlite").as_uri() + "?mode=ro", uri=True)
        ) as db:
            if (
                db.execute("SELECT 1 FROM sqlite_master WHERE name='retention_meta'").fetchone()
                and authority is None
            ):
                raise PurgeBlocked("Current memory retention configuration is missing")
    authority_hash = None
    if authority is not None:
        authority = Path(authority).absolute()
        if registered_authority is not None and Path(registered_authority).absolute() != authority:
            raise PurgeBlocked(
                "Authority relocation requires a separate owner-reviewed identity migration"
            )
        load_authority(authority)
        if authority.is_relative_to(destination) or backup_roots is None:
            raise PurgeBlocked(
                "Restore requires an external current authority and explicit backup_roots"
            )
        with authority.open("rb") as stream:
            authority_hash = hashlib.file_digest(stream, "sha256").hexdigest()
        assert_restore_allowed(source, authority)
    from .resources import MIN_FREE_BYTES, ResourceRefused, admit_storage

    admission = admit_storage(source, 0, additional_entries=0)
    volume = destination.parent
    while not volume.exists():
        volume = volume.parent
    required = MIN_FREE_BYTES + admission["owned_accounted_bytes"] * 2 + 8 * 1024**2
    free = shutil.disk_usage(volume).free
    if free < required:
        raise ResourceRefused(
            "restore_destination_headroom", free_bytes=free, required_bytes=required
        )
    with _staging(destination) as stage:
        for name in files:
            target = stage / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source / name, target)
        if _inventory(stage) != files:
            raise ValueError("Backup integrity changed during restore")
        if authority is not None:
            shutil.copyfile(source / "manifest.json", stage / "manifest.json")
            assert_restore_allowed(stage, authority)
            with authority.open("rb") as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != authority_hash:
                    raise PurgeBlocked(
                        "Current revocation authority changed before restore activation"
                    )
            current_config = read_retention_config(context) if context is not None else None
            historical_config = (
                read_retention_config(historical_home) if historical_home is not None else None
            )
            if config not in (current_config, historical_config):
                raise PurgeBlocked(
                    "Current retention configuration changed before restore activation"
                )
            restored_config = {
                "authority": str(authority),
                "backup_roots": [str(Path(path).absolute()) for path in backup_roots],
            }
            (stage / "retention.local.json").write_text(
                json.dumps(restored_config, indent=2), encoding="utf-8"
            )
    return {"directory": str(destination), "files": len(files), "format": 1}
