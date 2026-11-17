"""Validated, local plugin packs and an approval-bound Wasm broker."""

import base64
import ctypes
import hashlib
import io
import json
import os
import re
import stat
import threading
import zipfile
from contextlib import contextmanager
from ctypes import wintypes as w
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from uuid import uuid4

from .configuration import runtime_path
from .plugin_sandbox import Sandbox, SandboxError

EXAMPLES = Path(__file__).parent / "assets" / "plugins"
PACK_SUFFIX = ".dodo-pack.zip"
PACK_LIMIT = 2 * 1024**2
MODULE_LIMIT = 512 * 1024
MANIFEST_LIMIT = 128 * 1024
PERMISSIONS = {"read:related": "Read titles, paths and kinds for visible related notes."}
COLOURS = {"canvas", "rail", "pane", "raised", "selection", "text", "muted", "accent"}
EXECUTABLE = {"command", "panel"}
NAME = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,62}[a-z0-9])?")
VERSION = re.compile(r"(?:0|[1-9][0-9]{0,5})(?:\.(?:0|[1-9][0-9]{0,5})){2}")
ARCHIVE_PATH = re.compile(r"[a-z0-9][a-z0-9._-]*(?:/[a-z0-9][a-z0-9._-]*)*")
DIGEST = re.compile(r"[0-9a-f]{64}")
GENERATION = re.compile(r"[0-9a-f]{32}")
RGB = re.compile(r"#[0-9a-fA-F]{6}")


class PluginError(Exception):
    pass


@dataclass(frozen=True, slots=True)
class PluginVersion:
    id: str
    name: str
    version: str
    permissions: tuple[str, ...]
    entries: tuple[dict, ...]
    sha256: str
    enabled: bool = False
    generation: str | None = None


@contextmanager
def _windows_approval_lock(name):
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    create = kernel.CreateMutexW
    create.argtypes, create.restype = (w.LPVOID, w.BOOL, w.LPCWSTR), w.HANDLE
    wait = kernel.WaitForSingleObject
    wait.argtypes, wait.restype = (w.HANDLE, w.DWORD), w.DWORD
    release = kernel.ReleaseMutex
    release.argtypes, release.restype = (w.HANDLE,), w.BOOL
    close = kernel.CloseHandle
    close.argtypes, close.restype = (w.HANDLE,), w.BOOL
    handle = create(None, False, name)
    if not handle:
        raise PluginError("The Windows plugin approval lock is unavailable.")
    try:
        if wait(handle, 2000) not in {
            0,
            0x80,
        }:  # An abandoned lock still protects the atomic state file.
            raise PluginError("Plugin approval is busy. Try again.")
        try:
            yield
        finally:
            release(handle)
    finally:
        close(handle)


def _object(value, required, optional=()):
    if not isinstance(value, dict) or not set(required) <= value.keys() <= set(required) | set(
        optional
    ):
        raise PluginError("Plugin data has missing or unsupported fields.")
    return value


def _text(value, maximum, *, multiline=False):
    if (
        not isinstance(value, str)
        or not value.strip()
        or len(value) > maximum
        or any(ord(char) < 32 and (not multiline or char not in "\n\r\t") for char in value)
    ):
        raise PluginError(f"Expected non-empty text of at most {maximum} characters.")
    return value


def _matching(value, pattern, name):
    if not isinstance(value, str) or not pattern.fullmatch(value):
        raise PluginError(f"Invalid plugin {name}.")
    return value


def _pairs(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise PluginError("Plugin JSON contains duplicate fields.")
        value[key] = item
    return value


def _json(content):
    try:
        return json.loads(content, object_pairs_hook=_pairs)
    except (ValueError, UnicodeError, RecursionError) as error:
        raise PluginError("Plugin JSON is invalid.") from error


def _archive_path(name):
    if len(name) > 160 or any(part.endswith(".") for part in name.split("/")):
        raise PluginError("Plugin archive member path is invalid.")
    return _matching(name, ARCHIVE_PATH, "archive member path")


def _luminance(colour):
    channels = [int(colour[index : index + 2], 16) / 255 for index in (1, 3, 5)]
    linear = [
        value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4
        for value in channels
    ]
    return sum(
        value * weight for value, weight in zip(linear, (0.2126, 0.7152, 0.0722), strict=True)
    )


def _theme(value):
    tokens = {
        key: _matching(colour, RGB, "theme colour").upper()
        for key, colour in _object(value, COLOURS).items()
    }
    for foreground in ("text", "muted", "accent"):
        for surface in ("canvas", "rail", "pane", "raised", "selection"):
            low, high = sorted((_luminance(tokens[foreground]), _luminance(tokens[surface])))
            if (high + 0.05) / (low + 0.05) < 4.5:
                raise PluginError(f"Theme needs 4.5:1 contrast between {foreground} and {surface}.")
    return tokens


def _entry(value, assets):
    if not isinstance(value, dict):
        raise PluginError("Expected a plugin entry object.")
    common = ("id", "title", "kind")
    kind = value.get("kind")
    entry = {
        "id": _matching(value.get("id"), NAME, "entry ID"),
        "title": _text(value.get("title"), 100),
        "kind": kind,
    }
    if kind == "theme":
        _object(value, (*common, "tokens"))
        entry["tokens"] = _theme(value["tokens"])
    elif kind == "saved_view":
        _object(value, (*common, "query"), ("project", "collection"))
        entry["query"] = _text(value["query"], 300)
        entry.update(
            {key: _text(value[key], 160) for key in ("project", "collection") if key in value}
        )
    elif kind == "note_template":
        _object(value, (*common, "note_title", "text"))
        entry["note_title"] = _text(value["note_title"], 160)
        entry["text"] = _text(value["text"], 24_000, multiline=True)
    elif kind == "canvas_template":
        from .canvas_store import validate_board

        _object(value, (*common, "document"))
        try:
            board = validate_board(value["document"])
        except (ValueError, TypeError, KeyError) as error:
            raise PluginError("The plugin Canvas document is invalid.") from error
        if any(node["type"] not in {"text", "group"} for node in board["nodes"]):
            raise PluginError("Plugin Canvas templates may contain only text and group cards.")
        entry["document"] = board
    elif isinstance(kind, str) and kind in EXECUTABLE:
        _object(value, (*common, "module", "permission"))
        module = _archive_path(_text(value["module"], 160))
        if module not in assets:
            raise PluginError("The executable entry needs a declared Wasm module.")
        permission = value["permission"]
        if permission is not None and (
            not isinstance(permission, str) or permission not in PERMISSIONS
        ):
            raise PluginError("Executable entry requests an unknown permission.")
        entry.update(module=module, permission=permission)
    else:
        raise PluginError("Unsupported plugin entry type.")
    return entry


def _read_pack(content):
    if not 1 <= len(content) <= PACK_LIMIT:
        raise PluginError("Plugin pack must be at most 2 MiB.")
    try:
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            members = archive.infolist()
            if len(members) > 18 or sum(info.file_size for info in members) > PACK_LIMIT:
                raise PluginError("Plugin archive expands beyond its limits.")
            assets = {}
            for member in members:
                name = _archive_path(member.filename)
                if (
                    name in assets
                    or member.flag_bits & 1
                    or stat.S_IFMT(member.external_attr >> 16) not in {0, stat.S_IFREG}
                ):
                    raise PluginError(
                        "Plugin archive includes a duplicate, encrypted or redirected member."
                    )
                maximum = MANIFEST_LIMIT if name == "manifest.json" else MODULE_LIMIT
                if member.file_size > maximum:
                    raise PluginError("Plugin archive member exceeds its limit.")
                with archive.open(member) as stream:
                    data = stream.read(maximum + 1)
                if len(data) != member.file_size:
                    raise PluginError("Plugin archive member length changed.")
                assets[name] = data
    except (zipfile.BadZipFile, RuntimeError, NotImplementedError, OSError) as error:
        raise PluginError("The plugin pack could not be read as a ZIP archive.") from error
    manifest = _object(
        _json(assets.pop("manifest.json", b"")),
        ("schema", "id", "name", "version", "permissions", "entries", "files"),
    )
    if type(manifest["schema"]) is not int or manifest["schema"] != 1:
        raise PluginError("Unsupported plugin manifest schema.")
    files = _object(manifest["files"], assets)
    for path, data in assets.items():
        if (
            not path.endswith(".wasm")
            or not data.startswith(b"\0asm\x01\0\0\0")
            or _matching(files[path], DIGEST, "module digest") != hashlib.sha256(data).hexdigest()
        ):
            raise PluginError("Declared Wasm module does not match the pack manifest.")
    permissions = manifest["permissions"]
    if (
        not isinstance(permissions, list)
        or not all(isinstance(item, str) for item in permissions)
        or len(permissions) != len(set(permissions))
        or set(permissions) - PERMISSIONS.keys()
    ):
        raise PluginError("Plugin permissions must be recognised and unique.")
    entries = manifest["entries"]
    if not isinstance(entries, list) or not 1 <= len(entries) <= 24:
        raise PluginError("Plugin needs between one and 24 entries.")
    entries = tuple(_entry(entry, assets) for entry in entries)
    executable = [entry for entry in entries if entry["kind"] in EXECUTABLE]
    if (
        len({entry["id"] for entry in entries}) != len(entries)
        or set(permissions) != {entry["permission"] for entry in executable if entry["permission"]}
        or set(assets) != {entry["module"] for entry in executable}
    ):
        raise PluginError("Plugin entry IDs, modules or permissions do not match the manifest.")
    return PluginVersion(
        id=_matching(manifest["id"], NAME, "ID"),
        name=_text(manifest["name"], 100),
        version=_matching(manifest["version"], VERSION, "version"),
        permissions=tuple(sorted(permissions)),
        entries=entries,
        sha256=hashlib.sha256(content).hexdigest(),
    ), assets


def _read_file(path, maximum):
    try:
        with path.open("rb") as stream:
            return stream.read(maximum + 1)
    except OSError as error:
        raise PluginError("A plugin file is unavailable.") from error


def _atomic(path, content):
    staging = path.with_name(f".{uuid4().hex}.tmp")
    try:
        with staging.open("xb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(staging, path)
    except OSError as error:
        raise PluginError("The plugin change could not be saved.") from error
    finally:
        staging.unlink(missing_ok=True)


def _context(permission, context):
    if permission is None:
        return {}
    if not isinstance(context, dict) or not isinstance(context.get("items"), list):
        raise PluginError("The related notes context is unavailable.")
    items = []
    for row in context["items"][:40]:
        _object(row, ("title", "path", "kind"))
        path = _text(row["path"], 360)
        parts = PurePosixPath(path).parts
        if (
            not parts
            or path.startswith("/")
            or "\\" in path
            or ":" in path
            or ".." in parts
            or path != PurePosixPath(path).as_posix()
            or parts[0].casefold() in {"private", "archive", "inbox", "secrets"}
        ):
            raise PluginError("A related note is outside shared memory.")
        items.append(
            {"title": _text(row["title"], 240), "path": path, "kind": _text(row["kind"], 80)}
        )
    return {"items": items}


def _result(entry, result, context):
    if entry["kind"] == "command":
        _object(result, ("kind", "text"))
        if result["kind"] != "message":
            raise PluginError("Command output must be a message.")
        return {"kind": "message", "text": _text(result["text"], 1500, multiline=True)}
    _object(result, ("kind", "title"), ("rows", "select"))
    if result["kind"] != "panel" or (("rows" in result) == ("select" in result)):
        raise PluginError("Panel output must provide rows or select related notes.")
    title = _text(result["title"], 100)
    items = context.get("items", [])
    if "select" in result:
        if result["select"] != "related" or entry["permission"] != "read:related":
            raise PluginError("Plugin may select only its approved related notes.")
        rows = [
            {"label": item["title"], "value": item["kind"], "path": item["path"]} for item in items
        ]
    else:
        if not isinstance(result["rows"], list) or len(result["rows"]) > 40:
            raise PluginError("A plugin panel can contain at most 40 rows.")
        rows = []
        allowed = {item["path"] for item in items}
        for row in result["rows"]:
            _object(row, ("label", "value"), ("path",))
            clean = {
                "label": _text(row["label"], 240),
                "value": _text(row["value"], 600, multiline=True),
            }
            if "path" in row:
                if _text(row["path"], 360) not in allowed:
                    raise PluginError("Plugin tried to open a note outside its approved input.")
                clean["path"] = row["path"]
            rows.append(clean)
    return {"kind": "panel", "title": title, "rows": rows}


class PluginStore:
    def __init__(self, root: Path | None = None):
        self.root = Path(root) if root is not None else runtime_path("desktop") / "plugins"
        self.root = self.root.absolute()
        self._lock = threading.RLock()
        self._sandbox = Sandbox()
        digest = hashlib.sha256(str(self.root.resolve()).casefold().encode()).hexdigest()
        self._mutex_name = f"Local\\Dodo.Plugin.Approval.{digest}"

    def close(self):
        self._sandbox.close()

    @contextmanager
    def _guard(self):
        with self._lock:
            if os.name == "nt":
                with _windows_approval_lock(self._mutex_name):
                    yield
            else:
                yield

    def _directory(self):
        try:
            self.root.mkdir(parents=True, exist_ok=True)
            if self.root.resolve(strict=True) != self.root:
                raise PluginError("The plugin directory cannot be redirected.")
        except OSError as error:
            raise PluginError("The plugin directory is unavailable.") from error

    def _path(self, plugin_id, version):
        plugin_id = _matching(plugin_id, NAME, "ID")
        version = _matching(version, VERSION, "version")
        return self.root / f"{plugin_id}--{version}{PACK_SUFFIX}"

    def _state(self):
        self._directory()
        path = self.root / "state.json"
        if not path.exists():
            return {"active": {}, "previous": {}}
        if path.resolve(strict=True) != path:
            raise PluginError("Plugin approval state cannot be redirected.")
        state = _object(_json(_read_file(path, MANIFEST_LIMIT)), ("active", "previous"))
        upgraded = False
        for entries in state.values():
            if not isinstance(entries, dict):
                raise PluginError("Plugin approval state is invalid.")
            for plugin_id, approval in entries.items():
                _matching(plugin_id, NAME, "ID")
                _object(approval, ("version", "sha256", "permissions"), ("generation",))
                _matching(approval["version"], VERSION, "version")
                _matching(approval["sha256"], DIGEST, "approval digest")
                if not isinstance(approval["permissions"], list) or any(
                    not isinstance(permission, str) or permission not in PERMISSIONS
                    for permission in approval["permissions"]
                ):
                    raise PluginError("Plugin approval state contains an invalid permission.")
                if "generation" not in approval:
                    approval["generation"] = uuid4().hex
                    upgraded = True
                _matching(approval["generation"], GENERATION, "approval generation")
        if upgraded:
            self._save(state)
        return state

    def _save(self, state):
        content = json.dumps(state, sort_keys=True, ensure_ascii=True, indent=2).encode()
        if len(content) > MANIFEST_LIMIT:
            raise PluginError("Plugin approval state exceeds its limit.")
        _atomic(self.root / "state.json", content)

    @staticmethod
    def _binding(plugin):
        return {
            "version": plugin.version,
            "sha256": plugin.sha256,
            "permissions": list(plugin.permissions),
        }

    @classmethod
    def _matches(cls, approval, plugin):
        return approval is not None and all(
            approval[key] == value for key, value in cls._binding(plugin).items()
        )

    def _read(self, path):
        if not path.is_file() or path.resolve(strict=True) != path:
            raise PluginError("The installed plugin is unavailable or redirected.")
        plugin, assets = _read_pack(_read_file(path, PACK_LIMIT))
        if self._path(plugin.id, plugin.version) != path:
            raise PluginError("Installed plugin identity does not match its filename.")
        return plugin, assets

    def install(self, source: Path) -> PluginVersion:
        content = _read_file(Path(source), PACK_LIMIT)
        plugin, _ = _read_pack(content)
        with self._guard():
            self._directory()
            destination = self._path(plugin.id, plugin.version)
            if destination.exists():
                old, _ = self._read(destination)
                if old.sha256 != plugin.sha256:
                    raise PluginError(
                        "This plugin version is already installed with different bytes."
                    )
            else:
                _atomic(destination, content)
        return plugin

    def versions(self) -> list[PluginVersion]:
        with self._guard():
            state = self._state()
            versions = []
            for path in self.root.glob(f"*{PACK_SUFFIX}"):
                try:
                    plugin, _ = self._read(path)
                except PluginError:
                    continue  # Modified or corrupt packs never reach the execution path.
                approval = state["active"].get(plugin.id)
                if self._matches(approval, plugin):
                    plugin = replace(plugin, enabled=True, generation=approval["generation"])
                versions.append(plugin)
        return sorted(
            versions, key=lambda item: (item.id, tuple(map(int, item.version.split("."))))
        )

    def active(self) -> list[PluginVersion]:
        return [plugin for plugin in self.versions() if plugin.enabled]

    def enable(self, plugin_id, version, permissions, *, expected_sha256: str | None = None):
        with self._guard():
            state = self._state()
            plugin, _ = self._read(self._path(plugin_id, version))
            if set(permissions) != set(plugin.permissions):
                raise PluginError("Approve the exact permissions requested by this plugin version.")
            if expected_sha256 is not None and expected_sha256 != plugin.sha256:
                raise PluginError("The plugin bytes changed before they could be approved.")
            previous = state["active"].get(plugin_id)
            if previous and not self._matches(previous, plugin):
                state["previous"][plugin_id] = previous
            state["active"][plugin_id] = {**self._binding(plugin), "generation": uuid4().hex}
            self._save(state)

    def disable(self, plugin_id):
        with self._guard():
            state = self._state()
            state["active"].pop(plugin_id, None)
            self._save(state)

    def _previous(self, plugin_id, state):
        approval = state["previous"].get(plugin_id)
        if approval is None:
            return None
        plugin, _ = self._read(self._path(plugin_id, approval["version"]))
        if not self._matches(approval, plugin):
            raise PluginError("The previously approved plugin bytes have changed.")
        return plugin

    def previous(self, plugin_id) -> PluginVersion | None:
        with self._guard():
            return self._previous(plugin_id, self._state())

    def rollback(self, plugin_id) -> PluginVersion:
        with self._guard():
            state = self._state()
            plugin = self._previous(plugin_id, state)
            if plugin is None:
                raise PluginError("There is no previously approved version to restore.")
            state["previous"].pop(plugin_id)
            generation = uuid4().hex
            state["active"][plugin_id] = {**self._binding(plugin), "generation": generation}
            self._save(state)
            return replace(plugin, enabled=True, generation=generation)

    def run(
        self,
        plugin_id,
        action,
        context,
        *,
        expected_sha256: str | None = None,
        expected_generation: str | None = None,
    ) -> dict:
        with self._guard():
            approval = self._state()["active"].get(plugin_id)
            if approval is None:
                raise PluginError("Enable this plugin before running it.")
            plugin, assets = self._read(self._path(plugin_id, approval["version"]))
            if not self._matches(approval, plugin):
                raise PluginError("Plugin bytes or permissions changed after approval.")
            if expected_sha256 is not None and expected_sha256 != plugin.sha256:
                raise PluginError("The selected plugin version changed before it could run.")
            if expected_generation is not None and expected_generation != approval["generation"]:
                raise PluginError("The selected plugin approval changed before it could run.")
            entry = next((item for item in plugin.entries if item["id"] == action), None)
            if entry is None or entry["kind"] not in EXECUTABLE:
                raise PluginError("This plugin action is not executable.")
            context = _context(entry["permission"], context)
            request = {
                "operation": "run",
                "context": context,
                "module": base64.b64encode(assets[entry["module"]]).decode("ascii"),
            }

        @contextmanager
        def authorised():
            with self._guard():
                if self._state()["active"].get(plugin_id) != approval:
                    raise PluginError("The plugin was disabled or changed during invocation.")
                yield

        try:
            result = _result(entry, self._sandbox.execute(request, authorise=authorised), context)
        except SandboxError as error:
            raise PluginError(str(error)) from error
        with authorised():
            return result
