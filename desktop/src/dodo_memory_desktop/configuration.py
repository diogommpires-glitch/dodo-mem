"""Public builds use explicit local state and source roots."""

import json
import os
import stat
from pathlib import Path


def configured_path(value):
    if not isinstance(value, str) or not value:
        raise ValueError("Choose a nonempty absolute local path.")
    path = Path(value)
    if not path.is_absolute() or str(path).startswith(("\\\\", "//")) or ".." in path.parts:
        raise ValueError("Choose an absolute local path without parent traversal.")
    for part in (*reversed(path.parents), path):
        try:
            info = part.lstat()
        except FileNotFoundError:
            break
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise ValueError("Configured paths cannot contain redirects.")
    return path.resolve()


def runtime_path(component):
    value = os.environ.get("DODO_MEM_RUNTIME_ROOT")
    if not value:
        raise ValueError("Set DODO_MEM_RUNTIME_ROOT to an absolute local state directory.")
    return configured_path(str(configured_path(value) / component))


def source_roots(variable):
    values = json.loads(os.environ.get(variable, "[]"))
    if not isinstance(values, list) or len(values) > 32:
        raise ValueError(f"{variable} must contain a JSON array of up to 32 local roots.")
    return tuple(dict.fromkeys(configured_path(value) for value in values))
