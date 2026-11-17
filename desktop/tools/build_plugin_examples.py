"""Rebuild the bundled examples from reviewed local data and WebAssembly text."""

import hashlib
import json
import zipfile
from pathlib import Path

from wasmtime import wat2wasm

OUTPUT = Path(__file__).resolve().parents[1] / "src/dodo_memory_desktop/assets/plugins"


def wasm(reply):
    payload = json.dumps(reply, ensure_ascii=True, separators=(",", ":")).encode("ascii")
    encoded = "".join(f"\\{byte:02x}" for byte in payload)
    offset = 49_152
    pointer = offset << 32 | len(payload)
    return bytes(
        wat2wasm(f"""
        (module
          (memory (export "memory") 1 1)
          (data (i32.const {offset}) "{encoded}")
          (func (export "run") (param i32 i32) (result i64)
            i64.const {pointer}))
    """)
    )


def pack_versions(
    identity,
    name,
    entries,
    modules=None,
    permissions=(),
    *,
    next_entries=(),
    next_modules=None,
):
    modules = modules or {}
    releases = (
        ("1.0.0", "-1.0.0", entries, modules),
        ("1.1.0", "", [*entries, *next_entries], {**modules, **(next_modules or {})}),
    )
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for version, suffix, release_entries, release_modules in releases:
        manifest = {
            "schema": 1,
            "id": identity,
            "name": name,
            "version": version,
            "permissions": list(permissions),
            "entries": release_entries,
            "files": {
                path: hashlib.sha256(blob).hexdigest() for path, blob in release_modules.items()
            },
        }
        with zipfile.ZipFile(OUTPUT / f"{identity}{suffix}.dodo-pack.zip", "w") as archive:
            payloads = {"manifest.json": json.dumps(manifest, indent=2).encode(), **release_modules}
            for path, data in sorted(payloads.items()):
                entry = zipfile.ZipInfo(path)
                entry.compress_type = zipfile.ZIP_DEFLATED
                archive.writestr(entry, data)


def main():
    pack_versions(
        "research-starter",
        "Research starter",
        [
            {
                "id": "ink-and-orchid",
                "title": "Ink and orchid",
                "kind": "theme",
                "tokens": {
                    "canvas": "#09070C",
                    "rail": "#100D14",
                    "pane": "#18131F",
                    "raised": "#221B2C",
                    "selection": "#382B48",
                    "text": "#F2ECF8",
                    "muted": "#A99AB9",
                    "accent": "#C5A7FF",
                },
            },
            {
                "id": "research-methods",
                "title": "Find research methods",
                "kind": "saved_view",
                "query": "experimental methods",
            },
            {
                "id": "paper-notes",
                "title": "Paper reading notes",
                "kind": "note_template",
                "note_title": "Paper reading notes",
                "text": "## Question\n\n## Method\n\n## Evidence and limits\n\n## Follow-up",
            },
            {
                "id": "evidence-map",
                "title": "Evidence map",
                "kind": "canvas_template",
                "document": {
                    "title": "Evidence map",
                    "nodes": [
                        {
                            "id": "question",
                            "type": "text",
                            "x": 0,
                            "y": 0,
                            "width": 250,
                            "height": 130,
                            "text": "What is the question?",
                        },
                        {
                            "id": "evidence",
                            "type": "text",
                            "x": 330,
                            "y": 0,
                            "width": 250,
                            "height": 130,
                            "text": "What evidence supports the answer?",
                        },
                    ],
                    "edges": [
                        {
                            "id": "supports",
                            "fromNode": "question",
                            "fromSide": "right",
                            "toNode": "evidence",
                            "toSide": "left",
                        },
                    ],
                },
            },
        ],
        next_entries=(
            {
                "id": "study-log",
                "title": "Study log",
                "kind": "note_template",
                "note_title": "Study log",
                "text": "## Aim\n\n## Observations\n\n## Open questions",
            },
        ),
    )
    pack_versions(
        "relationship-tools",
        "Relationship tools",
        [
            {
                "id": "related",
                "title": "Related notes",
                "kind": "panel",
                "module": "related.wasm",
                "permission": "read:related",
            },
            {
                "id": "instructions",
                "title": "Show relationship help",
                "kind": "command",
                "module": "help.wasm",
                "permission": None,
            },
        ],
        {
            "related.wasm": wasm({"kind": "panel", "title": "Related notes", "select": "related"}),
            "help.wasm": wasm(
                {
                    "kind": "message",
                    "text": "Open a note in Library before running the Related notes panel.",
                }
            ),
        },
        ("read:related",),
        next_modules={
            "help.wasm": wasm(
                {
                    "kind": "message",
                    "text": "Open a note in Library, run Related notes, then open a result.",
                }
            ),
        },
    )


if __name__ == "__main__":
    main()
