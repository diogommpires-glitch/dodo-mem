import hashlib
import json
import os
import subprocess

import pytest
from pypdf import PdfWriter
from pypdf.generic import (
    DecodedStreamObject,
    DictionaryObject,
    FloatObject,
    NameObject,
    NumberObject,
)

from dodo_memory.evidence import source_location
from dodo_memory.ingest import Ingestor
from dodo_memory.repositories import capture_repository
from dodo_memory.source_registry import SourceRegistry
from dodo_memory.store import Vault


@pytest.fixture
def intake(tmp_path):
    home = tmp_path / "memory"
    (home / "vault").mkdir(parents=True)
    root = tmp_path / "intake"
    root.mkdir()
    vault = Vault(home)
    return vault, SourceRegistry(vault, allowed_roots=[root]), root


def indexed(intake, name, content, *, parser="text-v1", kind="note", scope="shared"):
    vault, registry, root = intake
    source = root / name
    source.write_bytes(content)
    job = registry.submit(source, parser=parser, kind=kind, scope=scope)
    assert Ingestor(vault, registry).run_next(job_id=job["id"])
    result = registry.job(job["id"], scopes=(scope,))
    assert result["status"] == "indexed", result["error"]
    return result


def native(intake):
    rows = [
        {"type": "session_meta", "payload": {"id": "synthetic-session"}},
        {
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "synthetic question"}],
            },
        },
        {"type": "event_msg", "payload": {"type": "task_complete"}},
        {
            "type": "response_item",
            "payload": {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "synthetic answer"}],
            },
        },
    ]
    content = "".join(json.dumps(row) + "\n" for row in rows).encode()
    return indexed(
        intake,
        "synthetic.jsonl",
        content,
        parser="codex-jsonl-v1",
        kind="conversation",
        scope="private",
    )


def export(intake):
    def message(identity, role):
        return {
            "id": identity,
            "author": {"role": role},
            "content": {"parts": ["synthetic only"]},
            "create_time": None,
        }

    rows = [
        {
            "id": "conversation-one",
            "title": "Synthetic",
            "current_node": "answer-two",
            "mapping": {
                "root": {"parent": None, "message": None},
                "question": {"parent": "root", "message": message("user-one", "user")},
                "answer-one": {
                    "parent": "question",
                    "message": message("assistant-one", "assistant"),
                },
                "answer-two": {
                    "parent": "question",
                    "message": message("assistant-two", "assistant"),
                },
            },
        }
    ]
    return indexed(
        intake,
        "export.json",
        json.dumps(rows).encode(),
        parser="conversation-json-v1",
        kind="conversation",
        scope="private",
    )


def test_private_scope_is_checked_before_any_vault_metadata_or_io():
    class NoIO:
        def get(self, *args, **kwargs):
            raise AssertionError("Unauthorised lookup reached vault IO")

    with pytest.raises(PermissionError):
        source_location(NoIO(), "private/unknown.md")


@pytest.mark.parametrize(
    "path", ["../private/a.md", "C:/private/a.md", "/private/a.md", "documents/../private/a.md"]
)
def test_caller_cannot_supply_an_arbitrary_file_path(path):
    with pytest.raises(ValueError, match="relative"):
        source_location(object(), path)


def test_exact_managed_text_lines_context_and_provenance(intake):
    vault, _, _ = intake
    job = indexed(intake, "source.txt", b"first\r\nsecond\r\nthird\r\nfourth\r\n")
    result = source_location(vault, job["path"], lines=[2, 3], context=1)
    assert result["original"] == str(vault.home / job["original"])
    assert (result["source_id"], result["source_version"], result["parser"]) == (
        job["source_id"],
        job["sha256"],
        "text-v1",
    )
    assert result["locator"] == {
        "kind": "lines",
        "start": 2,
        "end": 3,
        "context_start": 1,
        "context_end": 4,
        "total": 4,
    }
    assert "second" not in json.dumps(result)


def test_desktop_preview_uses_exact_verified_line_range(intake):
    vault, _, _ = intake
    job = indexed(intake, "preview.txt", b"first\r\nsecond\r\nthird\r\nfourth\r\n")
    result = source_location(vault, job["path"], lines=[2, 3], context=1, preview=True)
    assert result["preview_text"] == "second\r\nthird"
    assert result["preview_truncated"] is False
    (vault.home / job["original"]).write_bytes(b"first\r\nsecond\r\nthird\r\nchanged")
    with pytest.raises(ValueError, match="recorded source hash"):
        source_location(vault, job["path"], lines=[2, 3], preview=True)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"lines": [0, 1]},
        {"lines": [2, 1]},
        {"lines": [True, 2]},
        {"lines": [1, 4]},
        {"lines": [1, 1], "context": -1},
        {"lines": [1, 1], "context": 51},
        {"lines": [1, 1], "messages": [1, 1]},
        {"context": 1},
        {"conversation": "unbound"},
    ],
)
def test_invalid_ambiguous_and_out_of_range_locators_reject(intake, kwargs):
    job = indexed(intake, "source.txt", b"one\ntwo\nthree\n")
    with pytest.raises(ValueError):
        source_location(intake[0], job["path"], **kwargs)


def test_selected_context_span_has_a_finite_bound(intake):
    job = indexed(intake, "many.txt", b"line\n" * 201)
    with pytest.raises(ValueError, match="200"):
        source_location(intake[0], job["path"], lines=[1, 201])


def test_native_message_range_resolves_original_lines_and_stays_private(intake):
    vault, _, _ = intake
    job = native(intake)
    with pytest.raises(PermissionError):
        source_location(vault, job["path"], messages=[1, 2])
    result = source_location(vault, job["path"], ("private",), messages=[2, 2], context=1)
    locator = result["locator"]
    assert (
        locator["kind"],
        locator["total"],
        locator["context_start"],
        locator["context_end"],
    ) == ("messages", 2, 1, 2)
    assert [item["line"] for item in locator["items"]] == [2, 4]
    assert [item["source_ref"] for item in locator["items"]] == [
        f"sha256:{job['sha256']}#L2",
        f"sha256:{job['sha256']}#L4",
    ]
    assert "synthetic answer" not in json.dumps(result)
    with pytest.raises(ValueError):
        source_location(vault, job["path"], ("private",), messages=[3, 3])


def test_export_range_requires_real_branch_and_preserves_message_ids(intake):
    vault, _, _ = intake
    job = export(intake)
    kwargs = {
        "messages": [2, 2],
        "conversation": "conversation-one",
        "branch": "answer-two",
        "context": 1,
    }
    result = source_location(vault, job["path"], ("private",), **kwargs)
    assert [item["id"] for item in result["locator"]["items"]] == ["user-one", "assistant-two"]
    assert [item["json_pointer"] for item in result["locator"]["items"]] == [
        "/0/mapping/question/message",
        "/0/mapping/answer-two/message",
    ]
    with pytest.raises(ValueError, match="branch"):
        source_location(
            vault, job["path"], ("private",), messages=[1, 1], conversation="conversation-one"
        )
    with pytest.raises(ValueError, match="branch"):
        source_location(vault, job["path"], ("private",), **{**kwargs, "branch": "missing"})


def test_changed_original_and_registered_message_artifact_reject(intake):
    vault, registry, _ = intake
    job = native(intake)
    artifact = (
        vault.home
        / "derived"
        / job["source_id"]
        / job["sha256"]
        / job["parser"]
        / "transcript.json"
    )
    artifact.write_bytes(artifact.read_bytes() + b" ")
    with pytest.raises(ValueError, match="hash"):
        source_location(vault, job["path"], ("private",), messages=[1, 1])
    original = vault.home / job["original"]
    original.write_bytes(original.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="hash"):
        source_location(vault, job["path"], ("private",))
    assert registry.job(job["id"], ("private",))["status"] == "indexed"


def test_managed_original_binding_cannot_be_replaced_with_arbitrary_path(intake):
    vault, _, root = intake
    job = indexed(intake, "source.txt", b"synthetic\n")
    with vault.connect() as db:
        db.execute(
            "UPDATE source_versions SET original=? WHERE source_id=?",
            (str(root / "source.txt"), job["source_id"]),
        )
    with pytest.raises(ValueError, match="original"):
        source_location(vault, job["path"])


def test_qualified_code_opens_captured_lines_after_live_checkout_changes(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    env = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}

    def git(*args):
        return (
            subprocess.run(
                ["git", "-C", str(root), *args], check=True, capture_output=True, env=env
            )
            .stdout.decode()
            .strip()
        )

    git("init", "-q")
    git("config", "user.name", "Synthetic fixture")
    git(
        "config",
        "user.email",
        "239468521+diogommpires-glitch@users.noreply.github.com",
    )
    source = root / "calculation.py"
    source.write_bytes(b"def add(value):\n    return value + 1\n")
    git("add", ".")
    git("commit", "-qm", "synthetic")
    head = git("rev-parse", "HEAD")
    vault = Vault(tmp_path / "memory")
    registry = SourceRegistry(vault, repository_roots=[root])
    job = capture_repository(registry, root)["jobs"][0]
    assert Ingestor(vault, registry).run_next(job_id=job["id"])
    job = registry.job(job["id"], ("private",))
    source.write_bytes(b"changed live checkout\n")
    result = source_location(vault, job["path"], ("private",), lines=[2, 2])
    assert result["repository"]["head"] == head
    assert result["repository"]["relative_path"] == "calculation.py"
    assert result["repository"]["representation"] == "working-tree-bytes"
    assert result["locator"]["total"] == 2
    assert (
        result["sha256"] == hashlib.sha256(b"def add(value):\n    return value + 1\n").hexdigest()
    )


def pdf_source(intake):
    vault, _, root = intake
    writer = PdfWriter()
    page = writer.add_blank_page(100, 200)
    page[NameObject("/UserUnit")] = FloatObject(2.5)
    page[NameObject("/Rotate")] = NumberObject(90)
    source = root / "source.pdf"
    writer.write(source)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    original = vault.home / "originals" / (digest + ".pdf")
    original.parent.mkdir()
    original.write_bytes(source.read_bytes())
    note = vault.root / "paper.md"
    note.write_text(f"---\nsource: sha256:{digest}\n---\nSynthetic PDF\n", encoding="utf-8")
    return vault, "paper.md"


def test_pdf_region_uses_actual_page_units_box_and_rotation(intake):
    vault, path = pdf_source(intake)
    result = source_location(vault, path, page=1, bbox=[10, 20, 90, 180])
    assert result["region"] == {
        "rect": [10.0, 20.0, 90.0, 180.0],
        "coordinate_space": "pdf_user_space_bottom_left",
        "unit": "pdf_user_unit",
        "user_unit_scale_points": 2.5,
        "page_box": [0.0, 0.0, 100.0, 200.0],
        "rotation": 90,
    }
    assert "crop" not in result


def test_desktop_pdf_preview_extracts_only_the_selected_verified_page(intake):
    vault, _, root = intake
    writer = PdfWriter()
    for label in ("First synthetic page", "Second synthetic page"):
        page = writer.add_blank_page(100, 200)
        font = DictionaryObject(
            {
                NameObject("/Type"): NameObject("/Font"),
                NameObject("/Subtype"): NameObject("/Type1"),
                NameObject("/BaseFont"): NameObject("/Helvetica"),
            }
        )
        page[NameObject("/Resources")] = DictionaryObject(
            {NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})}
        )
        stream = DecodedStreamObject()
        stream.set_data(f"BT /F1 12 Tf 10 20 Td ({label}) Tj ET".encode("ascii"))
        page[NameObject("/Contents")] = stream
    source = root / "preview.pdf"
    writer.write(source)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    original = vault.home / "originals" / (digest + ".pdf")
    original.parent.mkdir()
    original.write_bytes(source.read_bytes())
    (vault.root / "paper.md").write_text(
        f"---\nsource: sha256:{digest}\n---\nSynthetic PDF\n", encoding="utf-8"
    )
    result = source_location(vault, "paper.md", page=2, preview=True)
    assert result["page"] == result["pages"] == 2
    assert result["preview_text"] == "Second synthetic page"


@pytest.mark.parametrize(
    "bbox",
    [
        [0, 0, 101, 200],
        [20, 20, 10, 30],
        [0, 0, float("nan"), 10],
        [0, 0, float("inf"), 10],
        [False, 0, 10, 20],
    ],
)
def test_pdf_rejects_outside_nonfinite_or_ambiguous_regions(intake, bbox):
    vault, path = pdf_source(intake)
    with pytest.raises(ValueError):
        source_location(vault, path, page=1, bbox=bbox)
    with pytest.raises(ValueError, match="page"):
        source_location(vault, path, bbox=[0, 0, 10, 10])


@pytest.mark.parametrize(
    "path",
    ["PRIVATE/unknown.md", "Private/unknown.md", "private./unknown.md", "private /unknown.md"],
)
def test_windows_scope_aliases_cannot_bypass_pre_io_denial(path):
    class NoIO:
        def get(self, *args, **kwargs):
            raise AssertionError("Unauthorised lookup reached vault IO")

    with pytest.raises((PermissionError, ValueError)):
        source_location(NoIO(), path)


def test_claude_native_message_ids_and_export_object_pointers(intake):
    vault, _, _ = intake
    rows = [
        {
            "type": "user",
            "uuid": "u",
            "sessionId": "s",
            "message": {"role": "user", "content": "synthetic question"},
        },
        {
            "type": "assistant",
            "uuid": "a",
            "parentUuid": "u",
            "sessionId": "s",
            "message": {"role": "assistant", "content": "synthetic response"},
        },
    ]
    native_job = indexed(
        intake,
        "claude.jsonl",
        "".join(json.dumps(row) + "\n" for row in rows).encode(),
        parser="claude-jsonl-v1",
        kind="conversation",
        scope="private",
    )
    native_location = source_location(vault, native_job["path"], ("private",), messages=[1, 2])
    assert [item["id"] for item in native_location["locator"]["items"]] == ["u", "a"]
    export_data = {
        "uuid": "c",
        "name": "Synthetic",
        "chat_messages": [
            {"uuid": "u", "sender": "human", "text": "synthetic question"},
            {"uuid": "a", "sender": "assistant", "text": "synthetic response"},
        ],
    }
    job = indexed(
        intake,
        "claude-export.json",
        json.dumps(export_data).encode(),
        parser="conversation-json-v1",
        kind="conversation",
        scope="private",
    )
    location = source_location(
        vault, job["path"], ("private",), messages=[1, 2], conversation="c", branch="a"
    )
    assert [item["json_pointer"] for item in location["locator"]["items"]] == [
        "/chat_messages/0",
        "/chat_messages/1",
    ]


def test_missing_message_inventory_does_not_guess_locators(intake):
    vault, _, _ = intake
    job = native(intake)
    with vault.connect() as db:
        db.execute("DELETE FROM source_artifacts WHERE job_id=?", (job["id"],))
    assert source_location(vault, job["path"], ("private",))["sha256"] == job["sha256"]
    with pytest.raises(ValueError, match="registered parser artifact"):
        source_location(vault, job["path"], ("private",), messages=[1, 1])


def test_redirected_original_cannot_escape_retention_directory(intake):
    vault, _, root = intake
    job = indexed(intake, "source.txt", b"synthetic\n")
    original = vault.home / job["original"]
    original.unlink()
    try:
        original.symlink_to(root / "source.txt")
    except OSError as error:
        pytest.skip(f"OS does not permit fixture symlink creation: {error}")
    with pytest.raises(ValueError, match="links|redirections"):
        source_location(vault, job["path"])


def test_oversized_retained_original_rejects_before_reading(intake):
    from dodo_memory.evidence import MAX_SOURCE_BYTES

    vault, _, _ = intake
    job = indexed(intake, "source.txt", b"synthetic\n")
    with (vault.home / job["original"]).open("r+b") as stream:
        stream.truncate(MAX_SOURCE_BYTES + 1)
    with pytest.raises(ValueError, match="size limit"):
        source_location(vault, job["path"])


def test_pdf_invalid_user_unit_is_not_silently_claimed_as_points(intake):
    vault, _, root = intake
    writer = PdfWriter()
    page = writer.add_blank_page(100, 200)
    page[NameObject("/UserUnit")] = FloatObject(0)
    source = root / "invalid-units.pdf"
    writer.write(source)
    content = source.read_bytes()
    digest = hashlib.sha256(content).hexdigest()
    original = vault.home / "originals" / (digest + ".pdf")
    original.parent.mkdir()
    original.write_bytes(content)
    (vault.root / "invalid.md").write_text(
        f"---\nsource: sha256:{digest}\n---\nSynthetic\n", encoding="utf-8"
    )
    assert source_location(vault, "invalid.md", page=1)["page"] == 1
    with pytest.raises(ValueError, match="coordinate units"):
        source_location(vault, "invalid.md", page=1, bbox=[0, 0, 10, 10])


def test_export_json_pointer_preserves_bom_array_root_and_escaped_node(intake):
    vault, _, _ = intake
    original = [
        {
            "id": "c",
            "mapping": {
                "node/~one": {
                    "parent": None,
                    "message": {
                        "id": "m",
                        "author": {"role": "user"},
                        "content": {"parts": ["synthetic"]},
                    },
                }
            },
        }
    ]
    job = indexed(
        intake,
        "bom.json",
        b"\xef\xbb\xbf" + json.dumps(original).encode(),
        parser="conversation-json-v1",
        kind="conversation",
        scope="private",
    )
    location = source_location(
        vault, job["path"], ("private",), messages=[1, 1], conversation="c", branch="node/~one"
    )
    assert location["locator"]["items"][0]["json_pointer"] == "/0/mapping/node~1~0one/message"
