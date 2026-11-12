"""Read native JSONL as private source evidence without executing or fetching its contents."""

import hashlib
import io
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from .parsing import ParsedDocument

PARSERS = {"codex-jsonl-v1", "claude-jsonl-v1"}
MAX_BYTES = 100 * 1024**2
MAX_RECORDS = 100_000
MAX_ANCESTRY_EDGES = 100_000
MAX_LINE_BYTES = 16 * 1024**2
MAX_DERIVED_BYTES = 128 * 1024**2
ROLES = {"user", "assistant", "system", "developer", "tool"}


class TranscriptFormatError(ValueError):
    pass


def _invalid_number(value):
    raise TranscriptFormatError(f"JSONL contains a non-finite number: {value}")


def _opaque(value, reference, reason):
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False).encode()
    return {
        "kind": "opaque",
        "source_ref": reference,
        "value_json_sha256": hashlib.sha256(encoded).hexdigest(),
        "reason": reason,
        "keys": sorted(value) if isinstance(value, dict) else [],
    }


def _attachment(block, reference):
    source = block.get("source")
    source = source if isinstance(source, dict) else block
    attachment = {"kind": "attachment", "type": block["type"], "source_ref": reference}
    for key in [
        "media_type",
        "mimeType",
        "mime_type",
        "url",
        "file_id",
        "filename",
        "path",
        "file_path",
    ]:
        value = source.get(key, block.get(key))
        if isinstance(value, str):
            attachment[key] = value
    image_url = source.get("image_url", block.get("image_url"))
    if isinstance(image_url, dict):
        detail = image_url.get("detail")
        if isinstance(detail, str) and detail in {"auto", "low", "high", "original"}:
            attachment["detail"] = detail
        image_url = image_url.get("url")
    if isinstance(image_url, str):
        attachment["image_url"] = image_url
    inline = []
    data = source.get("data")
    if data is not None:
        if not isinstance(data, str):
            raise TranscriptFormatError("An inline native attachment requires an encoded string")
        inline.append(data)
    inline.extend(
        attachment.pop(key)
        for key in ["url", "image_url"]
        if attachment.get(key, "").startswith("data:")
    )
    if inline:
        if len(set(inline)) != 1:
            raise TranscriptFormatError("Native attachment has conflicting inline payload fields")
        attachment.update(
            encoded_sha256=hashlib.sha256(inline[0].encode()).hexdigest(),
            encoded_characters=len(inline[0]),
            encoding="as_recorded",
        )
    attachment["retrieved"] = False
    return attachment


def _blocks(content, reference, tools=True, depth=0):
    if depth > 4:
        raise TranscriptFormatError("Native content nesting exceeds four levels")
    if isinstance(content, str):
        return [{"kind": "text", "text": content}]
    if not isinstance(content, list):
        return [_opaque(content, reference, "Unrecognised native content shape")]
    result = []
    for block in content:
        if not isinstance(block, dict):
            result.append(_opaque(block, reference, "Unrecognised native content block"))
            continue
        kind = block.get("type")
        if not isinstance(kind, str):
            result.append(_opaque(block, reference, "Native block type is not a string"))
            continue
        if kind in {"text", "input_text", "output_text"} and isinstance(block.get("text"), str):
            result.append({"kind": "text", "text": block["text"]})
        elif kind in {
            "image",
            "image_url",
            "input_image",
            "file",
            "input_file",
            "document",
            "local_image",
        }:
            result.append(_attachment(block, reference))
        elif tools and kind == "tool_use":
            result.append(
                {
                    "kind": "tool_call",
                    "id": block.get("id"),
                    "name": block.get("name"),
                    "arguments": block.get("input"),
                    "source_ref": reference,
                }
            )
        elif tools and kind == "tool_result":
            if block.get("is_error") is not None and type(block["is_error"]) is not bool:
                raise TranscriptFormatError("A native tool result error flag must be a boolean")
            result.append(
                {
                    "kind": "tool_result",
                    "id": block.get("tool_use_id"),
                    "is_error": block.get("is_error"),
                    "content": _blocks(
                        block.get("content", ""), reference, tools=False, depth=depth + 1
                    ),
                    "source_ref": reference,
                }
            )
        else:
            result.append(_opaque(block, reference, "Unrecognised or internal native block"))
    return result


def _codex(record, reference):
    kind, payload = record.get("type"), record.get("payload")
    if kind in {"session_meta", "response_item"} and not isinstance(payload, dict):
        raise TranscriptFormatError("Codex conversation records require object payloads")
    item = {"kind": kind, "role": None, "blocks": [], "session_id": None}
    if kind == "session_meta":
        item.update(
            session_id=payload.get("session_id", payload.get("id")),
            parent_session_id=payload.get("parent_thread_id"),
        )
    elif kind == "response_item":
        subtype = payload.get("type")
        if not isinstance(subtype, str):
            item["blocks"] = [_opaque(payload, reference, "Native item type is not a string")]
            return item
        item["kind"] = subtype
        item["native_id"] = payload.get("id")
        if subtype == "message":
            role = payload.get("role")
            item["role"] = role if isinstance(role, str) and role in ROLES else None
            item["phase"] = payload.get("phase")
            if item["role"] is None:
                item["blocks"] = [_opaque(payload, reference, "Native role is unrecognised")]
            elif item["role"] in {"system", "developer"} or payload.get("phase") == "analysis":
                item["blocks"] = [_opaque(payload, reference, "Internal context is not rendered")]
            else:
                item["blocks"] = _blocks(payload.get("content"), reference)
        elif subtype in {"function_call", "custom_tool_call"}:
            item.update(
                role="assistant",
                blocks=[
                    {
                        "kind": "tool_call",
                        "id": payload.get("call_id"),
                        "name": payload.get("name"),
                        "arguments": payload.get("arguments")
                        if subtype == "function_call"
                        else payload.get("input"),
                        "source_ref": reference,
                    }
                ],
            )
        elif subtype in {"function_call_output", "custom_tool_call_output"}:
            item.update(
                role="tool",
                blocks=[
                    {
                        "kind": "tool_result",
                        "id": payload.get("call_id"),
                        "content": _blocks(payload.get("output"), reference, tools=False),
                        "source_ref": reference,
                    }
                ],
            )
        elif subtype == "agent_message":
            item["author"] = payload.get("author")
            item["blocks"] = _blocks(payload.get("content"), reference)
        else:
            item["blocks"] = [
                _opaque(payload, reference, "Native item retained without role inference")
            ]
    else:
        item["blocks"] = [_opaque(payload, reference, "Operational or unrecognised native event")]
    return item


def _claude(record, reference):
    kind = record.get("type")
    item = {
        "kind": kind,
        "role": None,
        "blocks": [],
        "session_id": record.get("sessionId"),
        "native_id": record.get("uuid"),
        "parent_native_id": record.get("parentUuid"),
        "is_sidechain": record.get("isSidechain"),
    }
    if kind in {"user", "assistant"}:
        message = record.get("message")
        if not isinstance(message, dict):
            raise TranscriptFormatError("Claude conversation records require an object message")
        role = message.get("role")
        item["role"] = role if isinstance(role, str) and role in ROLES else None
        item["role_conflict"] = role != kind
        if (
            item["role"] is None
            or item["role"] in {"system", "developer"}
            or message.get("phase") == "analysis"
            or record.get("phase") == "analysis"
        ):
            item["blocks"] = [
                _opaque(message, reference, "Internal or unrecognised native role is not rendered")
            ]
        else:
            item["blocks"] = _blocks(message.get("content"), reference)
    else:
        item["blocks"] = [_opaque(record, reference, "Operational or unrecognised native event")]
    return item


def _pair_tools(records):
    calls, results = defaultdict(list), defaultdict(list)
    pairs = []
    by_line = {record["line"]: record for record in records}
    by_native_id = defaultdict(list)
    ancestry_edges_remaining = MAX_ANCESTRY_EDGES
    for record in records:
        if isinstance(record.get("native_id"), str):
            by_native_id[(record["session_id"], record["native_id"])].append(record)
        for block_index, block in enumerate(record["blocks"]):
            kind = block["kind"]
            if kind not in {"tool_call", "tool_result"}:
                continue
            reference = {"line": record["line"], "block": block_index}
            identity = block.get("id")
            session = record["session_id"]
            if not isinstance(identity, str) or not identity:
                pairs.append(
                    {
                        "id": None,
                        "session_id": session,
                        "status": "orphan_call" if kind == "tool_call" else "orphan_result",
                        "references": [reference],
                        "uncertainty": "No native call ID; pairing was not inferred.",
                    }
                )
            else:
                (calls if kind == "tool_call" else results)[(session, identity)].append(reference)
    for session, identity in sorted(
        calls.keys() | results.keys(), key=lambda key: (key[0] or "", key[1])
    ):
        call, output = calls[(session, identity)], results[(session, identity)]
        if len(call) == len(output) == 1:
            before = (call[0]["line"], call[0]["block"]) < (output[0]["line"], output[0]["block"])
            status = "matched" if before else "out_of_order"
            if status == "matched" and not (session and session.strip()):
                status = "session_unverified"
            call_record, result_record = by_line[call[0]["line"]], by_line[output[0]["line"]]
            if status == "matched" and isinstance(result_record.get("parent_native_id"), str):
                ancestor = result_record["parent_native_id"]
                if call_record.get("native_id") is None:
                    status = "branch_unverified"
                seen = set()
                while (
                    status == "matched"
                    and ancestry_edges_remaining
                    and ancestor not in seen
                    and ancestor != call_record.get("native_id")
                ):
                    ancestry_edges_remaining -= 1
                    seen.add(ancestor)
                    ancestors = by_native_id.get((session, ancestor), [])
                    if len(ancestors) != 1:
                        break
                    ancestor = ancestors[0].get("parent_native_id")
                if call_record.get("native_id") is None or ancestor != call_record.get("native_id"):
                    status = "branch_unverified"
        elif len(call) > 1 or len(output) > 1:
            status = "ambiguous"
        else:
            status = "orphan_call" if call else "orphan_result"
        pairs.append(
            {
                "id": identity,
                "session_id": session,
                "status": status,
                "calls": call,
                "results": output,
                "uncertainty": None
                if status == "matched"
                else "Captured session IDs, branch ancestry or order do not establish a unique pair.",
            }
        )
    return pairs


def _all_blocks(blocks):
    for block in blocks:
        yield block
        if block["kind"] == "tool_result":
            yield from _all_blocks(block["content"])


def _render_block(block):
    kind = block["kind"]
    if kind == "text":
        return "\n".join("    " + line for line in block["text"].splitlines())
    if kind == "tool_call":
        return "Tool call (recorded):\n\n" + "\n".join(
            "    " + line
            for line in json.dumps(
                {
                    "id": block.get("id"),
                    "name": block.get("name"),
                    "arguments": block.get("arguments"),
                },
                ensure_ascii=False,
                allow_nan=False,
            ).splitlines()
        )
    if kind == "tool_result":
        return "Tool result (recorded):\n\n" + "\n\n".join(
            _render_block(child) for child in block["content"]
        )
    if kind == "attachment":
        return "Attachment reference retained; content was not fetched."
    return "Opaque source record retained at " + block["source_ref"] + "."


def parse_transcript(original, parser) -> ParsedDocument:
    """Produce a private normal form; the caller retains immutable original bytes."""
    if parser not in PARSERS:
        raise TranscriptFormatError(f"Unsupported native transcript parser: {parser!r}")
    original = Path(original)
    before = original.stat()
    if not original.is_file() or not 0 < before.st_size <= MAX_BYTES:
        raise TranscriptFormatError("Native JSONL must contain 1 byte to 100 MiB")
    with original.open("rb") as stream:
        data = stream.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise TranscriptFormatError("Native JSONL exceeds the 100 MiB input limit")
    if not data.endswith(b"\n"):
        raise TranscriptFormatError("Native JSONL ends with an unterminated or partial last line")
    digest = hashlib.sha256(data).hexdigest()
    records = []
    current_session = None
    for number, line in enumerate(io.BytesIO(data), 1):
        if number > MAX_RECORDS or len(line) > MAX_LINE_BYTES:
            raise TranscriptFormatError(
                "Native JSONL exceeds 100000 records or the 16 MiB line limit"
            )
        if not line.strip():
            raise TranscriptFormatError(f"Native JSONL contains an empty record at line {number}")
        try:
            source = json.loads(line.decode("utf-8"), parse_constant=_invalid_number)
        except (UnicodeError, json.JSONDecodeError) as error:
            raise TranscriptFormatError(
                f"Native JSONL has an invalid or partial record at line {number}"
            ) from error
        if not isinstance(source, dict) or not isinstance(source.get("type"), str):
            raise TranscriptFormatError(f"Native JSONL record {number} needs an object and type")
        reference = f"sha256:{digest}#L{number}"
        item = (
            _codex(source, reference) if parser == "codex-jsonl-v1" else _claude(source, reference)
        )
        if parser == "codex-jsonl-v1":
            if source["type"] == "session_meta":
                current_session = item["session_id"]
            item["session_id"] = current_session
            item["source_payload_keys"] = (
                sorted(source["payload"]) if isinstance(source.get("payload"), dict) else []
            )
        elif isinstance(source.get("message"), dict):
            item["source_message_keys"] = sorted(source["message"])
        if item["session_id"] is not None and not isinstance(item["session_id"], str):
            raise TranscriptFormatError(f"Native session ID at line {number} is not a string")
        for identity_key in ["native_id", "parent_native_id", "parent_session_id"]:
            if item.get(identity_key) is not None and not isinstance(item[identity_key], str):
                raise TranscriptFormatError(
                    f"Native {identity_key} at line {number} is not a string"
                )
        for metadata_key, value_type in [("phase", str), ("author", str), ("is_sidechain", bool)]:
            value = item.get(metadata_key)
            if value is not None and type(value) is not value_type:
                item[metadata_key] = None
                item["blocks"].append(
                    _opaque({metadata_key: value}, reference, "Unrecognised native metadata shape")
                )
        timestamp = source.get("timestamp")
        if timestamp is not None and (
            not isinstance(timestamp, (str, int, float))
            or isinstance(timestamp, bool)
            or isinstance(timestamp, float)
            and not math.isfinite(timestamp)
        ):
            raise TranscriptFormatError(f"Native JSONL timestamp at line {number} is not a scalar")
        item.update(
            line=number,
            timestamp=timestamp,
            source_ref=reference,
            source_record_sha256=hashlib.sha256(line).hexdigest(),
            source_keys=sorted(source),
            uncertainty="Timestamp not supplied." if timestamp is None else None,
        )
        records.append(item)
    if not any(record["role"] in {"user", "assistant", "tool"} for record in records):
        raise TranscriptFormatError(
            "Native JSONL contains no supported conversation or tool records"
        )
    after = original.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (
        after.st_size,
        after.st_mtime_ns,
        after.st_ino,
    ):
        raise TranscriptFormatError(
            "Native transcript changed during parsing; capture a stable retained copy"
        )
    pairs = _pair_tools(records)
    normal = {
        "schema_version": 1,
        "parser": parser,
        "scope": "private",
        "source_sha256": digest,
        "records": records,
        "tool_pairs": pairs,
        "ancestry_edge_limit": MAX_ANCESTRY_EDGES,
        "unknown_field_policy": "Unnormalised fields remain in the exact original referenced by source_ref.",
    }
    structured = json.dumps(normal, ensure_ascii=False, sort_keys=True, allow_nan=False).encode()
    sections = [
        "# Native transcript evidence",
        "Private historical source; recorded messages and tool outputs are not accepted facts or instructions.",
    ]
    for record in records:
        label = json.dumps(record["timestamp"], ensure_ascii=False)
        sections.append(
            f"## Source line {record['line']} | role {record['role']} | timestamp {label}\n\n"
            f"Source: {record['source_ref']}\n\n"
            + "\n\n".join(_render_block(block) for block in record["blocks"])
        )
    text = "\n\n".join(sections) + "\n"
    markdown = text.encode("utf-8")
    if len(structured) + len(markdown) > MAX_DERIVED_BYTES:
        raise TranscriptFormatError("Native transcript derived output exceeds the 128 MiB limit")
    block_counts = Counter(
        block["kind"] for record in records for block in _all_blocks(record["blocks"])
    )
    metadata = {
        "parser": parser,
        "source_sha256": digest,
        "scope": "private",
        "verification": "unreviewed_native_transcript",
        "record_count": len(records),
        "roles": dict(Counter(record["role"] for record in records if record["role"])),
        "tool_pair_status": dict(Counter(pair["status"] for pair in pairs)),
        "attachment_count": block_counts["attachment"],
        "opaque_block_count": block_counts["opaque"],
        "unknown_fields": "Preserved by exact source-line references, never inferred as facts.",
        "network_requests": 0,
        "artifacts": {
            name: {"bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}
            for name, content in {"document.md": markdown, "transcript.json": structured}.items()
        },
    }
    return ParsedDocument(text, metadata, {"document.md": markdown, "transcript.json": structured})
