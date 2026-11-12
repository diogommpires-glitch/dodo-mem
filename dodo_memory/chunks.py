"""Markdown sections, paragraphs, fences and complete table rows for retrieval."""

import re

HEADING = re.compile(r"^#{1,6}\s+(.+)$")
SEPARATOR = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)*\|?\s*$")
FENCE = re.compile(r"^\s*(" + chr(96) + r"{3,}|~{3,})")
CLOSING = re.compile(r"^\s*([" + chr(96) + r"~]+)\s*$")


def _blocks(text):
    lines = text.splitlines()
    position = 0
    while position < len(lines):
        line = lines[position]
        if not line.strip():
            position += 1
            continue
        heading = HEADING.match(line)
        if heading:
            yield "heading", heading[1]
            position += 1
            continue
        start = position
        fence = FENCE.match(line)
        if fence:
            position += 1
            while position < len(lines):
                closing = CLOSING.match(lines[position])
                position += 1
                if (
                    closing
                    and set(closing[1]) == {fence[1][0]}
                    and len(closing[1]) >= len(fence[1])
                ):
                    break
            yield "fence", "\n".join(lines[start:position])
            continue
        if position + 1 < len(lines) and "|" in line and SEPARATOR.match(lines[position + 1]):
            position += 2
            while position < len(lines) and "|" in lines[position] and lines[position].strip():
                position += 1
            yield "table", "\n".join(lines[start:position])
            continue
        position += 1
        while (
            position < len(lines) and lines[position].strip() and not HEADING.match(lines[position])
        ):
            if FENCE.match(lines[position]):
                break
            if (
                position + 1 < len(lines)
                and "|" in lines[position]
                and SEPARATOR.match(lines[position + 1])
            ):
                break
            position += 1
        yield "paragraph", "\n".join(lines[start:position])


def _table_parts(text, maximum):
    lines = text.splitlines()
    header = "\n".join(lines[:2])
    if len(header) > maximum // 2:
        raise ValueError(
            "Table header exceeds the bounded retrieval representation; original remains authoritative"
        )
    batches, rows, size = [], [], len(header) + 2
    for row in lines[2:]:
        if len(header) + len(row) + 256 > maximum:
            raise ValueError(
                "Single table row exceeds bounded retrieval representation; never split cells silently"
            )
        if rows and size + len(row) + 1 > maximum - 256:
            batches.append(rows)
            rows, size = [], len(header) + 2
        rows.append(row)
        size += len(row) + 1
    if rows or not batches:
        batches.append(rows)
    for index, batch in enumerate(batches, 1):
        yield (
            f"Table segment {index}/{len(batches)}. Complete rows with repeated original header; "
            "the complete table remains in the canonical source.\n\n"
            + header
            + "\n"
            + "\n".join(batch)
        )


def _fence_parts(text, target, maximum):
    lines = text.splitlines()
    opening = FENCE.match(lines[0])[1]
    closing = CLOSING.match(lines[-1]) if len(lines) > 1 else None
    closed = bool(closing and set(closing[1]) == {opening[0]} and len(closing[1]) >= len(opening))
    ending = lines[-1] if closed else opening
    body = lines[1:-1] if closed else lines[1:]
    overhead = len(lines[0]) + len(ending) + 256
    batches, current, size = [], [], overhead
    for line in body:
        if len(line) + overhead > maximum:
            raise ValueError(
                "Single code line exceeds bounded representation; never cut original lines silently"
            )
        if current and size + len(line) + 1 > target:
            batches.append(current)
            current, size = [], overhead
        current.append(line)
        size += len(line) + 1
    if current or not batches:
        batches.append(current)
    for index, batch in enumerate(batches, 1):
        uncertainty = " The original fence was unterminated." if not closed else ""
        yield (
            f"Code segment {index}/{len(batches)}. Complete original lines; canonical source retained."
            + uncertainty
            + "\n\n"
            + lines[0]
            + "\n"
            + "\n".join(batch)
            + "\n"
            + ending
        )


def section_chunks(text, title, *, target=3000, maximum=48000):
    heading, pending, caption = title, "", None
    for kind, block in _blocks(text):
        if kind == "heading":
            if pending:
                yield heading, pending
                pending = ""
            heading, caption = block, None
            continue
        if kind == "table":
            if pending:
                yield heading, pending
                pending = ""
            prefix = (caption + "\n\n") if caption else ""
            if len(prefix) + len(block) <= target:
                yield heading, prefix + block
            else:
                for part in _table_parts(block, maximum - len(prefix)):
                    yield heading, prefix + part
            caption = None
            continue
        caption = block if kind == "paragraph" and re.match(r"(?i)^table\s+\S", block) else None
        if len(block) > maximum:
            if kind == "fence":
                pieces = list(_fence_parts(block, target, maximum))
            else:
                # Only prose is split by characters; table cells and source-code lines stay intact.
                pieces = [
                    block[offset : offset + target] for offset in range(0, len(block), target)
                ]
        else:
            pieces = [block]
        for piece in pieces:
            candidate = pending + ("\n\n" if pending else "") + piece
            if pending and len(candidate) > target:
                yield heading, pending
                pending = piece
            else:
                pending = candidate
    if pending:
        yield heading, pending
