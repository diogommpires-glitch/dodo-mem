"""Bounded, local text extraction for the whole-library index."""

import importlib.util
import io
import json
import os
import re
import subprocess
import sys
import sysconfig
import zipfile
from contextlib import nullcontext
from pathlib import Path
from xml.etree import ElementTree

from .whole_index import SECRET_BYTES, history_source

CODE = frozenset(
    [
        ".py",
        ".pyi",
        ".js",
        ".ts",
        ".tsx",
        ".jsx",
        ".css",
        ".html",
        ".htm",
        ".c",
        ".cpp",
        ".h",
        ".hpp",
        ".cs",
        ".rs",
        ".go",
        ".java",
        ".kt",
        ".sh",
        ".ps1",
        ".bat",
        ".cmd",
        ".sql",
        ".lua",
        ".rb",
        ".php",
        ".swift",
        ".r",
        ".m",
        ".dart",
        ".vue",
        ".svelte",
        ".asm",
        ".s",
        ".zig",
    ]
)
NOTES = frozenset({".md", ".markdown", ".org", ".rst"})
DATA = frozenset({".csv", ".tsv", ".json", ".jsonl", ".yaml", ".yml", ".toml", ".xml", ".ini"})
DOCUMENTS = frozenset({".txt", ".log", ".pdf", ".docx", ".pptx", ".xlsx", ".odt", ".ods", ".odp"})
IMAGES = frozenset({".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg", ".tiff", ".bmp", ".heic"})
MEDIA = frozenset({".mp3", ".wav", ".flac", ".ogg", ".m4a", ".mp4", ".mkv", ".mov", ".webm"})
OFFICE = frozenset({".docx", ".pptx", ".xlsx", ".odt", ".ods", ".odp"})
TEXT = CODE | NOTES | DATA | {".txt", ".log", ".svg"}
SUPPORTED = TEXT | OFFICE | {".pdf"}
MAX_EXTRACTED = 150_000
MAX_DOCUMENT_PAGES = 100
INLINE_LINK = re.compile(r"\[[^\]\r\n]{0,200}\]\(([^)\r\n]{1,500})\)|\[\[([^]\r\n]{1,500})\]\]")
HEADING = re.compile(r"^\s{0,3}#{1,5}\s+([\w][\w \-]{5,70})\s*$")
TITLE = re.compile(r"^ {0,3}#[ \t]+(.+?)\s*$")
WIKI_DESTINATION = "\0wiki:"


class EncryptedDocument(ValueError):
    pass


def category(path: Path) -> str:
    ext = path.suffix.casefold()
    if ext == ".dodo-addon" or path.name.casefold().endswith(".dodo-pack.zip"):
        return "add-in"
    if history_source(path) in {"codex", "claude"} or account_bundle(path):
        return "chat"
    if ext in NOTES | {".txt", ".json", ".jsonl"} and any(
        part.casefold() in {"chat", "chats", "conversations"} for part in path.parts[:-1]
    ):
        return "chat"
    for name, extensions in (
        ("code", CODE),
        ("note", NOTES),
        ("data", DATA),
        ("document", DOCUMENTS),
        ("image", IMAGES),
        ("media", MEDIA),
    ):
        if ext in extensions:
            return name
    return "other"


def text_content(data: bytes) -> str:
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        return data.decode("utf-16")
    if b"\x00" in data:
        raise ValueError("Binary data in text file")
    return data.decode("utf-8-sig")


def lines(text: str):
    content = text[:MAX_EXTRACTED].splitlines()
    for start in range(0, len(content), 24):
        block = content[start : start + 24]
        value = "\n".join(block)[:8000]
        if value.strip():
            yield ("lines", start + 1, start + len(block), value)


def office_entries(data: bytes, suffix: str):
    pattern = {
        ".docx": r"word/document\.xml",
        ".pptx": r"ppt/slides/slide\d+\.xml",
        ".xlsx": r"xl/(?:sharedStrings|worksheets/sheet\d+)\.xml",
        ".odt": r"content\.xml",
        ".ods": r"content\.xml",
        ".odp": r"content\.xml",
    }[suffix]
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        if len(archive.infolist()) > 3000:
            raise ValueError("Document has too many parts")
        entries = sorted(
            (item for item in archive.infolist() if re.fullmatch(pattern, item.filename)),
            key=lambda item: (
                item.filename.rsplit("/", 1)[0],
                int(re.search(r"(\d+)\.xml$", item.filename)[1])
                if re.search(r"(\d+)\.xml$", item.filename)
                else 0,
            ),
        )
        total_bytes = 0
        total_text = 0
        for number, item in enumerate(entries[:MAX_DOCUMENT_PAGES], start=1):
            total_bytes += item.file_size
            if item.flag_bits & 1:
                raise EncryptedDocument
            if total_bytes > 8 * 1024 * 1024:
                raise ValueError("Document size limit exceeded")
            raw = archive.read(item)
            if b"<!DOCTYPE" in raw.upper() or b"<!ENTITY" in raw.upper():
                raise ValueError("Document XML contains prohibited declarations")
            tree = ElementTree.fromstring(raw)
            if suffix == ".xlsx":
                local_name = {"t", "v"}
            elif suffix == ".pptx" or suffix == ".docx":
                local_name = {"t"}
            else:
                local_name = {"p", "h"}
            value = " ".join(
                "".join(element.itertext())[:8000]
                for element in tree.iter()
                if element.tag.rsplit("}", 1)[-1] in local_name
            )[: min(16000, MAX_EXTRACTED - total_text)].strip()
            if value:
                total_text += len(value)
                kind = (
                    "slide" if suffix == ".pptx" else "sheet" if suffix == ".xlsx" else "document"
                )
                yield kind, number, number, value
            if total_text >= MAX_EXTRACTED:
                break


def pdf_entries(data: bytes):
    if importlib.util.find_spec("pypdf") is None:
        raise ValueError("PDF parser is unavailable")
    # Invoke the real interpreter so the Windows Job covers the parser, including virtualenv use.
    roots = (str(Path(__file__).absolute().parents[1]), sysconfig.get_path("purelib"))
    script = (
        f"import sys; sys.path[:0] = {roots!r}; "
        "from dodo_memory_desktop.whole_index_extract import pdf_process; pdf_process()"
    )
    with subprocess.Popen(
        [sys._base_executable, "-I", "-c", script],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    ) as parser:
        job = nullcontext()
        if os.name == "nt":
            from .whole_index_windows import limit_parser_process

            job = limit_parser_process(int(parser._handle))
        try:
            with job:
                stdout, _ = parser.communicate(data, timeout=12)
        finally:
            if parser.poll() is None:
                parser.kill()
        if parser.returncode:
            raise subprocess.CalledProcessError(parser.returncode, parser.args)
    output = json.loads(stdout)
    if output == "encrypted":
        raise EncryptedDocument
    return output


def extract(path: Path, data: bytes):
    suffix = path.suffix.casefold()
    if (source := history_source(path)) in {"codex", "claude"}:
        return list(chat_entries(data, source)), "", ""
    if account_bundle(path):
        return list(account_entries(data)), "", ""
    if suffix == ".pdf":
        passages, text = list(pdf_entries(data)), ""
    elif suffix in OFFICE:
        passages, text = list(office_entries(data, suffix)), ""
    else:
        text = text_content(data)
        passages = list(lines(text))
    decoded = text or "\n".join(passage[3] for passage in passages)
    if SECRET_BYTES.search(decoded.encode("utf-8", errors="replace")):
        raise PermissionError("Extracted content matches a credential pattern")
    return passages, text[:MAX_EXTRACTED], ""


def account_bundle(path: Path) -> bool:
    return path.suffix.casefold() == ".zip" and path.name.casefold().startswith(
        (
            "claude-memory-",
            "claude-data-export",
            "chatgpt-export",
            "chatgpt-data-export",
        )
    )


def chat_text(message: dict) -> str:
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    return "\n".join(
        part["text"]
        for part in content
        if isinstance(part, dict)
        and part.get("type") in {"text", "input_text", "output_text"}
        and isinstance(part.get("text"), str)
    )


def chat_entries(data: bytes, source: str):
    remaining = MAX_EXTRACTED
    for number, line in enumerate(data.splitlines(), start=1):
        if len(line) > 512 * 1024:
            continue
        try:
            event = json.loads(line)
        except (UnicodeError, ValueError):
            continue
        if not isinstance(event, dict):
            continue
        if source == "codex":
            message = event.get("payload")
            if event.get("type") != "response_item" or not isinstance(message, dict):
                continue
            if message.get("type") != "message" or message.get("role") not in {"user", "assistant"}:
                continue
        else:
            message = event.get("message")
            if event.get("type") not in {"user", "assistant"} or not isinstance(message, dict):
                continue
        text = chat_text(message)[: min(8000, remaining)].strip()
        if text and not SECRET_BYTES.search(text.encode("utf-8", errors="replace")):
            yield "chat", number, number, text
            remaining -= len(text)
        if remaining <= 0:
            break


def account_entries(data: bytes):
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        if len(archive.infolist()) > 3000:
            raise ValueError("Too many export parts")
        remaining = MAX_EXTRACTED
        for number, entry in enumerate(archive.infolist(), start=1):
            if entry.flag_bits & 1 or entry.file_size > 8 * 1024 * 1024:
                continue
            if entry.filename.casefold().endswith(".md"):
                text = archive.read(entry).decode("utf-8-sig")[: min(8000, remaining)]
                label = Path(entry.filename).name
                chunks = [(label, text)]
            elif Path(entry.filename).name.casefold() == "conversations.json":
                records = json.loads(archive.read(entry))
                chunks = []
                for conversation in records if isinstance(records, list) else []:
                    if not isinstance(conversation, dict):
                        continue
                    messages = conversation.get("chat_messages", [])
                    mapping = conversation.get("mapping")
                    if isinstance(mapping, dict):
                        messages = [
                            node["message"]
                            for node in mapping.values()
                            if isinstance(node, dict) and isinstance(node.get("message"), dict)
                        ]
                    text_parts = []
                    for message in messages if isinstance(messages, list) else []:
                        if not isinstance(message, dict):
                            continue
                        author = message.get("author") or {}
                        role = author.get("role") if isinstance(author, dict) else None
                        role = role or message.get("sender")
                        if role not in {"user", "assistant", "human"}:
                            continue
                        text = chat_text(message)
                        content = message.get("content")
                        if isinstance(content, dict) and isinstance(content.get("parts"), list):
                            text = "\n".join(
                                part for part in content["parts"] if isinstance(part, str)
                            )
                        if not text and isinstance(message.get("text"), str):
                            text = message["text"]
                        text_parts.append(text[:3000])
                    chunks.append(
                        (
                            str(
                                conversation.get("title")
                                or conversation.get("name")
                                or "Conversation"
                            ),
                            "\n".join(text_parts[:30]),
                        )
                    )
            else:
                continue
            for label, content in chunks:
                text = f"{label[:200]}\n{content}"[: min(8000, remaining)]
                if text.strip() and not SECRET_BYTES.search(text.encode("utf-8", errors="replace")):
                    yield "archive", number, number, text
                    remaining -= len(text)
                if remaining <= 0:
                    return


def links_and_topics(text: str):
    references, topics = {}, {}
    title = None
    in_fence = False
    for number, value in enumerate(text.splitlines(), start=1):
        if value.lstrip().startswith(("```", "~~~")):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        if len(references) < 80:
            for match in INLINE_LINK.finditer(value):
                target = (match[1] or match[2]).split("|", 1)[0].strip("<> ")
                if target.partition("#")[0] and len(references) < 80:
                    if match[2]:
                        target = WIKI_DESTINATION + target
                    references.setdefault(target, number)
        if title is None and (heading := TITLE.match(value)):
            title = normalise_note_title(re.sub(r"[ \t]+#+$", "", heading[1]))
        if len(topics) < 12 and (heading := HEADING.match(value)):
            words = heading[1].casefold().split()
            if 3 <= len(words) <= 8:
                topics.setdefault(" ".join(words), number)
    return references, topics, title


def normalise_note_title(title: str):
    return " ".join(title.casefold().split())


def pdf_process():
    from pypdf import PdfReader

    document = PdfReader(io.BytesIO(sys.stdin.buffer.read(16 * 1024 * 1024 + 1)))
    if document.is_encrypted:
        sys.stdout.buffer.write(b'"encrypted"')
        return
    passages = []
    remaining = MAX_EXTRACTED
    for page in range(min(len(document.pages), MAX_DOCUMENT_PAGES)):
        text = (document.pages[page].extract_text() or "")[: min(12000, remaining)]
        if text.strip():
            passages.append(("page", page + 1, page + 1, text))
            remaining -= len(text)
        if remaining <= 0:
            break
    sys.stdout.buffer.write(json.dumps(passages).encode("utf-8"))


if __name__ == "__main__":
    pdf_process()
