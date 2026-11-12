"""Deterministic private evidence anchors and resumable local question-draft receipts."""

import hashlib
import json
import os
import random
import re
import unicodedata
from collections import defaultdict
from pathlib import Path
from urllib.parse import urlsplit

from .evaluation import digest
from .source_registry import atomic_bytes


def anchor_candidates(document, text):
    """Quotes/spans use Unicode characters in LF-normalized text, never model output."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    beginning = text.find("\n---\n", 4) + 5 if text.startswith("---\n") else 0
    rows = []
    for match in re.finditer(r"\S[\s\S]*?(?=\n[ \t]*\n|\Z)", text[beginning:]):
        quote = match.group().strip()
        if not 120 <= len(quote) <= 1600 or len(re.findall(r"[^\W\d_]+", quote)) < 25:
            continue
        if quote.startswith(("---", chr(96) * 3, "~~~")) or all(
            line.startswith("#") for line in quote.splitlines()
        ):
            continue
        start = beginning + match.start()
        end = start + len(quote)
        rows.append(
            {
                **{
                    key: document[key]
                    for key in ["path", "source_id", "source_version", "family_id"]
                },
                "quote": quote,
                "span": {"start": start, "end": end},
                "locator": {
                    "line_start": text.count("\n", 0, start) + 1,
                    "line_end": text.count("\n", 0, end - 1) + 1,
                },
            }
        )
    return rows


def question_text_key(value):
    return " ".join(unicodedata.normalize("NFC", value).casefold().split())


def allocate_anchors(rows, *, total=240, development_count=60, seed=17):
    """Assign whole families before model calls; fail rather than approximate split sizes."""
    families, seen = defaultdict(list), set()
    for row in sorted(rows, key=digest):
        key = question_text_key(row["quote"])
        if key not in seen:
            seen.add(key)
            families[row["family_id"]].append(row)
    if len(families) < 2:
        raise ValueError("Question authoring requires independent source families")
    if not 0 < development_count < total or sum(map(len, families.values())) < total:
        raise ValueError("Insufficient distinct qualified anchors for the requested exact split")
    names = sorted(families)
    random.Random(seed).shuffle(names)
    capacities = {name: min(total, len(families[name])) for name in names}
    states = {0: ()}
    for name in names:
        for capacity, chosen in list(states.items()):
            combined = capacity + capacities[name]
            proposed = (*chosen, name)
            if combined not in states or len(proposed) < len(states[combined]):
                states[combined] = proposed
    capacity_total = sum(capacities.values())
    feasible = [
        (capacity, chosen)
        for capacity, chosen in states.items()
        if development_count <= capacity <= capacity_total - (total - development_count)
    ]
    if not feasible:
        raise ValueError("Independent families cannot support the exact requested split")
    _, chosen = min(
        feasible,
        key=lambda item: (
            abs(item[0] - capacity_total * development_count / total),
            abs(len(item[1]) - len(names) * development_count / total),
            item[0],
        ),
    )
    development = set(chosen)

    def choose(group, count):
        buckets = [list(families[name]) for name in names if (name in development) == group]
        selected = []
        while len(selected) < count:
            for bucket in buckets:
                if bucket and len(selected) < count:
                    selected.append(bucket.pop(0))
        return selected

    dev, held = choose(True, development_count), choose(False, total - development_count)
    plan = []
    while dev or held:
        for bucket, split, count in [(dev, "development", 1), (held, "held_out", 3)]:
            for _ in range(min(count, len(bucket))):
                row = bucket.pop(0)
                ordinal = len(plan) + 1
                plan.append(
                    {
                        **row,
                        "id": f"a{ordinal:04d}",
                        "reference_id": f"e{ordinal:04d}",
                        "split": split,
                        "requested_language": "en" if ordinal % 2 else "pt",
                    }
                )
    return plan


def validate_drafts(payload, anchors):
    if (
        not isinstance(payload, dict)
        or set(payload) != {"questions"}
        or not isinstance(payload["questions"], list)
    ):
        raise ValueError("Local model did not return the required question object")
    expected = {row["id"] for row in anchors}
    drafts, seen = {}, set()
    for row in payload["questions"]:
        if not isinstance(row, dict) or set(row) != {"id", "question"} or row["id"] not in expected:
            raise ValueError("Question draft changed its fixed evidence identity or schema")
        query = row["question"]
        if not isinstance(query, str) or not 12 <= len(query.strip()) <= 320 or "\n" in query:
            raise ValueError("Question draft is empty, multiline, or outside its length bound")
        query = query.strip()
        canonical = question_text_key(query)
        if row["id"] in drafts or canonical in seen:
            raise ValueError("Duplicate question ID or wording in local model output")
        drafts[row["id"]] = query
        seen.add(canonical)
    if drafts.keys() != expected:
        raise ValueError("Local model omitted a requested evidence anchor")
    return drafts


def store_batch(path, payload):
    envelope = {"payload": payload, "sha256": digest(payload)}
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != envelope:
            raise ValueError("Question batch receipt is immutable")
        return
    atomic_bytes(path, json.dumps(envelope, ensure_ascii=False, sort_keys=True).encode())
    if os.name != "nt":
        path.chmod(0o600)


def load_batch(path, input_sha):
    if not path.exists():
        return None
    envelope = json.loads(path.read_text(encoding="utf-8"))
    payload = envelope["payload"]
    if envelope["sha256"] != digest(payload):
        raise ValueError("Question batch receipt checksum mismatch")
    if payload["input_sha256"] != input_sha:
        raise ValueError("Question batch receipt belongs to different frozen inputs")
    return payload


def observe_generator(
    profile, endpoint, client, *, expected_build, verify_artifact=False, memory_check=None
):
    """Observe the running owner-approved model; never start or reconfigure a provider."""
    url = urlsplit(endpoint)
    if (
        url.scheme != "http"
        or url.hostname not in {"127.0.0.1", "::1"}
        or url.port != profile["port"]
    ):
        raise PermissionError("Generator identity requires the approved numeric loopback port")
    origin = endpoint.rsplit("/v1/chat/completions", 1)[0]
    models = client.get(origin + "/v1/models")
    models.raise_for_status()
    props = client.get(origin + "/props")
    props.raise_for_status()
    observed = props.json()
    argv = profile["argv"]
    model = Path(argv[argv.index("-m") + 1]).absolute()
    executable = Path(argv[0]).absolute()
    if (
        [row["id"] for row in models.json()["data"]] != [profile["alias"]]
        or observed.get("model_alias") != profile["alias"]
        or Path(observed.get("model_path", "")).absolute() != model
        or observed.get("build_info") != expected_build
        or model.resolve() != model
        or executable.resolve() != executable
    ):
        raise ValueError("Live generator model/build/path differs from the approved identity")
    before = model.stat()
    if before.st_size != profile["artifact"]["bytes"]:
        raise ValueError("Generator artifact byte count differs from its approved pin")
    if verify_artifact:
        checksum = hashlib.sha256()
        with model.open("rb") as stream:
            while block := stream.read(8 * 1024**2):
                if memory_check:
                    memory_check()
                checksum.update(block)
        if checksum.hexdigest() != profile["artifact"]["sha256"]:
            raise ValueError("Generator artifact hash differs from its approved pin")
    after = model.stat()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        raise ValueError("Generator artifact changed during identity verification")
    with executable.open("rb") as stream:
        executable_sha = hashlib.file_digest(stream, "sha256").hexdigest()
    fields = [
        "model_alias",
        "model_ftype",
        "model_path",
        "default_generation_settings",
        "total_slots",
        "chat_template",
        "chat_template_caps",
        "bos_token",
        "eos_token",
        "build_info",
    ]
    if any(field not in observed for field in fields):
        raise ValueError(
            "Live generator omitted a required tokenizer/template/config identity field"
        )
    return {
        "profile_sha256": digest(profile),
        "artifact_sha256": profile["artifact"]["sha256"],
        "executable_sha256": executable_sha,
        "runtime_sha256": digest({field: observed[field] for field in fields}),
        "model_stat": {
            "device": after.st_dev,
            "inode": after.st_ino,
            "bytes": after.st_size,
            "mtime_ns": after.st_mtime_ns,
        },
    }
