"""Frozen retrieval evaluation with grouped splits and paired cluster uncertainty."""

import hashlib
import json
import math
import os
import random
import re
import tempfile
import unicodedata
from collections import defaultdict
from pathlib import Path

from .store import writer_lock


def digest(value):
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _normalized(text):
    return " ".join(unicodedata.normalize("NFC", text).split())


def _reference_locator(document, reference):
    locator, text = reference["locator"], document["text"]
    quote = _normalized(reference["quote"])
    if set(locator) == {"line_start", "line_end"}:
        start, end = locator["line_start"], locator["line_end"]
        lines = text.splitlines()
        if type(start) is not int or type(end) is not int or not 1 <= start <= end <= len(lines):
            raise ValueError("Reference line locator is outside the frozen view")
        if quote not in _normalized("\n".join(lines[start - 1 : end])):
            raise ValueError("Reference quote does not occur at its line locator")
    elif set(locator) == {"heading"} and isinstance(locator["heading"], str):
        headings = list(re.finditer(r"(?m)^(#{1,6}) (.+)$", text))
        matching = [heading for heading in headings if heading[2].strip() == locator["heading"]]
        if len(matching) != 1:
            raise ValueError("Reference heading locator must be unique in its frozen view")
        heading = matching[0]
        end = next(
            (
                other.start()
                for other in headings
                if other.start() > heading.start() and len(other[1]) <= len(heading[1])
            ),
            len(text),
        )
        if quote not in _normalized(text[heading.start() : end]):
            raise ValueError("Reference quote does not occur at its heading locator")
    elif "page" in locator and set(locator) <= {"page", "bbox", "units"}:
        if type(locator["page"]) is not int or locator["page"] < 1:
            raise ValueError("Reference page locator must be a positive integer")
        if "bbox" in locator:
            bbox = locator["bbox"]
            if (
                locator.get("units") not in {"pt", "normalized"}
                or not isinstance(bbox, list)
                or len(bbox) != 4
                or any(
                    type(value) not in {int, float} or not math.isfinite(value) for value in bbox
                )
                or bbox[0] > bbox[2]
                or bbox[1] > bbox[3]
                or (locator["units"] == "normalized" and any(not 0 <= value <= 1 for value in bbox))
            ):
                raise ValueError("Reference bbox locator has invalid coordinates or units")
        regions = [
            region for region in document.get("locators", []) if region["locator"] == locator
        ]
        start, end = reference["span"]["start"], reference["span"]["end"]
        if not any(
            0 <= region["span"]["start"] <= start < end <= region["span"]["end"] <= len(text)
            for region in regions
        ):
            raise ValueError("Reference page locator lacks matching frozen structured provenance")
    else:
        raise ValueError("Unsupported reference locator")


def dataset(bank, references, manifest, root, scopes=("shared",)):
    bank, references, manifest = json.loads(
        json.dumps([bank, references, manifest], allow_nan=False)
    )
    root = Path(root).absolute()
    if root.resolve() != root:
        raise ValueError("Corpus root cannot redirect through symlinks")
    if bank.get("schema_version") != 1 or bank.get("suite") not in {
        "synthetic",
        "pilot",
        "smoke",
        "full",
    }:
        raise ValueError("Expected a versioned synthetic, pilot, smoke or full question bank")
    questions = bank["questions"]
    required_count = {"smoke": 60, "full": 240}.get(bank["suite"])
    if required_count and len(questions) != required_count:
        raise ValueError(f"{bank['suite']} requires {required_count} authored questions")
    if (
        not questions
        or any(not isinstance(q["id"], str) or not q["id"].strip() for q in questions)
        or len({q["id"] for q in questions}) != len(questions)
    ):
        raise ValueError("Question IDs must be non-empty and unique")
    documents, source_versions, total_bytes = {}, {}, 0
    for document in manifest["documents"]:
        relative = document["path"]
        path = root / relative
        if Path(relative).is_absolute() or ".." in Path(relative).parts or ":" in relative:
            raise ValueError("Corpus document path must be relative")
        if path.resolve() != path or not path.is_relative_to(root):
            raise ValueError("Corpus document cannot escape or redirect")
        if document["scope"] not in scopes:
            raise PermissionError("Corpus source scope is not granted by the operator")
        for key in ["source_id", "source_version", "family_id"]:
            if not isinstance(document[key], str) or not document[key]:
                raise ValueError(f"Corpus requires explicit {key}")
        with path.open("rb") as stream:
            content = stream.read(16 * 1024 * 1024 + 1)
        total_bytes += len(content)
        if len(content) > 16 * 1024 * 1024 or total_bytes > 128 * 1024 * 1024:
            raise ValueError("Normalized corpus exceeds the 16 MiB/view or 128 MiB total budget")
        if hashlib.sha256(content).hexdigest() != document["sha256"]:
            raise ValueError("Frozen source hash mismatch")
        if relative in documents:
            raise ValueError("Duplicate corpus path")
        text = content.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")
        documents[relative] = {**document, "text": text}
        source_key = document["source_id"]
        if (
            source_key in source_versions
            and source_versions[source_key]["family_id"] != document["family_id"]
        ):
            raise ValueError("Stable source revisions cannot belong to different split families")
        source_versions[source_key] = document
    refs = {}
    for reference in references:
        if not isinstance(reference["id"], str) or not reference["id"].strip():
            raise ValueError("Reference IDs must be non-empty strings")
        if reference["id"] in refs:
            raise ValueError("Duplicate reference ID")
        document = documents[reference["path"]]
        if (reference["source_id"], reference["source_version"]) != (
            document["source_id"],
            document["source_version"],
        ):
            raise ValueError("Reference source version differs from its frozen document")
        start, end = reference["span"]["start"], reference["span"]["end"]
        if not (
            type(start) is int and type(end) is int and 0 <= start < end <= len(document["text"])
        ):
            raise ValueError("Reference quote span is outside its source")
        if (
            not reference.get("verified_by")
            or not isinstance(reference.get("locator"), dict)
            or not reference["locator"]
        ):
            raise ValueError("References require an independently reviewed locator")
        if not reference["quote"] or _normalized(reference["quote"]) not in _normalized(
            document["text"][start:end]
        ):
            raise ValueError("Reference quote does not occur in its frozen span")
        _reference_locator(document, reference)
        refs[reference["id"]] = reference
    for question in questions:
        if not isinstance(question["query"], str) or not question["query"].strip():
            raise ValueError("Question query is empty")
        if (
            not isinstance(question["source_families"], list)
            or not question["source_families"]
            or any(
                not isinstance(value, str) or not value.strip()
                for value in question["source_families"]
            )
            or not isinstance(question["paraphrase_family"], str)
            or not question["paraphrase_family"].strip()
        ):
            raise ValueError("Questions need source and paraphrase families")
        if type(question["absence"]) is not bool or question["absence"] == bool(
            question["expected"]
        ):
            raise ValueError(
                "Absence questions have no expected evidence; positive questions require evidence"
            )
        if len(set(question["expected"])) != len(question["expected"]):
            raise ValueError("Expected evidence IDs must be unique")
        for ref_id in question["expected"]:
            reference = refs[ref_id]
            family = documents[reference["path"]]["family_id"]
            if family not in question["source_families"]:
                raise ValueError("Question omits an expected source family")
    return {
        "bank": bank,
        "references": refs,
        "documents": documents,
        "hashes": {
            "questions": digest(bank),
            "references": digest(references),
            "corpus": digest(manifest),
        },
    }


def assign_splits(questions, development_count, seed):
    if not 0 < development_count < len(questions):
        raise ValueError("Development count must leave independent held-out questions")
    parents = {}

    def find(key):
        parents.setdefault(key, key)
        while parents[key] != key:
            key = parents[key]
        return key

    for question in questions:
        nodes = ["source:" + value for value in question["source_families"]]
        nodes.append("paraphrase:" + question["paraphrase_family"])
        roots = sorted({find(node) for node in nodes})
        for node in roots[1:]:
            parents[node] = roots[0]
    groups = defaultdict(list)
    for question in questions:
        groups[find("source:" + question["source_families"][0])].append(question["id"])
    if len(groups) < 2:
        raise ValueError("No independent source/paraphrase groups remain for held-out evaluation")
    components = sorted(groups.values(), key=lambda ids: sorted(ids))
    random.Random(seed).shuffle(components)
    reachable = {0: ()}
    for index, ids in enumerate(components):
        for count, selected in list(reachable.items()):
            reachable.setdefault(count + len(ids), (*selected, index))
    choices = [count for count in reachable if 0 < count < len(questions)]
    target = min(choices, key=lambda count: (abs(count - development_count), count))
    development = set(reachable[target])
    result = {}
    for index, ids in enumerate(components):
        group = digest(sorted(ids))
        for question_id in ids:
            result[question_id] = {
                "split": "development" if index in development else "held_out",
                "group": group,
            }
    return result


def score_retrieval(question, references, hits, answer=None):
    hits = hits[:10]
    expected = [references[key] for key in question["expected"]]
    found, located, quoted, read, first_ranks = [], [], [], [], []
    for reference in expected:
        matching = [
            (index, hit)
            for index, hit in enumerate(hits, 1)
            if (hit["source_id"], hit["source_version"])
            == (reference["source_id"], reference["source_version"])
        ]
        found.append(bool(matching))
        quote = _normalized(reference["quote"])
        located.append(
            any(
                hit.get("locator") == reference["locator"]
                and (
                    quote in _normalized(hit.get("text", ""))
                    or quote in _normalized(hit.get("read_text", ""))
                )
                for _, hit in matching
            )
        )
        quoted.append(any(quote in _normalized(hit.get("text", "")) for _, hit in matching))
        read.append(any(quote in _normalized(hit.get("read_text", "")) for _, hit in matching))
        if matching:
            first_ranks.append(matching[0][0])
    required_sources = {(ref["source_id"], ref["source_version"]) for ref in expected}
    seen = set()
    gains = []
    for hit in hits:
        source = (hit["source_id"], hit["source_version"])
        gains.append(int(source in required_sources and source not in seen))
        seen.add(source)
    dcg = sum(gain / math.log2(rank + 1) for rank, gain in enumerate(gains, 1))
    ideal = sum(1 / math.log2(rank + 1) for rank in range(1, min(10, len(required_sources)) + 1))
    count = len(expected)
    evidence_found = [quote or context for quote, context in zip(quoted, read, strict=True)]
    scores = {
        "source_recall_at_10": len(required_sources & seen) / len(required_sources)
        if required_sources
        else None,
        "evidence_recall_at_10": sum(evidence_found) / count if count else None,
        "all_required_evidence_at_10": float(all(evidence_found)) if count else None,
        "exact_locator_recall_at_10": sum(
            found and location for found, location in zip(evidence_found, located, strict=True)
        )
        / count
        if count
        else None,
        "quote_coverage": sum(quoted) / count if count else None,
        "read_coverage": sum(read) / count if count else None,
        "mrr_at_10": 1 / min(first_ranks) if first_ranks else (0.0 if count else None),
        "ndcg_at_10": dcg / ideal if ideal else None,
        "empty_retrieval": float(not hits),
        "answer_abstention_accuracy": None,
        "answer_correctness": None,
    }
    if answer is not None:
        if type(answer.get("abstained")) is not bool:
            raise ValueError("Answer abstention must be explicit")
        scores["answer_abstention_accuracy"] = float(answer["abstained"] == question["absence"])
        if answer.get("grade") is not None:
            grade = answer["grade"]
            if (
                not answer.get("graded_by")
                or type(grade) not in {int, float}
                or not math.isfinite(grade)
                or not 0 <= grade <= 1
            ):
                raise ValueError("Answer grades require a reviewer and value in [0,1]")
            scores["answer_correctness"] = float(grade)
    return scores


def percentile95(values):
    if not values or any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("Latency observations must be finite, non-negative seconds")
    return sorted(values)[math.ceil(0.95 * len(values)) - 1]


def _record_path(directory, key):
    if not re.fullmatch(r"[0-9a-f]{64}", key):
        raise ValueError("Evaluation record key must be a SHA-256 digest")
    return Path(directory) / (key + ".json")


def save_result(directory, key, payload):
    path = _record_path(directory, key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with writer_lock(path.parent):
        existing = cached_result(path.parent, key)
        if existing is not None:
            if existing != payload:
                raise ValueError("Completed evaluation receipt conflict")
            return
        envelope = {"payload": payload, "payload_sha256": digest(payload)}
        descriptor, temporary = tempfile.mkstemp(prefix=key + ".", suffix=".tmp", dir=path.parent)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
                json.dump(envelope, stream, ensure_ascii=False, sort_keys=True, allow_nan=False)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            Path(temporary).unlink(missing_ok=True)


def cached_result(directory, key):
    path = _record_path(directory, key)
    if not path.exists():
        return None
    envelope = json.loads(path.read_text(encoding="utf-8"))
    if digest(envelope["payload"]) != envelope["payload_sha256"]:
        raise ValueError("Evaluation result checksum mismatch")
    return envelope["payload"] if envelope["payload"].get("status") == "complete" else None


def paired_comparison(left, right, seed, iterations=2000):
    if iterations < 100:
        raise ValueError("Bootstrap requires at least 100 iterations")
    a = {row["question_id"]: row for row in left}
    b = {row["question_id"]: row for row in right}
    if len(a) != len(left) or len(b) != len(right):
        raise ValueError("Paired comparisons require unique question IDs")
    differences = defaultdict(list)
    for question_id in sorted(a.keys() & b.keys()):
        first, second = a[question_id], b[question_id]
        if first["group"] != second["group"]:
            raise ValueError("Paired source-group assignment differs")
        if not all(math.isfinite(row["value"]) for row in [first, second]):
            raise ValueError("Paired values must be finite")
        differences[first["group"]].append(second["value"] - first["value"])
    if not differences:
        return {"paired_questions": 0, "independent_groups": 0, "difference": None, "ci95": None}
    groups = list(differences.values())
    values = [value for group in groups for value in group]
    bootstrap, rng = [], random.Random(seed)
    for _ in range(iterations):
        sample = [value for _ in groups for value in rng.choice(groups)]
        bootstrap.append(sum(sample) / len(sample))
    bootstrap.sort()
    interval = None
    if len(groups) >= 2:
        interval = [
            bootstrap[math.floor(0.025 * (iterations - 1))],
            bootstrap[math.ceil(0.975 * (iterations - 1))],
        ]
    return {
        "paired_questions": len(values),
        "independent_groups": len(groups),
        "difference": sum(values) / len(values),
        "ci95": interval,
    }
