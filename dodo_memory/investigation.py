"""Bounded, read-only investigations and explicit collection coverage manifests."""

import hashlib
import json
import math
import re
import sqlite3
import time
import unicodedata
from pathlib import PurePosixPath

from .evidence import _relative_note, _signature, _verified_bytes
from .store import _search_label, metadata, now, scope_for


class _Limit(Exception):
    pass


def _integer(value, name, low, high):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{name} must be an integer between {low} and {high}")


def _policy(scopes, project, collection, broad, max_items, max_bytes, max_chars, timeout):
    if not isinstance(scopes, (list, tuple, set, frozenset)) or any(
        not isinstance(scope, str) or scope not in {"shared", "private", "archive", "inbox"}
        for scope in scopes
    ):
        raise ValueError("scopes must be an explicit collection of known scope names")
    if type(broad) is not bool or (broad and project is not None):
        raise ValueError("broad must be boolean and cannot be combined with project")
    project = _search_label(project, "project", 2048)
    collection = _search_label(collection, "collection", 160)
    _integer(max_items, "max_items", 1, 1000)
    _integer(max_bytes, "max_bytes", 1, 16 * 1024**2)
    _integer(max_chars, "max_chars", 1024, 500_000)
    if type(timeout) not in (int, float) or not math.isfinite(timeout) or not 0 < timeout <= 60:
        raise ValueError("timeout must be finite and greater than zero, at most 60 seconds")
    return {
        "scopes": tuple(sorted(set(scopes))),
        "project": project,
        "collection": collection,
        "broad": broad,
    }


def _routing(alias, policy):
    clause, values = "", []
    if not policy["broad"]:
        if policy["project"] is None:
            clause += f" AND {alias}.project IS NULL"
        else:
            clause += f" AND ({alias}.project=? OR {alias}.project IS NULL)"
            values.append(policy["project"])
    if policy["collection"] is not None:
        clause += f" AND {alias}.collection=?"
        values.append(policy["collection"])
    return clause, values


def _tables(db):
    return {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}


def _visibility(db, policy, path):
    names = _tables(db)
    placeholders = ",".join("?" for _ in policy["scopes"])
    scope = f"CASE WHEN lower({path}) LIKE 'private/%' THEN 'private' WHEN lower({path}) LIKE 'archive/%' THEN 'archive' WHEN lower({path}) LIKE 'inbox/%' THEN 'inbox' ELSE 'shared' END"
    clause = f"{scope} IN ({placeholders})"
    values = list(policy["scopes"])
    if "purge_denials" in names:
        clause += f" AND NOT EXISTS (SELECT 1 FROM purge_denials p WHERE p.kind='path' AND p.value={path})"
        if "ingest_jobs" in names:
            clause += f" AND NOT EXISTS (SELECT 1 FROM ingest_jobs j JOIN purge_denials p ON (p.kind='source' AND p.value=j.source_id) OR (p.kind='version' AND p.value=j.source_id||':'||j.sha256) WHERE j.path={path})"
        if "accepted_knowledge_versions" in names:
            clause += f" AND NOT EXISTS (SELECT 1 FROM accepted_knowledge_versions v JOIN purge_denials p ON p.kind='knowledge' AND p.value=v.entry_key WHERE v.path={path})"
    if "accepted_knowledge_versions" in names:
        at = now()
        clause += f""" AND NOT EXISTS (
            SELECT 1 FROM accepted_knowledge_versions v WHERE v.path={path} AND (
                v.revision != (SELECT MAX(r.revision) FROM accepted_knowledge_versions r
                    WHERE r.entry_key=v.entry_key AND (r.valid_from IS NULL OR r.valid_from<=?))
                OR (v.valid_from IS NOT NULL AND v.valid_from>?)
                OR (v.valid_to IS NOT NULL AND v.valid_to<=?)
            ))"""
        values.extend((at, at, at))
    return clause, values


def _documents(db, policy):
    clause, values = _visibility(db, policy, "d.path")
    routing, routing_values = _routing("d", policy)
    return clause + routing, [*values, *routing_values]


def _deadline(db, deadline):
    db.set_progress_handler(lambda: int(time.monotonic() >= deadline), 1000)


def _inventory(vault, policy, deadline, *, after=None, max_items=0):
    vault.assert_retention_ready()
    fingerprint = hashlib.sha256(json.dumps(policy, sort_keys=True).encode())
    rows, total, offset = [], 0, 0
    found = after is None
    with vault.connect() as db:
        _deadline(db, deadline)
        db.execute("BEGIN")
        documents, values = _documents(db, policy)
        names = _tables(db)
        curated = (
            ""
            if "ingest_jobs" not in names
            else " AND NOT EXISTS (SELECT 1 FROM ingest_jobs j WHERE j.path=d.path)"
        )
        statement = f"""SELECT 'note:'||d.path AS id,d.path,d.sha256 AS view_sha256,
            NULL AS source_id,NULL AS source_version,NULL AS parser,
            d.project,d.collection,'indexed' AS ingest_status
            FROM indexed_documents d WHERE {documents}{curated}"""
        if "ingest_jobs" in names:
            routing, routing_values = _routing("s", policy)
            placeholders = ",".join("?" for _ in policy["scopes"])
            revoked = vault.source_revocation_sql(db, "s.id", "j.sha256")
            if "purge_denials" in names:
                revoked += " AND NOT EXISTS (SELECT 1 FROM purge_denials p WHERE p.kind='path' AND p.value=j.path)"
            statement += f""" UNION ALL SELECT 'job:'||j.id,j.path,j.view_sha256,
                j.source_id,j.sha256,j.parser,s.project,s.collection,j.status
                FROM ingest_jobs j JOIN sources s ON s.id=j.source_id
                WHERE s.scope IN ({placeholders}) AND j.status!='purging'{routing}{revoked}"""
            values.extend((*policy["scopes"], *routing_values))
        try:
            for row in db.execute(statement + " ORDER BY id", values):
                if time.monotonic() >= deadline:
                    raise TimeoutError("Investigation inventory deadline exceeded")
                item = dict(row)
                fingerprint.update(json.dumps(item, sort_keys=True).encode())
                total += 1
                if item["id"] == after:
                    found, offset = True, total
                elif found and len(rows) < max_items:
                    rows.append(item)
        except sqlite3.OperationalError as error:
            if time.monotonic() >= deadline:
                raise TimeoutError("Investigation inventory deadline exceeded") from error
            raise
    return {
        "items": rows,
        "total": total,
        "offset": offset,
        "found": found,
        "snapshot_sha256": fingerprint.hexdigest(),
    }


def _validate_inspected(vault, inspected, policy):
    path = inspected["path"]
    vault.assert_retention_ready()
    if vault.retention is not None:
        vault.retention.assert_path_allowed(path)
    with vault.connect() as db:
        clause, values = _documents(db, policy)
        live = db.execute(
            f"SELECT sha256 FROM indexed_documents d WHERE {clause} AND d.path=?",
            (*values, path),
        ).fetchone()
        if (
            live is None
            or live["sha256"] != inspected["view_sha256"]
            or not vault.document_published(
                path, inspected["_fields"], db=db, content_hash=inspected["view_sha256"]
            )
        ):
            raise PermissionError("Evidence is no longer published in the requested scope")
    if _signature(vault.resolve(path).stat()) != inspected["_signature"]:
        raise ValueError("Evidence changed during inspection; retry against the new view")


def _inspect(vault, item, policy, budget):
    if time.monotonic() >= budget["deadline"]:
        raise _Limit("deadline")
    if budget["reads"] >= budget["max_reads"]:
        raise _Limit("read_budget")
    if item["path"] is None or item["view_sha256"] is None:
        raise ValueError("Indexed evidence must identify its canonical path and hash")
    path = _relative_note(item["path"], policy["scopes"])
    if scope_for(PurePosixPath(path)) not in policy["scopes"]:
        raise PermissionError("Evidence is unavailable in the granted scopes")
    vault.assert_retention_ready()
    if vault.retention is not None:
        vault.retention.assert_path_allowed(path)
    target = vault.resolve(path)
    before = target.stat()
    if before.st_size > budget["bytes"]:
        raise _Limit("byte_budget")
    content = _verified_bytes(target, item["view_sha256"], budget["bytes"])
    budget["bytes"] -= len(content)
    budget["reads"] += 1
    fields, text = metadata(content.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n"))
    inspected = {
        "path": path,
        "view_sha256": item["view_sha256"],
        "view_text_sha256": hashlib.sha256(text.encode()).hexdigest(),
        "source": fields.get("source", path),
        "source_id": fields.get("source_id"),
        "source_version": fields.get("source_version"),
        "parser": fields.get("parser"),
        "source_uncertainty": {
            "status": fields.get("status", "curated"),
            "observed_at": fields.get("observed_at"),
            "text_origin": fields.get("text_origin"),
            "original_region": "unknown",
        },
        "original_locator": None,
        "text": text,
        "_signature": _signature(before),
        "_fields": fields,
    }
    _validate_inspected(vault, inspected, policy)
    return inspected


def _budget(max_bytes, max_reads, timeout):
    return {
        "bytes": max_bytes,
        "reads": 0,
        "max_reads": max_reads,
        "deadline": time.monotonic() + timeout,
    }


def _review(
    vault,
    collection,
    *,
    scopes=("shared",),
    project=None,
    broad=False,
    max_items=100,
    max_bytes=8 * 1024**2,
    max_chars=60_000,
    timeout=10,
    cursor=None,
):
    """Enumerate every eligible version/parser job and curated view, independently of search."""
    policy = _policy(scopes, project, collection, broad, max_items, max_bytes, max_chars, timeout)
    if policy["collection"] is None:
        raise ValueError("Exhaustive review requires an explicit collection")
    if cursor is not None and (
        not isinstance(cursor, dict)
        or set(cursor) != {"snapshot_sha256", "after"}
        or not isinstance(cursor["snapshot_sha256"], str)
        or (cursor["after"] is not None and not isinstance(cursor["after"], str))
    ):
        raise ValueError("cursor must identify a collection snapshot and its last consumed item")
    budget = _budget(max_bytes, max_items, timeout)
    inventory = _inventory(
        vault,
        policy,
        budget["deadline"],
        after=cursor["after"] if cursor else None,
        max_items=max_items,
    )
    snapshot, offset = inventory["snapshot_sha256"], inventory["offset"]
    if cursor is not None and (cursor["snapshot_sha256"] != snapshot or not inventory["found"]):
        raise ValueError("Collection snapshot changed; restart exhaustive enumeration")
    items, advanced = [], 0
    inspected_items = []
    counts = dict.fromkeys(("eligible", "processed", "failed", "excluded", "unprocessed"), 0)

    def manifest():
        remaining = inventory["total"] - offset - advanced
        after = items[advanced - 1]["id"] if advanced else cursor["after"] if cursor else None
        return {
            "policy": policy,
            "universe": "registered ingestion jobs and indexed curated views; not unregistered files",
            "processing": "complete canonical text inspection; no semantic review or original-region claim",
            "items": items,
            "counts": counts,
            "complete": offset == 0
            and remaining == 0
            and not (counts["failed"] or counts["excluded"]),
            "enumeration": {
                "snapshot_sha256": snapshot,
                "total": inventory["total"],
                "offset": offset,
                "returned": len(items),
                "remaining": remaining,
                "complete": offset == 0 and len(items) == inventory["total"],
                "next_cursor": {"snapshot_sha256": snapshot, "after": after} if remaining else None,
            },
        }

    if len(json.dumps(manifest())) > max_chars:
        raise ValueError("Review policy metadata exceeds the output budget")
    for entry in inventory["items"]:
        item = {**entry, "eligible": entry["ingest_status"] == "indexed"}
        if item["ingest_status"] == "failed":
            item.update(status="failed", reason="ingestion_failed")
        elif not item["eligible"]:
            item.update(status="excluded", reason="ingestion_" + item["ingest_status"])
        else:
            try:
                inspected = _inspect(vault, item, policy, budget)
                inspected_items.append(inspected)
                item.update(
                    {key: value for key, value in inspected.items() if not key.startswith("_")},
                    status="processed",
                )
            except _Limit as error:
                item.update(status="unprocessed", reason=str(error))
            except (OSError, ValueError):
                item.update(status="failed", reason="evidence_unavailable")
        items.append(item)
        counts["eligible"] += int(item["eligible"])
        counts[item["status"]] += 1
        advanced += int(item["status"] != "unprocessed")
        if len(json.dumps(manifest())) > max_chars:
            counts[item["status"]] -= 1
            advanced -= int(item["status"] != "unprocessed")
            item = {
                **entry,
                "eligible": item["eligible"],
                "status": "unprocessed",
                "reason": "output_budget",
            }
            items[-1] = item
            counts["unprocessed"] += 1
            if len(json.dumps(manifest())) > max_chars:
                items.pop()
                counts["unprocessed"] -= 1
                counts["eligible"] -= int(item["eligible"])
            break
        if item["status"] == "unprocessed":
            break
    for inspected in inspected_items:
        _validate_inspected(vault, inspected, policy)
    result = manifest()
    if (
        result["complete"]
        and _inventory(vault, policy, budget["deadline"])["snapshot_sha256"] != snapshot
    ):
        raise ValueError("Collection snapshot changed during inspection; restart review")
    return result, inspected_items


def review_collection(
    vault,
    collection,
    *,
    scopes=("shared",),
    project=None,
    broad=False,
    max_items=100,
    max_bytes=8 * 1024**2,
    max_chars=60_000,
    timeout=10,
    cursor=None,
):
    """Return bounded collection coverage; a cursor never certifies earlier pages as read."""
    result, _ = _review(
        vault,
        collection,
        scopes=scopes,
        project=project,
        broad=broad,
        max_items=max_items,
        max_bytes=max_bytes,
        max_chars=max_chars,
        timeout=timeout,
        cursor=cursor,
    )
    return result


def _fold(value):
    return "".join(
        char
        for char in unicodedata.normalize("NFKD", value.casefold())
        if not unicodedata.combining(char)
    )


def _quote(inspected, query, context_chars, deadline):
    text = inspected["text"]
    terms = {_fold(term) for term in re.findall(r"[^\W_]+", query, re.UNICODE)[:32]}
    match = None
    for index, word in enumerate(re.finditer(r"[^\W_]+", text, re.UNICODE)):
        if index % 1000 == 0 and time.monotonic() >= deadline:
            raise _Limit("deadline")
        if _fold(word.group()) in terms:
            match = word
            break
    if match is None:
        return None
    start = max(text.rfind("\n", 0, match.start()) + 1, match.start() - 500)
    end = text.find("\n", match.end())
    end = min(len(text) if end < 0 else end, match.end() + 500)
    left, right = max(0, start - context_chars), min(len(text), end + context_chars)
    return {
        **{
            key: value
            for key, value in inspected.items()
            if key != "text" and not key.startswith("_")
        },
        "quote": text[start:end],
        "context": text[left:right],
        "locator": {
            "kind": "canonical_text_characters",
            "start": start,
            "end": end,
            "context_start": left,
            "context_end": right,
        },
    }


def _candidates(vault, query, policy, maximum, deadline):
    words = re.findall(r"[^\W_]+", query, re.UNICODE)[:32]
    if not words:
        return []
    vault.assert_retention_ready()
    with vault.connect() as db:
        _deadline(db, deadline)
        clause, values = _documents(db, policy)
        placeholders = ",".join("?" for _ in policy["scopes"])
        try:
            visible, visible_values = _visibility(db, policy, "chunks.path")
            match = " OR ".join('"' + word + '"' for word in words)
            if db.execute(
                f"""SELECT 1 FROM chunks WHERE chunks MATCH ?
                AND chunks.scope IN ({placeholders}) AND {visible} AND NOT EXISTS (
                    SELECT 1 FROM indexed_documents d WHERE d.path=chunks.path) LIMIT 1""",
                (match, *policy["scopes"], *visible_values),
            ).fetchone():
                raise ValueError(
                    "Search context metadata is missing; explicitly reindex before investigating"
                )
            rows = db.execute(
                f"""SELECT DISTINCT chunks.path,d.sha256 AS view_sha256 FROM chunks
                JOIN indexed_documents d ON d.path=chunks.path
                WHERE chunks MATCH ? AND chunks.scope IN ({placeholders}) AND {clause}
                ORDER BY CASE WHEN d.project=? THEN 0
                    WHEN d.project IS NULL AND d.accepted=1 THEN 1 ELSE 2 END,
                    bm25(chunks),chunks.path,chunks.rowid LIMIT ?""",
                (match, *policy["scopes"], *values, policy["project"], maximum),
            ).fetchall()
        except sqlite3.OperationalError as error:
            if time.monotonic() >= deadline:
                raise _Limit("deadline") from error
            raise
    return list({row["path"]: dict(row) for row in rows}.values())


def investigate(
    vault,
    question,
    subquestions,
    *,
    scopes=("shared",),
    project=None,
    collection=None,
    broad=False,
    exhaustive=False,
    max_items=100,
    max_hits=5,
    max_reads=20,
    max_bytes=8 * 1024**2,
    max_chars=60_000,
    context_chars=500,
    timeout=10,
    cursor=None,
):
    """Inspect evidence for an explicit decomposition; found means evidence, never an answer."""
    policy = _policy(scopes, project, collection, broad, max_items, max_bytes, max_chars, timeout)
    question = _search_label(question, "question", 2000)
    if (
        question is None
        or not isinstance(subquestions, (list, tuple))
        or not 1 <= len(subquestions) <= 12
    ):
        raise ValueError("Investigation needs a question and 1 to 12 explicit subquestions")
    decomposition, identifiers = [], set()
    for part in subquestions:
        if not isinstance(part, dict) or set(part) != {"id", "query"}:
            raise ValueError("Each subquestion must contain exactly id and query")
        identity, query = (
            _search_label(part["id"], "subquestion id", 80),
            _search_label(part["query"], "query", 2000),
        )
        if identity is None or query is None or identity in identifiers:
            raise ValueError("Subquestion ids and queries must be non-empty; ids must be unique")
        identifiers.add(identity)
        decomposition.append({"id": identity, "query": query})
    _integer(max_hits, "max_hits", 1, 20)
    _integer(max_reads, "max_reads", 1, 100)
    _integer(context_chars, "context_chars", 0, 2000)
    if type(exhaustive) is not bool or (cursor is not None and not exhaustive):
        raise ValueError("exhaustive must be boolean; cursors apply only to exhaustive review")
    budget = _budget(max_bytes, max_reads, timeout)
    coverage = {"exhaustive": exhaustive, "complete": False}
    results = [
        {**part, "status": "unknown", "evidence": [], "failures": []} for part in decomposition
    ]
    result = {
        "question": question,
        "policy": policy,
        "subquestions": results,
        "coverage": coverage,
        "answer": None,
        "status_meaning": "found is inspected retrieval evidence; unknown is unresolved, not proof of absence",
        "unresolved": [part["id"] for part in results],
    }
    baseline = len(json.dumps(result))
    if baseline + 1024 > max_chars:
        raise ValueError("Question decomposition exceeds the output budget")
    cache = {}
    if exhaustive:
        review, inspected_items = _review(
            vault,
            collection,
            scopes=scopes,
            project=project,
            broad=broad,
            max_items=min(max_items, max_reads),
            max_bytes=max_bytes,
            max_chars=max_chars - baseline,
            timeout=timeout,
            cursor=cursor,
        )
        coverage.update(review)
        processed = {item["path"] for item in coverage["items"] if item["status"] == "processed"}
        cache = {item["path"]: item for item in inspected_items if item["path"] in processed}
        for item in coverage["items"]:
            item.pop("text", None)
    for part in results:
        evidence, failures = part["evidence"], part["failures"]
        try:
            candidates = (
                list(cache.values())
                if exhaustive
                else _candidates(vault, part["query"], policy, max_hits, budget["deadline"])
            )
            for candidate in candidates:
                if len(evidence) >= max_hits:
                    break
                try:
                    path = candidate["path"]
                    if path not in cache:
                        cache[path] = _inspect(vault, candidate, policy, budget)
                    quote = _quote(cache[path], part["query"], context_chars, budget["deadline"])
                    if quote is None:
                        continue
                    evidence.append(quote)
                    if len(json.dumps(result)) + 100 > max_chars:
                        evidence.pop()
                        raise _Limit("output_budget")
                except _Limit as error:
                    failures.append({"reason": str(error)})
                    break
                except (OSError, ValueError):
                    failures.append({"path": candidate["path"], "reason": "evidence_unavailable"})
        except _Limit as error:
            failures.append({"reason": str(error)})
        part["status"] = "found" if evidence else "failed" if failures else "unknown"
    for inspected in cache.values():
        if "_signature" in inspected:
            _validate_inspected(vault, inspected, policy)
    vault.assert_retention_ready()
    result["unresolved"] = [part["id"] for part in results if part["status"] != "found"]
    if len(json.dumps(result)) > max_chars:
        raise ValueError("Investigation failure metadata exceeds the output budget")
    return result
