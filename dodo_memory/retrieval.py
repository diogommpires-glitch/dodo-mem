"""Atomic vector publication and gateway-owned eligibility before retrieval limits."""

import hashlib
import re
import time
from collections import defaultdict

from .embeddings import EmbeddingError, VectorIndex, reciprocal_rank_fusion
from .investigation import _documents, _visibility
from .store import metadata

MAX_SCAN_BYTES = 128 * 1024**2
MAX_SCAN_CHUNKS = 100000


def synchronize_vectors(vault, db, chunks, *, paths=None, reset=False):
    selected = sorted({row[0] for row in chunks}) if reset or paths is None else paths
    db.execute("""CREATE TABLE IF NOT EXISTS embedding_pending (
        path TEXT PRIMARY KEY, source_sha256 TEXT NOT NULL, reason TEXT NOT NULL)""")
    try:
        _publish_vectors(vault, db, chunks, paths=paths, reset=reset)
    except EmbeddingError as error:
        exists = db.execute("SELECT 1 FROM sqlite_master WHERE name='embedding_vectors'").fetchone()
        if exists:
            if reset:
                db.execute("DELETE FROM embedding_vectors")
            else:
                db.executemany(
                    "DELETE FROM embedding_vectors WHERE path=?", ((p,) for p in selected)
                )
        db.executemany(
            "INSERT OR REPLACE INTO embedding_pending SELECT path,sha256,? "
            "FROM indexed_documents WHERE path=?",
            ((str(error)[:500], p) for p in selected),
        )
    else:
        if reset:
            db.execute("DELETE FROM embedding_pending")
        else:
            db.executemany("DELETE FROM embedding_pending WHERE path=?", ((p,) for p in selected))


def _publish_vectors(vault, db, chunks, *, paths=None, reset=False):
    with vault.embedding() as embedder:
        if embedder is None:
            exists = db.execute(
                "SELECT 1 FROM sqlite_master WHERE name='embedding_vectors'"
            ).fetchone()
            if exists:
                if reset:
                    db.execute("DELETE FROM embedding_vectors")
                else:
                    db.executemany(
                        "DELETE FROM embedding_vectors WHERE path=?",
                        ((path,) for path in paths or ()),
                    )
            return
        index = VectorIndex(db, embedder.space)
        grouped = defaultdict(list)
        for path, _, heading, body, _, _ in chunks:
            grouped[path].append({"heading": heading, "body": body})
        selected = sorted(grouped) if paths is None else paths
        for path in selected:
            row = db.execute(
                "SELECT sha256 FROM indexed_documents WHERE path=?", (path,)
            ).fetchone()
            if row is None:
                index.delete(path)
                continue
            digest = row["sha256"]

            def validate(path=path, digest=digest):
                vault.assert_retention_ready()
                content = vault.resolve(path).read_bytes()
                fields, _ = metadata(
                    content.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n")
                )
                current = db.execute(
                    "SELECT sha256 FROM indexed_documents WHERE path=?", (path,)
                ).fetchone()
                if (
                    hashlib.sha256(content).hexdigest() != digest
                    or current is None
                    or current["sha256"] != digest
                    or not vault.document_published(path, fields, db=db, content_hash=digest)
                ):
                    raise ValueError(
                        "Canonical source changed during vector publication; retry indexing"
                    )

            validate()
            if index.current(path, digest, grouped[path]):
                validate()
                continue
            prepared = embedder.embed_chunks(grouped[path])
            index.replace(path, digest, prepared, revalidate=validate)
        if reset:
            db.execute(
                """DELETE FROM embedding_vectors WHERE NOT EXISTS
                (SELECT 1 FROM indexed_documents d WHERE d.path=embedding_vectors.path)"""
            )


def routed_search(vault, query, scopes, limit, *, project, collection, broad, mode):
    if not query.strip():
        return []
    policy = {"scopes": scopes, "project": project, "collection": collection, "broad": broad}
    deadline = time.monotonic() + 60
    with vault.connect() as db:
        db.set_progress_handler(lambda: int(time.monotonic() >= deadline), 1000)
        visible, visible_values = _visibility(db, policy, "chunks.path")
        placeholders = ",".join("?" for _ in scopes)
        if db.execute(
            f"""SELECT 1 FROM chunks WHERE chunks.scope IN ({placeholders})
            AND {visible} AND NOT EXISTS (SELECT 1 FROM indexed_documents d WHERE d.path=chunks.path) LIMIT 1""",
            (*scopes, *visible_values),
        ).fetchone():
            raise ValueError(
                "Search context metadata is missing; explicitly reindex before searching"
            )
        clause, values = _documents(db, policy)
        columns = """chunks.path,title,heading,body,source,d.sha256 AS indexed_sha256,
            CASE WHEN d.project=? THEN 0 WHEN d.project IS NULL AND d.accepted=1 THEN 1 ELSE 2 END AS priority"""
        source = "FROM chunks JOIN indexed_documents d ON d.path=chunks.path"
        eligible = f"chunks.scope IN ({placeholders}) AND {clause}"
        parameters = (project, *scopes, *values)
        lexical = []
        if mode == "exact":
            rows = db.execute(
                f"""SELECT DISTINCT {columns} {source} WHERE {eligible}
                AND (instr(chunks.body,?)>0 OR instr(chunks.path,?)>0 OR instr(chunks.heading,?)>0)
                ORDER BY priority,chunks.path,chunks.rowid LIMIT ?""",
                (*parameters, query, query, query, limit),
            ).fetchall()
        else:
            if mode == "hybrid":
                words = re.findall(r"[^\W_]+", query, re.UNICODE)[:32]
                if words:
                    match = " OR ".join('"' + word + '"' for word in words)
                    lexical = [
                        dict(row)
                        for row in db.execute(
                            f"""SELECT DISTINCT {columns},bm25(chunks) AS rank
                        {source} WHERE {eligible} AND chunks MATCH ?
                        ORDER BY priority,rank,chunks.path,chunks.rowid LIMIT 20""",
                            (*parameters, match),
                        )
                    ]
            with vault.embedding() as embedder:
                if embedder is None:
                    raise EmbeddingError(
                        "Dense/hybrid retrieval is not configured; explicitly choose keyword or exact"
                    )
                query_vector = embedder.embed_query(query)
                index = VectorIndex(db, embedder.space)
                checked = set()

                def candidates():
                    consumed = 0
                    for count, candidate in enumerate(
                        db.execute(f"SELECT {columns} {source} WHERE {eligible}", parameters), 1
                    ):
                        consumed += len(candidate["body"].encode("utf-8"))
                        if (
                            count > MAX_SCAN_CHUNKS
                            or consumed > MAX_SCAN_BYTES
                            or time.monotonic() >= deadline
                        ):
                            raise EmbeddingError(
                                "Eligible dense scan exceeded its bounded resource budget"
                            )
                        key = (candidate["path"], candidate["indexed_sha256"])
                        if key not in checked:
                            vault.get(key[0], scopes, expected_sha256=key[1])
                            checked.add(key)
                        yield candidate

                dense = index.rank(query_vector, candidates(), 20 if mode == "hybrid" else limit)
                rows = reciprocal_rank_fusion(lexical, dense, limit) if mode == "hybrid" else dense
        results, seen = [], set()
        for row in rows:
            entry = dict(row)
            key = (entry["path"], entry["heading"], entry["body"])
            if key in seen:
                continue
            seen.add(key)
            note = vault.get(entry["path"], scopes, expected_sha256=entry.pop("indexed_sha256"))
            live_clause, live_values = _documents(db, policy)
            if not db.execute(
                f"SELECT 1 FROM indexed_documents d WHERE d.path=? AND {live_clause}",
                (entry["path"], *live_values),
            ).fetchone():
                raise ValueError("Search eligibility changed during retrieval; retry")
            entry.pop("priority", None)
            entry.update(
                {key: value for key, value in note.items() if key not in {"path", "source", "text"}}
            )
            entry["retrieval_mode"] = mode
            results.append(entry)
    vault.assert_retention_ready()
    return results
