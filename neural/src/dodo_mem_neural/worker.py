"""Bounded read-only catalogue ingestion into versioned local neural state."""

import hashlib
import json
import re
import sqlite3
import stat
import time
from bisect import bisect_right
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

from .catalogue_changes import defer_sqlite_busy, open_changes
from .chat_adapters import HistoricalSourceError, parse_json_chat
from .organisation import refresh_neighbours
from .sqlite_history import ClineHistory, discover_cline_databases
from .vector_store import VectorStore

CHUNKER_VERSION = "tokens224-overlap32-retokenise-v2"
CENTROID_POLICY = "file-centroid-unique-overlap-corrected-v2"
_SECRETS = re.compile(
    rb"-----BEGIN [A-Z ]*PRIVATE KEY-----|\b(?:AKIA[A-Z0-9]{16}|gh[pousr]_[A-Za-z0-9]{20,}"
    rb"|sk-(?:proj-)?[A-Za-z0-9_-]{25,})\b|(?i:\bBearer\s+[A-Za-z0-9._~+/-]{24,}={0,2})|"
    rb"(?im:\b[A-Za-z_]*(?:auth_?token|api_?key|access_?token|password)[\"']?\s*[:=]\s*[\"'][^\r\n\"']{8,}[\"'])"
)


def chunk_text(text, tokenizer, *, start_line=1, end_line=None, locator="", metadata=None):
    """Yield at most 224 body tokens with 32 tokens of overlap, without runtime padding."""
    if tokenizer.padding is not None or tokenizer.truncation is not None:
        tokenizer = Tokenizer.from_str(tokenizer.to_str())
        tokenizer.no_padding()
        tokenizer.no_truncation()
    yield from _chunk_text_unpadded(
        text,
        tokenizer,
        start_line=start_line,
        end_line=end_line,
        locator=locator,
        metadata=metadata,
    )


def _chunk_text_unpadded(
    text, tokenizer, *, start_line=1, end_line=None, locator="", metadata=None
):
    encoding = tokenizer.encode(text, add_special_tokens=False)
    line_ends = [index for index, char in enumerate(text) if char == "\n"]
    text_digest = hashlib.sha256(text.encode()).hexdigest()
    token_start = 0
    while token_start < len(encoding.ids):
        token_end = min(token_start + 224, len(encoding.ids))
        char_start = encoding.offsets[token_start][0]
        char_end = encoding.offsets[token_end - 1][1]
        body = text[char_start:char_end].strip()
        # A continuation wordpiece can split into more tokens at a chunk's new beginning.
        while len(tokenizer.encode(body, add_special_tokens=False).ids) > 224:
            token_end -= 1
            char_end = encoding.offsets[token_end - 1][1]
            body = text[char_start:char_end].strip()
        if body:
            first = start_line + bisect_right(line_ends, char_start - 1)
            last = start_line + bisect_right(line_ends, char_end - 1)
            yield {
                **(metadata or {}),
                "text": body,
                "start": min(first, end_line) if end_line is not None else first,
                "end": min(last, end_line) if end_line is not None else last,
                "token_start": token_start,
                "token_end": token_end,
                "locator": json.dumps(
                    [locator, text_digest, token_start, token_end], separators=(",", ":")
                ),
            }
        if token_end == len(encoding.ids):
            break
        token_start = max(token_start + 1, token_end - 32)


def _chunk_weight(row):
    return max(1, row["token_end"] - row["token_start"] - (32 if row["token_start"] else 0))


class NeuralWorker:
    def __init__(
        self,
        catalogue_db,
        store,
        vectors,
        embedder,
        tokenizer,
        *,
        scope_resolver=None,
        policy_version="whole-catalogue-private-bearer-v2",
        extractor_version="catalogue-v1",
        resource_guard=None,
        file_vectors=None,
        blocked_roots=(),
        source_providers=None,
        progress_callback=None,
    ):
        self.catalogue_path = Path(catalogue_db).absolute()
        self.catalogue = sqlite3.connect(
            self.catalogue_path.as_uri() + "?mode=ro", uri=True, timeout=5
        )
        self.catalogue.row_factory = sqlite3.Row
        self.changes = None
        try:
            self.changes = open_changes(self.catalogue_path)
            self.store, self.vectors = store, vectors
            if store.encoder_id != vectors.encoder_id:
                raise ValueError("State and vector store encoder versions must match")
            if (
                getattr(store, "derived_dir", None) is not None
                and store.cache_uuid != vectors.cache_uuid
            ):
                raise ValueError("Neural metadata and vectors must bind the same cache UUID.")
            with store.db:
                if store.get_meta("centroid_policy") != CENTROID_POLICY:
                    store.db.execute("UPDATE sources SET ready=0")
                    store.db.execute(
                        "UPDATE jobs SET status='pending',priority=-10 WHERE status<>'obsolete'"
                    )
                    store.set_meta("scan_cursor", "")
                    store.set_meta("centroid_policy", CENTROID_POLICY)
                    store.reset_progress("Unique-passage centroid policy changed.")
            self.embedder = embedder
            self.tokenizer = Tokenizer.from_str(tokenizer.to_str())
            self.tokenizer.no_padding()
            self.tokenizer.no_truncation()
            self.scope_resolver = scope_resolver
            self.policy_version, self.extractor_version = policy_version, extractor_version
            self.resource_guard = resource_guard
            self.file_vectors = file_vectors or VectorStore(
                vectors.cache_dir / "files",
                vectors.state_dir / "files",
                encoder_id=store.encoder_id + ":file-weighted-v1",
                resource_guard=resource_guard,
                cache_uuid=getattr(store, "cache_uuid", ""),
            )
            self.blocked_roots = [
                Path(root).absolute()
                for root in [vectors.cache_dir, vectors.state_dir, store.path, *blocked_roots]
            ]
            self.source_providers = (
                list(source_providers)
                if source_providers is not None
                else [
                    ClineHistory(
                        database,
                        vectors.cache_dir.parent / "chat-snapshots",
                        resource_guard=resource_guard,
                    )
                    for database in discover_cline_databases()
                ]
            )
            self.progress_callback = progress_callback
            self._passages_read = 0
        except BaseException:
            # A failed constructor returns no object for its caller to close.
            self.catalogue.close()
            if self.changes is not None:
                self.changes.close()
            raise

    def _notify(self, stage, **counts):
        if self.progress_callback:
            self.progress_callback({"stage": stage, **counts})

    def close(self):
        self.catalogue.close()
        self.changes.close()

    def _guard(self):
        if self.resource_guard:
            self.resource_guard(background=True)

    def _scope(self, node):
        if self._provider(node["id"]):
            return "private"
        return self.scope_resolver(dict(node)) if self.scope_resolver else "private"

    def _provider(self, source_id):
        return next(
            (provider for provider in self.source_providers if provider.accepts(source_id)), None
        )

    def _allowed(self, node):
        if self._scope(node) is None:
            return False
        path = Path(node["path"]).absolute()
        return not any(
            part.casefold() in {".venv", "venv", "__pycache__"} for part in path.parts
        ) and not any(path == root or path.is_relative_to(root) for root in self.blocked_roots)

    def _version(self, node):
        return hashlib.sha256(
            json.dumps(
                [
                    node["digest"],
                    node["path"],
                    node["size"],
                    node["mtime_ns"],
                    node["file_id"],
                    self._scope(node),
                    self.policy_version,
                    self.extractor_version,
                    self.store.encoder_id,
                    self.store.chunker_id,
                    CENTROID_POLICY,
                ],
                separators=(",", ":"),
            ).encode()
        ).hexdigest()

    def _node(self, source_id):
        provider = self._provider(source_id)
        if provider:
            return provider.node(source_id)
        return self.catalogue.execute(
            """SELECT n.* FROM nodes n JOIN nodes r ON r.id=n.root_id
            WHERE n.id=? AND n.kind='file' AND n.state='indexed'
            AND n.digest IS NOT NULL AND r.state='online'""",
            (source_id,),
        ).fetchone()

    def _queue_current(self, node):
        scope = self._scope(node)
        if scope is None:
            self.store.invalidate_source(node["id"])
            return
        previous = self.store.db.execute(
            "SELECT scan_epoch,provider FROM sources WHERE source_id=?", (node["id"],)
        ).fetchone()
        self.store.queue_source(
            node["id"],
            node["digest"],
            node["path"],
            scope,
            self._version(node),
            scan_epoch=int(self.store.get_meta("scan_epoch", previous[0] if previous else 0)),
            provider=previous[1] if previous else "catalogue",
        )

    @defer_sqlite_busy()
    def consume_changes(self, limit=256):
        if not 1 <= limit <= 4096:
            raise ValueError("Catalogue change batch must be in [1,4096]")
        rows = self.changes.execute(
            "SELECT node_id,revision FROM neural_changes ORDER BY changed_at,node_id LIMIT ?",
            (limit,),
        ).fetchall()
        for node_id, revision in rows:
            self._guard()
            node = self._node(node_id)
            if node is None or not self._allowed(node):
                self.store.invalidate_source(node_id)
            else:
                self._queue_current(node)
                with self.store.db:
                    self.store.db.execute(
                        """UPDATE jobs SET priority=10
                        WHERE source_id=? AND status='pending'""",
                        (node_id,),
                    )
            # A concurrent newer change increments revision and cannot be acknowledged here.
            with self.changes:
                self.changes.execute(
                    "DELETE FROM neural_changes WHERE node_id=? AND revision=?", (node_id, revision)
                )
        return len(rows)

    @defer_sqlite_busy()
    def refresh_changed(self, limit=128):
        if not 1 <= limit <= 1024:
            raise ValueError("Ready-source refresh batch must be in [1,1024]")
        self._guard()
        self._notify("refreshing")
        cursor = self.store.get_meta("refresh_cursor", "")
        rows = self.store.db.execute(
            """SELECT source_id,version FROM sources WHERE active=1 AND ready=1
            AND provider='catalogue' AND source_id>? ORDER BY source_id LIMIT ?""",
            (cursor, limit),
        ).fetchall()
        changed = unavailable = 0
        for source in rows:
            node = self._node(source["source_id"])
            if node is None or not self._allowed(node):
                self.store.invalidate_source(source["source_id"])
                unavailable += 1
            elif self._version(node) != source["version"]:
                self._queue_current(node)
                changed += 1
        with self.store.db:
            self.store.set_meta(
                "refresh_cursor", rows[-1]["source_id"] if len(rows) == limit else ""
            )
        return {"changed": changed, "unavailable": unavailable, "checked": len(rows)}

    @defer_sqlite_busy()
    def scan(self, limit=256):
        if not 1 <= limit <= 4096:
            raise ValueError("Catalogue scan batch must be in [1,4096]")
        self._guard()
        changes = self.consume_changes(limit)
        self._notify("scanning", pending=self.store.pending_count())
        external = []
        for provider in self.source_providers:
            try:
                external.append(provider.scan(self.store, self._version, limit=limit))
            except HistoricalSourceError:
                sources = self.store.db.execute(
                    "SELECT source_id FROM sources WHERE provider=?", (provider.prefix,)
                ).fetchall()
                for source in sources:
                    self.store.invalidate_source(source["source_id"])
                external.append({"queued": 0, "complete": False, "unavailable": True})
        cursor = self.store.get_meta("scan_cursor", "")
        epoch = self.store.get_meta("scan_epoch")
        if not cursor or epoch is None:
            epoch = str(time.time_ns())
        rows = self.catalogue.execute(
            """SELECT n.* FROM nodes n JOIN nodes r ON r.id=n.root_id
            WHERE n.id>? AND n.kind='file' AND n.state='indexed'
            AND n.digest IS NOT NULL AND r.state='online' ORDER BY n.id LIMIT ?""",
            (cursor, limit),
        ).fetchall()
        for node in rows:
            if not self._allowed(node):
                self.store.invalidate_source(node["id"])
                continue
            self.store.queue_source(
                node["id"],
                node["digest"],
                node["path"],
                self._scope(node),
                self._version(node),
                scan_epoch=int(epoch),
            )
        with self.store.db:
            self.store.set_meta("scan_epoch", epoch)
            if rows:
                self.store.set_meta("scan_cursor", rows[-1]["id"])
        complete = len(rows) < limit
        if complete:
            self.store.finish_scan(int(epoch))
        return {
            "queued": len(rows) + sum(result["queued"] for result in external),
            "complete": complete and all(result["complete"] for result in external),
            "cursor": rows[-1]["id"] if rows else cursor,
            "changes": changes,
            "unavailable_providers": sum(row.get("unavailable", False) for row in external),
        }

    def _chat_messages(self, node):
        path = Path(node["path"])
        parts = {part.casefold() for part in path.parts}
        adapter = None
        if path.suffix.casefold() == ".jsonl":
            if ".codex" in parts and path.name.startswith("rollout-"):
                adapter = "codex"
            elif ".claude" in parts:
                adapter = "claude"
        elif path.name in {"api_conversation_history.json", "conversations.json"}:
            adapter = "cline" if path.name.startswith("api_") else "export"
        if adapter is None:
            return None
        for parent in (*reversed(path.parents), path):
            info = parent.lstat()
            attributes = getattr(info, "st_file_attributes", 0)
            if stat.S_ISLNK(info.st_mode) or attributes & (0x400 | 0x1000 | 0x40000 | 0x400000):
                raise OSError("Historical source path became unavailable")
        if not stat.S_ISREG(info.st_mode) or info.st_size > 16 * 1024**2:
            raise ValueError("Historical chat source exceeds the regular-file 16 MiB bound")
        with path.open("rb") as stream:
            raw = stream.read(16 * 1024**2 + 1)
        if len(raw) > 16 * 1024**2:
            raise ValueError("Historical chat source exceeds the regular-file 16 MiB bound")
        if hashlib.sha256(raw).hexdigest() != node["digest"]:
            raise OSError("Historical chat changed since catalogue extraction")
        return [
            message
            for message in parse_json_chat(raw, adapter)
            if not _SECRETS.search(message.text.encode())
        ]

    def _chunks(self, node):
        provider = self._provider(node["id"])
        if provider:
            try:
                passages = iter(provider.passages(node["id"]))
            except (OSError, ValueError) as error:
                raise HistoricalSourceError("provider_source_unavailable") from error
            while True:
                try:
                    passage = next(passages)
                except StopIteration:
                    break
                except (OSError, ValueError) as error:
                    raise HistoricalSourceError("provider_source_unavailable") from error
                self._passages_read += 1
                yield from _chunk_text_unpadded(
                    passage["text"],
                    self.tokenizer,
                    start_line=passage["start"],
                    end_line=passage["end"],
                    locator=passage["locator"],
                    metadata={
                        "historical": True,
                        "role": passage["role"],
                        "timestamp": passage["timestamp"],
                    },
                )
            return
        try:
            messages = self._chat_messages(node)
        except OSError as error:
            raise HistoricalSourceError("historical_source_unavailable") from error
        except ValueError as error:
            raise HistoricalSourceError("historical_source_invalid") from error
        if messages is not None:
            for message in messages:
                self._passages_read += 1
                line = (
                    int(message.locator.split(":")[1]) if message.locator.startswith("line:") else 1
                )
                yield from _chunk_text_unpadded(
                    message.text,
                    self.tokenizer,
                    start_line=line,
                    locator=message.locator,
                    metadata={
                        "role": message.role,
                        "timestamp": message.timestamp,
                        "historical": True,
                    },
                )
            return
        passages = self.catalogue.execute(
            "SELECT kind,start,end,text FROM passages WHERE node_id=? ORDER BY start,end,id",
            (node["id"],),
        )
        for passage in passages:
            if _SECRETS.search(passage["text"].encode()):
                continue
            self._passages_read += 1
            yield from _chunk_text_unpadded(
                passage["text"],
                self.tokenizer,
                start_line=passage["start"],
                end_line=passage["end"],
                locator=json.dumps([passage["kind"], passage["start"], passage["end"]]),
                metadata={"historical": passage["kind"] == "chat"},
            )

    @defer_sqlite_busy()
    def run_pending(self, limit=32):
        if not 1 <= limit <= 4096:
            raise ValueError("Job batch must be in [1,4096]")
        completed = stale = embedded = failed = 0
        self._passages_read = 0
        chunks = 0
        prepared = []
        running = []
        try:
            for _ in range(limit):
                self._guard()
                job = self.store.claim_job()
                if job is None:
                    break
                running.append(job["id"])
                self._notify("reading", batch_jobs=len(running), batch_embedded=embedded)
                node = self._node(job["source_id"])
                if node is None or not self._allowed(node) or self._version(node) != job["version"]:
                    self.store.invalidate_source(job["source_id"])
                    if node is not None and self._allowed(node):
                        self._queue_current(node)
                    stale += 1
                    continue
                try:
                    rows = self.store.stage_chunks(node["id"], job["version"], self._chunks(node))
                except HistoricalSourceError as error:
                    self.store.fail_job(job["id"], str(error))
                    failed += 1
                    continue
                chunks += len(rows)
                self._notify(
                    "embedding",
                    batch_jobs=len(running),
                    job_chunks=len(rows),
                    batch_embedded=embedded,
                    batch_passages=self._passages_read,
                )
                unique = {}
                for row in rows:
                    previous = unique.get(row["chunk_id"])
                    if previous is None or _chunk_weight(row) > _chunk_weight(previous):
                        unique[row["chunk_id"]] = row
                keys = list(unique)
                missing = [
                    key
                    for key, present in zip(keys, self.vectors.contains(keys), strict=True)
                    if not present
                ]
                for offset in range(0, len(missing), 16):
                    self._guard()
                    batch = missing[offset : offset + 16]
                    self.vectors.add(
                        batch, self.embedder([unique[key]["text"] for key in batch], query=False)
                    )
                    embedded += len(batch)
                    self._notify(
                        "embedding",
                        batch_jobs=len(running),
                        batch_embedded=embedded,
                        batch_passages=self._passages_read,
                    )
                if rows:
                    centroid = np.zeros(384, dtype=np.float64)
                    weight = 0
                    passages = list(unique.values())
                    for offset in range(0, len(passages), 128):
                        batch = passages[offset : offset + 128]
                        values = self.vectors.get([row["chunk_id"] for row in batch])
                        weights = np.asarray(
                            [_chunk_weight(row) for row in batch],
                            dtype=np.float64,
                        )
                        centroid += np.sum(values * weights[:, None], axis=0)
                        weight += float(np.sum(weights))
                    key = self.store.save_centroid(node["id"], job["version"], centroid, weight)
                    self.file_vectors.add([key], np.asarray([centroid], dtype=np.float32))
                prepared.append(job)
            if prepared:
                self._notify("publishing", batch_jobs=len(prepared), batch_embedded=embedded)
                self.vectors.publish()
                self.file_vectors.publish()
            for job in prepared:
                current = self._node(job["source_id"])
                if current is None or self._version(current) != job["version"]:
                    self.store.invalidate_source(job["source_id"])
                    if current is not None and self._allowed(current):
                        self._queue_current(current)
                    stale += 1
                elif self.store.complete_source(job["source_id"], job["version"]):
                    completed += 1
            affected = set()
            for job in prepared:
                affected.update(
                    refresh_neighbours(
                        self.store,
                        self.file_vectors,
                        job["source_id"],
                        passage_vectors=self.vectors,
                    )
                )
            updated = {job["source_id"] for job in prepared}
            for source_id in sorted(affected - updated):
                refresh_neighbours(
                    self.store, self.file_vectors, source_id, passage_vectors=self.vectors
                )
            cursor = self.store.get_meta("neighbour_cursor", "")
            sweep = self.store.centroid_rows(after=cursor, limit=min(limit, 256))
            for row in sweep:
                if row["source_id"] not in updated | affected:
                    refresh_neighbours(
                        self.store,
                        self.file_vectors,
                        row["source_id"],
                        passage_vectors=self.vectors,
                    )
            with self.store.db:
                self.store.set_meta(
                    "neighbour_cursor",
                    sweep[-1]["source_id"] if len(sweep) == min(limit, 256) else "",
                )
        except (OSError, ValueError, RuntimeError, sqlite3.Error):
            for job_id in running:
                self.store.retry_job(job_id)
            raise
        return {
            "completed": completed,
            "stale": stale,
            "embedded": embedded,
            "pending": self.store.pending_count(),
            "passages": self._passages_read,
            "chunks": chunks,
            "failed": failed,
        }
