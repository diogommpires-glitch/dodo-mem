"""Pinned local embeddings and a derived SQLite window-vector index."""

import hashlib
import json
import math
import re
import struct
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import httpx

from .parsing import _available_memory

PROFILE_SHA256 = "0768f6450d7e61249f55e54883cf3f9eb4095a03c532c673a5c4607469622a94"
# The port change leaves the qualified model and its existing vectors in the same space.
VECTOR_PROFILE_SHA256 = "18719db1d84e199ee358c68063c6b89938a6d260e667cc65445d5d30c981c377"
WINDOW_CHARACTERS = 2000
HEADING_CHARACTERS = 256
MAX_WINDOWS = 8192
MAX_RESPONSE_BYTES = 256 * 1024
MAX_ELIGIBLE_CHUNKS = 100000
MIN_AVAILABLE_BYTES = 5 * 1024**3
MAX_EMBED_SECONDS = 600


class EmbeddingError(RuntimeError):
    """Embedding failure with diagnostics that contain no source text or credentials."""


def _digest(value):
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
    ).hexdigest()


def _sha(value):
    if not isinstance(value, str) or re.fullmatch("[a-f0-9]{64}", value) is None:
        raise ValueError("Expected a lowercase SHA-256 digest")
    return value


def _path(value):
    if (
        not isinstance(value, str)
        or not value
        or "\\" in value
        or len(value) > 8192
        or PurePosixPath(value).is_absolute()
        or any(part in {"", ".", ".."} for part in value.split("/"))
    ):
        raise ValueError("Vector path must be a canonical relative document path")
    return value


def chunk_sha256(heading, body):
    if not isinstance(heading, str) or not isinstance(body, str):
        raise TypeError("Canonical chunk heading and body must be strings")
    return _digest([heading, body])


def _windows(heading, body):
    chunk_sha256(heading, body)
    if not body.strip() or len(body) > 100000:
        raise ValueError("Canonical embedding chunk must contain 1 to 100000 characters")
    prefix = heading[:HEADING_CHARACTERS] + "\n\n" if heading else ""
    return tuple(
        prefix + body[offset : offset + WINDOW_CHARACTERS]
        for offset in range(0, len(body), WINDOW_CHARACTERS)
    )


def _vector(values, dimensions):
    if not isinstance(values, (list, tuple)) or len(values) != dimensions:
        raise EmbeddingError("Local embedding has an unexpected dimension")
    if any(type(value) not in {float, int} or not math.isfinite(value) for value in values):
        raise EmbeddingError("Local embedding must contain only finite numbers")
    norm = math.sqrt(math.fsum(value * value for value in values))
    if not math.isfinite(norm) or abs(norm - 1) > 0.02:
        raise EmbeddingError("Local embedding is not L2-normalised")
    return tuple(value / norm for value in values)


def _reject_constant(_):
    raise ValueError("Non-finite JSON number")


@dataclass(frozen=True)
class EmbeddingSpace:
    space_id: str
    dimensions: int


@dataclass(frozen=True)
class QueryVector:
    space_id: str
    values: tuple[float, ...]


@dataclass(frozen=True)
class WindowEmbedding:
    window_index: int
    window_sha256: str
    values: tuple[float, ...]


@dataclass(frozen=True)
class EmbeddedChunk:
    space_id: str
    heading: str
    body: str
    chunk_sha256: str
    windows: tuple[WindowEmbedding, ...]


def make_runtime_check(home):
    """Verify the qualified Windows service before credentials, then guard every request."""
    from scripts.start_embeddings import prepare, process_identity
    from scripts.start_q8 import listener_pid

    home = Path(home).absolute()
    profile = prepare(home)
    if (
        hashlib.sha256((home / "profiles/qwen3-embedding-cpu.json").read_bytes()).hexdigest()
        != PROFILE_SHA256
    ):
        raise ValueError("Pinned embedding profile SHA-256 changed")
    assets = [
        Path(profile["runtime"]["directory"]) / name for name in profile["runtime"]["files"]
    ] + [Path(profile["artifact"]["path"])]

    def signature(path):
        if path.resolve(strict=True) != path.absolute():
            raise ValueError("Pinned embedding asset contains a redirect")
        stat = path.stat()
        return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns

    signatures = [signature(path) for path in assets]
    marker_path = home / ".state/embedding-server-active.json"

    def marker():
        if marker_path.resolve(strict=True) != marker_path:
            raise ValueError("Embedding ownership record contains a redirect")
        with marker_path.open("rb") as stream:
            content = stream.read(16385)
        if len(content) > 16384:
            raise ValueError("Embedding ownership record exceeds its byte limit")
        return json.loads(content, parse_constant=_reject_constant)

    try:
        expected = marker()
    except OSError as error:
        raise EmbeddingError("Pinned embedding ownership record is unavailable") from error
    if (
        not isinstance(expected, dict)
        or expected.get("profile_sha256") != PROFILE_SHA256
        or expected.get("port") != profile["port"]
        or expected.get("alias") != profile["alias"]
        or re.fullmatch("[a-f0-9]{32}", expected.get("launch_nonce", "")) is None
    ):
        raise ValueError("Embedding ownership record does not match the qualified profile")

    def check():
        try:
            identity = process_identity(listener_pid(profile["port"]))
        except (OSError, RuntimeError) as error:
            raise EmbeddingError("Pinned embedding listener ownership is unavailable") from error
        if (
            marker() != expected
            or hashlib.sha256((home / "profiles/qwen3-embedding-cpu.json").read_bytes()).hexdigest()
            != PROFILE_SHA256
            or [signature(path) for path in assets] != signatures
            or identity != expected["identity"]
        ):
            raise EmbeddingError("Pinned local embedding service ownership or assets changed")

    check()
    return check


class LocalEmbedder:
    """Use only the approved loopback profile; the caller proves live listener ownership."""

    def __init__(
        self,
        profile_path,
        token_path,
        *,
        runtime_check,
        transport=None,
        available_memory=_available_memory,
    ):
        content = Path(profile_path).read_bytes()
        if hashlib.sha256(content).hexdigest() != PROFILE_SHA256:
            raise ValueError("Pinned embedding profile SHA-256 changed")
        profile = json.loads(content)
        if not callable(runtime_check):
            raise TypeError("An embedding listener/runtime ownership check is required")
        if (
            profile["dimensions"] != 1024
            or profile["pooling"] != "last"
            or profile["normalization"] != "l2"
            or profile["port"] != 18092
        ):
            raise ValueError("Embedding profile is outside the qualified model space")
        self.space = EmbeddingSpace(
            _digest(
                {
                    "schema_version": 1,
                    "profile_sha256": VECTOR_PROFILE_SHA256,
                    "window_characters": WINDOW_CHARACTERS,
                    "heading_characters": HEADING_CHARACTERS,
                    "window_policy": "contiguous-unicode-heading-prefix-v1",
                    "chunk_aggregation": "maximum-cosine",
                    "storage": "little-endian-float32-l2",
                    "special_tokens": "tokenize-add-special-parse-plain-send-exact-token-ids-v1",
                }
            ),
            profile["dimensions"],
        )
        runtime_check()
        token_path = Path(token_path).absolute()
        if token_path.resolve(strict=True) != token_path:
            raise ValueError("Embedding credential path cannot contain a redirect")
        with token_path.open("rb") as stream:
            credential = stream.read(1025)
        if len(credential) > 1024:
            raise ValueError("Dedicated local embedding credential exceeds its byte limit")
        try:
            token = credential.decode("ascii").strip()
        except UnicodeDecodeError:
            raise ValueError("Dedicated local embedding credential must be ASCII") from None
        if re.fullmatch("[A-Za-z0-9_-]{32,1024}", token) is None:
            raise ValueError("Invalid dedicated local embedding credential")
        self._runtime_check = runtime_check
        self._available_memory = available_memory
        self._prefix = profile["query_prefix"]
        self._alias = profile["alias"]
        self._endpoint = f"http://127.0.0.1:{profile['port']}"
        self._http = httpx.Client(
            timeout=httpx.Timeout(60, connect=3),
            trust_env=False,
            follow_redirects=False,
            transport=transport,
            headers={"Authorization": "Bearer " + token},
        )

    def close(self):
        self._http.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def _request(self, route, payload, deadline):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise EmbeddingError("Local embedding exceeded its request deadline")
        if self._available_memory() < MIN_AVAILABLE_BYTES:
            raise EmbeddingError("Local embedding requires at least 5 GiB available RAM")
        self._runtime_check()
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise EmbeddingError("Local embedding exceeded its request deadline")
        try:
            try:
                with self._http.stream(
                    "POST",
                    self._endpoint + route,
                    json=payload,
                    timeout=httpx.Timeout(min(60, remaining), connect=min(3, remaining)),
                ) as response:
                    if response.status_code != 200:
                        raise EmbeddingError(
                            f"Local embedding HTTP status {response.status_code}; no vector accepted"
                        )
                    content = bytearray()
                    for block in response.iter_bytes():
                        if time.monotonic() >= deadline:
                            raise EmbeddingError("Local embedding exceeded its request deadline")
                        content.extend(block)
                        if len(content) > MAX_RESPONSE_BYTES:
                            raise EmbeddingError("Local embedding response exceeded its byte limit")
            except httpx.HTTPError:
                raise EmbeddingError("Local embedding HTTP request failed") from None
        finally:
            self._runtime_check()
        if self._available_memory() < MIN_AVAILABLE_BYTES:
            raise EmbeddingError("Local embedding exhausted its 5 GiB RAM floor")
        try:
            result = json.loads(content, parse_constant=_reject_constant)
        except ValueError:
            raise EmbeddingError("Local embedding returned invalid JSON") from None
        if not isinstance(result, dict):
            raise EmbeddingError("Local embedding returned an unexpected response")
        return result

    def _embed(self, text, deadline):
        if not isinstance(text, str) or not text.strip() or len(text.encode("utf-8")) > 32768:
            raise EmbeddingError("Embedding text is empty or exceeds its UTF-8 byte limit")
        tokens = self._request(
            "/tokenize", {"content": text, "add_special": True, "parse_special": False}, deadline
        ).get("tokens")
        if (
            not isinstance(tokens, list)
            or not tokens
            or any(type(token) is not int or token < 0 for token in tokens)
            or len(tokens) > 8190
        ):
            raise EmbeddingError("Local embedding token count exceeds the 8192-token window")
        response = self._request(
            "/v1/embeddings",
            {
                "model": self._alias,
                "input": tokens,
                "encoding_format": "float",
                "truncate": False,
            },
            deadline,
        )
        entries = response.get("data")
        usage = response.get("usage")
        if (
            response.get("model") != self._alias
            or not isinstance(entries, list)
            or len(entries) != 1
            or not isinstance(entries[0], dict)
            or entries[0].get("index") != 0
        ):
            raise EmbeddingError("Local embedding model or response identity changed")
        if (
            not isinstance(usage, dict)
            or type(usage.get("prompt_tokens")) is not int
            or usage["prompt_tokens"] != len(tokens)
        ):
            raise EmbeddingError("Local embedding token usage differs; truncation was not accepted")
        return _vector(entries[0].get("embedding"), self.space.dimensions)

    def embed_query(self, query):
        if not isinstance(query, str) or not query.strip() or len(query) > 4000:
            raise ValueError("Dense query must contain 1 to 4000 characters")
        return QueryVector(
            self.space.space_id, self._embed(self._prefix + query, time.monotonic() + 60)
        )

    def embed_chunks(self, chunks, *, cancelled=lambda: False):
        result = []
        planned = []
        count = 0
        for chunk in chunks:
            heading, body = chunk["heading"], chunk["body"]
            windows = _windows(heading, body)
            count += len(windows)
            if count > MAX_WINDOWS:
                raise EmbeddingError("Document exceeds the 8192-window embedding budget")
            planned.append((heading, body, windows))
        deadline = time.monotonic() + MAX_EMBED_SECONDS
        for heading, body, windows in planned:
            vectors = []
            for index, text in enumerate(windows):
                if cancelled():
                    raise EmbeddingError("Document embedding cancelled; no partial result accepted")
                vectors.append(
                    WindowEmbedding(
                        index,
                        hashlib.sha256(text.encode()).hexdigest(),
                        self._embed(text, deadline),
                    )
                )
            result.append(
                EmbeddedChunk(
                    self.space.space_id,
                    heading,
                    body,
                    chunk_sha256(heading, body),
                    tuple(vectors),
                )
            )
        return result


class VectorIndex:
    """Derived vectors only; the caller supplies already authorised canonical rows."""

    def __init__(self, db, space):
        self.db = db
        self.space = space
        _sha(space.space_id)
        if space.dimensions != 1024:
            raise ValueError("Vector index requires the qualified 1024-dimensional space")
        db.execute("""CREATE TABLE IF NOT EXISTS embedding_vectors (
            space_id TEXT NOT NULL,path TEXT NOT NULL,document_sha256 TEXT NOT NULL,
            chunk_sha256 TEXT NOT NULL,window_index INTEGER NOT NULL,
            window_sha256 TEXT NOT NULL,vector BLOB NOT NULL,
            PRIMARY KEY(space_id,path,document_sha256,chunk_sha256,window_index))""")
        db.execute("CREATE INDEX IF NOT EXISTS embedding_vectors_path ON embedding_vectors(path)")

    def pack(self, values):
        return struct.pack(f"<{self.space.dimensions}f", *_vector(values, self.space.dimensions))

    def _unpack(self, blob):
        if not isinstance(blob, bytes) or len(blob) != self.space.dimensions * 4:
            raise EmbeddingError("Stored vector has an invalid float32 shape")
        return _vector(struct.unpack(f"<{self.space.dimensions}f", blob), self.space.dimensions)

    def replace(self, path, document_sha256, chunks, *, revalidate):
        path, document_sha256 = _path(path), _sha(document_sha256)
        if not callable(revalidate):
            raise TypeError("Canonical publication must revalidate source/view/hash eligibility")
        rows = {}
        for chunk in chunks:
            expected = _windows(chunk.heading, chunk.body)
            digest = chunk_sha256(chunk.heading, chunk.body)
            if chunk.space_id != self.space.space_id or chunk.chunk_sha256 != digest:
                raise ValueError("Prepared chunk belongs to another model space or source bytes")
            if len(chunk.windows) != len(expected):
                raise ValueError("Prepared chunk window inventory is incomplete")
            for index, (window, text) in enumerate(zip(chunk.windows, expected, strict=True)):
                window_hash = hashlib.sha256(text.encode()).hexdigest()
                if window.window_index != index or window.window_sha256 != window_hash:
                    raise ValueError("Prepared window differs from the exact canonical chunk")
                rows[(digest, index)] = (
                    self.space.space_id,
                    path,
                    document_sha256,
                    digest,
                    index,
                    window_hash,
                    self.pack(window.values),
                )
                if len(rows) > MAX_WINDOWS:
                    raise ValueError("Prepared document exceeds its vector window budget")
        self.db.execute("SAVEPOINT replace_embedding_vectors")
        try:
            revalidate()
            self.delete(path)
            self.db.executemany(
                "INSERT INTO embedding_vectors VALUES (?,?,?,?,?,?,?)", rows.values()
            )
            revalidate()
        except BaseException:
            self.db.execute("ROLLBACK TO replace_embedding_vectors")
            raise
        finally:
            self.db.execute("RELEASE replace_embedding_vectors")
        return {"path": path, "space_id": self.space.space_id, "vectors": len(rows)}

    def delete(self, path):
        self.db.execute("DELETE FROM embedding_vectors WHERE path=?", (_path(path),))

    def current(self, path, document_sha256, chunks):
        """Reuse only a complete, valid inventory bound to these exact source bytes."""
        path, document_sha256 = _path(path), _sha(document_sha256)
        expected = {}
        for chunk in chunks:
            digest = chunk_sha256(chunk["heading"], chunk["body"])
            for index, text in enumerate(_windows(chunk["heading"], chunk["body"])):
                expected[(digest, index)] = hashlib.sha256(text.encode()).hexdigest()
        stored = self.db.execute(
            """SELECT space_id,document_sha256,chunk_sha256,window_index,
            window_sha256,vector FROM embedding_vectors WHERE path=?""",
            (path,),
        ).fetchall()
        if not expected or len(stored) != len(expected):
            return False
        for record in stored:
            if (
                record[0] != self.space.space_id
                or record[1] != document_sha256
                or expected.get((record[2], record[3])) != record[4]
            ):
                return False
            try:
                self._unpack(record[5])
            except EmbeddingError:
                return False
        return True

    def rank(self, query_vector, eligible_rows, limit):
        _limit(limit)
        if query_vector.space_id != self.space.space_id:
            raise EmbeddingError("Query and document vectors belong to different model spaces")
        query = _vector(query_vector.values, self.space.dimensions)
        ranked = {}
        for count, row in enumerate(eligible_rows, 1):
            if count > MAX_ELIGIBLE_CHUNKS:
                raise EmbeddingError("Eligible dense corpus exceeds its explicit scan budget")
            candidate = dict(row)
            key = _candidate_key(candidate)
            document_sha = _sha(candidate["indexed_sha256"])
            windows = _windows(candidate["heading"], candidate["body"])
            stored = self.db.execute(
                """SELECT window_index,window_sha256,vector FROM embedding_vectors
                WHERE space_id=? AND path=? AND document_sha256=? AND chunk_sha256=?
                ORDER BY window_index""",
                (self.space.space_id, candidate["path"], document_sha, key[1]),
            ).fetchall()
            if not stored:
                raise EmbeddingError(
                    "Eligible document vectors are missing or stale; rebuild explicitly"
                )
            if len(stored) != len(windows):
                raise EmbeddingError("Stored vector window inventory is incomplete")
            scores = []
            for index, (record, text) in enumerate(zip(stored, windows, strict=True)):
                if record[0] != index or record[1] != hashlib.sha256(text.encode()).hexdigest():
                    raise EmbeddingError(
                        "Stored vector window inventory differs from canonical bytes"
                    )
                vector = self._unpack(record[2])
                scores.append(
                    math.fsum(left * right for left, right in zip(query, vector, strict=True))
                )
            candidate["dense_score"] = max(scores)
            ranked[key] = candidate
        return sorted(
            ranked.values(),
            key=lambda row: (row["priority"], -row["dense_score"], _candidate_key(row)),
        )[:limit]


def _limit(limit):
    if type(limit) is not int or not 1 <= limit <= 20:
        raise ValueError("Vector result limit must be between 1 and 20")


def _candidate_key(row):
    _path(row["path"])
    if type(row["priority"]) is not int or row["priority"] not in {0, 1, 2}:
        raise ValueError("Eligible row requires the gateway's explicit project priority")
    return row["path"], chunk_sha256(row["heading"], row["body"])


def reciprocal_rank_fusion(lexical_rows, dense_rows, limit):
    """Fuse canonical chunks while preserving project tiers before the final limit."""
    _limit(limit)
    results = {}
    for ranked in (lexical_rows, dense_rows):
        seen = set()
        for rank, row in enumerate(ranked, 1):
            row = dict(row)
            key = _candidate_key(row)
            if key in seen:
                continue
            seen.add(key)
            if key not in results:
                results[key] = {**row, "hybrid_score": 0.0}
            elif results[key]["priority"] != row["priority"]:
                raise ValueError("Hybrid candidate priority changed between retrieval channels")
            results[key]["hybrid_score"] += 1 / (60 + rank)
    return sorted(
        results.values(),
        key=lambda row: (row["priority"], -row["hybrid_score"], _candidate_key(row)),
    )[:limit]
