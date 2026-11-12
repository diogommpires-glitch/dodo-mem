"""Read-only native backend contracts and disposable, explicitly scoped pilot corpora."""

import hashlib
import json
from pathlib import Path, PurePosixPath

QWEN_QUERY_PREFIX = "Instruct: Retrieve relevant documents for the given query\nQuery: "
QWEN_EMBED_MODEL = "Qwen/Qwen3-Embedding-0.6B"
QWEN_DIMENSIONS = 1024


class BackendError(RuntimeError):
    pass


def verify_models(directory, models, mode):
    if mode == "keyword":
        return
    directory = Path(directory).resolve()
    roles = {"embedding", "reranker"} if mode == "rerank" else {"embedding"}
    selected = [model for model in models if model["role"] in roles]
    if {model["role"] for model in selected} != roles:
        raise BackendError(f"Missing pinned model records for {sorted(roles)}")
    for model in selected:
        path = directory / model["file"]
        if path.resolve().parent != directory or not path.is_file():
            raise BackendError(f"Pinned {model['role']} model is unavailable")
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != model["sha256"]:
            raise BackendError(f"Pinned {model['role']} model hash mismatch")


def check_native_diagnostics(backend, diagnostics):
    if backend == "qmd" and "embedding error" in diagnostics.casefold():
        raise BackendError("QMD reported an embedding failure; its results are not valid")


def verify_engine_corpus(directory, manifest):
    directory = Path(directory).absolute()
    if directory.resolve() != directory:
        raise PermissionError("A pilot engine corpus root cannot contain symlinks")
    allowed = {entry["path"] for entry in manifest["documents"]}
    actual = set()
    for path in directory.rglob("*"):
        if path.resolve() != path:
            raise PermissionError("Symlinks are not allowed in a pilot engine corpus")
        if path.is_file():
            actual.add(path.relative_to(directory).as_posix())
    if actual != allowed:
        raise PermissionError("Pilot engine corpus has missing or unregistered files")


def _relative_path(value):
    path = PurePosixPath(value.replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts or ":" in value or not path.parts:
        raise ValueError(f"Expected a relative source path, got {value!r}")
    if {part.casefold() for part in path.parts} & {"private", "archive", "inbox"}:
        raise PermissionError(f"Only explicit shared sources may enter a pilot: {value!r}")
    if path.suffix.lower() not in {".md", ".txt"}:
        raise ValueError(f"Expected a normalized Markdown/text view, got {value!r}")
    return path


def _manifest_hash(documents):
    data = json.dumps(documents, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def freeze_corpus(home, paths, destination):
    home = Path(home).resolve()
    pilot = home / ".state/backend-pilot"
    destination = Path(destination).absolute()
    if not destination.resolve().is_relative_to(pilot) or destination.resolve() == pilot:
        raise ValueError("A frozen corpus must be inside .state/backend-pilot")
    if destination.exists():
        raise FileExistsError(destination)
    paths = sorted({_relative_path(value).as_posix() for value in paths})
    if not 1 <= len(paths) <= 100:
        raise ValueError("A pilot corpus requires 1 to 100 explicitly selected sources")
    root = home / "vault"
    documents, contents = [], []
    for relative in paths:
        path = root / relative
        if path.resolve() != path or not path.resolve().is_relative_to(root):
            raise ValueError(f"Symlinks are not pilot sources: {relative}")
        with path.open("rb") as stream:
            content = stream.read(16 * 1024 * 1024 + 1)
        if len(content) > 16 * 1024 * 1024:
            raise ValueError(f"Pilot text views are limited to 16 MiB: {relative}")
        content.decode("utf-8")
        documents.append(
            {"path": relative, "sha256": hashlib.sha256(content).hexdigest(), "bytes": len(content)}
        )
        contents.append(content)
    manifest = {"version": 1, "documents": documents, "corpus_sha256": _manifest_hash(documents)}
    destination.mkdir(parents=True)
    for entry, content in zip(documents, contents, strict=True):
        target = destination / entry["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    (destination / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def verify_corpus(directory):
    directory = Path(directory).resolve()
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if (
        manifest["version"] != 1
        or _manifest_hash(manifest["documents"]) != manifest["corpus_sha256"]
    ):
        raise ValueError("Frozen manifest hash/version mismatch")
    expected = {"manifest.json"}
    for entry in manifest["documents"]:
        relative = _relative_path(entry["path"]).as_posix()
        path = directory / relative
        if path.resolve() != path or not path.resolve().is_relative_to(directory):
            raise ValueError(f"Frozen source escaped its corpus: {relative}")
        content = path.read_bytes()
        if hashlib.sha256(content).hexdigest() != entry["sha256"] or len(content) != entry["bytes"]:
            raise ValueError(f"Frozen source hash mismatch: {relative}")
        expected.add(relative)
    actual = {
        path.relative_to(directory).as_posix() for path in directory.rglob("*") if path.is_file()
    }
    if actual != expected:
        raise ValueError("Frozen corpus contains unregistered files")
    return manifest


def native_request(backend, query, mode, limit=10, reranker_enabled=False):
    if mode not in {"keyword", "dense", "hybrid", "rerank"}:
        raise ValueError(f"Unsupported search mode: {mode!r}")
    if not query.strip() or len(query) > 4000 or not 1 <= limit <= 20:
        raise ValueError("Expected a non-empty query of at most 4000 characters and limit 1 to 20")
    if backend == "basic-memory":
        if mode == "rerank" and not reranker_enabled:
            raise BackendError("Basic Memory reranker must be explicitly configured and verified")
        return "bm_search", {
            "query": query,
            "mode": {"keyword": "text", "dense": "semantic"}.get(mode, "hybrid"),
            "project": "pilot",
            "page_size": limit,
            "output_format": "json",
            "min_similarity": 0.0,
        }
    if backend != "qmd":
        raise ValueError(f"Unsupported backend: {backend!r}")
    kinds = {"keyword": ["lex"], "dense": ["vec"]}.get(mode, ["lex", "vec"])
    return "query", {
        "searches": [{"type": kind, "query": query} for kind in kinds],
        "collections": ["pilot"],
        "limit": limit,
        "candidateLimit": 20,
        "minScore": 0,
        "rerank": mode == "rerank",
    }


def validate_results(backend, payload, manifest):
    if not isinstance(payload, dict):
        raise BackendError("Backend did not return a structured result object")
    if backend == "basic-memory" and set(payload) == {"result"}:
        payload = payload["result"]
        if not isinstance(payload, dict):
            raise BackendError("Basic Memory result wrapper did not contain an object")
    if payload.get("error"):
        raise BackendError(str(payload["error"]))
    results = payload.get("results")
    if not isinstance(results, list):
        raise BackendError("Backend omitted its explicit results array")
    allowed = {entry["path"] for entry in manifest["documents"]}
    for result in results:
        if not isinstance(result, dict):
            raise BackendError("Backend returned a malformed result item")
        path = result.get("file_path") if backend == "basic-memory" else result.get("file")
        if backend == "basic-memory" and isinstance(path, str) and path not in allowed:
            path = path.removeprefix("pilot/")
        if backend == "qmd" and isinstance(path, str):
            if path.startswith("qmd://"):
                path = path.removeprefix("qmd://pilot/")
            elif path not in allowed:
                path = path.removeprefix("pilot/")
        if path not in allowed:
            raise PermissionError(
                f"Backend returned a source outside the frozen manifest: {path!r}"
            )
    return results
