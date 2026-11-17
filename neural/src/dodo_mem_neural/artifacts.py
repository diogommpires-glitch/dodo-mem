"""Download a bounded, pinned artifact set and verify it before loading."""

import hashlib
import json
import os
import time
from pathlib import Path

import httpx

from .resources import DEFAULT_CACHE, check_resources

CANDIDATES = {
    "e5": (
        "intfloat/multilingual-e5-small",
        "614241f622f53c4eeff9890bdc4f31cfecc418b3",
        "openvino/openvino_model",
        "MIT",
    ),
    "qwen": (
        "OpenVINO/Qwen3-Embedding-0.6B-int8-ov",
        "80d352d7061ec3e5caa3f26dc4e1f67bb8a21a5e",
        "openvino_model",
        "Apache-2.0",
    ),
    "reranker": (
        "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
        "1427fd652930e4ba29e8149678df786c240d8825",
        "openvino/openvino_model",
        "Apache-2.0",
    ),
}
SMALL_FILES = {
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "config.json",
    "README.md",
    "LICENSE",
    "LICENSE.txt",
    "vocab.txt",
    "sentencepiece.bpe.model",
    "modules.json",
}
WEIGHT_SHA256 = {
    "e5": "22f5a10ec005a1606ec42758dedd52b59dfcb6e3d85de0b6b86946d0bc6cf232",
    "qwen": "b88066b1898f5e74eb31fe87578f10ab3d01709765a09bd8c4ea4fe256f62bb4",
    "reranker": "3b00b3db7a1a0d53c64b459e81900c1dfa7fd34c253556e4cc9b862fcd356c10",
}


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".new")
    with pending.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.flush()
        os.fsync(stream.fileno())
    replace_file(pending, path)


def replace_file(source, target):
    deadline = time.monotonic() + 1
    while True:
        try:
            Path(source).replace(target)
            return
        except PermissionError:
            if time.monotonic() >= deadline:
                raise
            # Windows readers without FILE_SHARE_DELETE can briefly hold an old snapshot.
            time.sleep(0.01)


def download(name, cache=DEFAULT_CACHE):
    repo, revision, stem, licence = CANDIDATES[name]
    target = Path(cache) / "models" / name
    target.mkdir(parents=True, exist_ok=True)
    with httpx.Client(timeout=120, follow_redirects=True, trust_env=False) as client:
        response = client.get(
            f"https://huggingface.co/api/models/{repo}/tree/{revision}",
            params={"recursive": "true"},
        )
        response.raise_for_status()
        selected = [
            entry
            for entry in response.json()
            if entry["type"] == "file"
            and (entry["path"] in SMALL_FILES or entry["path"] in {stem + ".xml", stem + ".bin"})
        ]
        names = {entry["path"] for entry in selected}
        if not {stem + ".xml", stem + ".bin", "tokenizer.json"} <= names:
            raise ValueError(f"Required artifacts absent from pinned {repo}@{revision}: {names}")
        manifest = {
            "publisher": repo.split("/")[0],
            "repository": repo,
            "revision": revision,
            "licence": licence,
            "files": selected,
        }
        atomic_json(target / "download-manifest.json", manifest)
        for entry in selected:
            destination = target / entry["path"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            expected = entry.get("lfs", {}).get("oid")
            if (
                destination.exists()
                and destination.stat().st_size == entry["size"]
                and expected
                and sha256(destination) == expected
            ):
                entry["sha256"] = expected
                continue
            check_resources(additional_disk=entry["size"], cache=cache)
            partial = destination.with_suffix(destination.suffix + ".partial")
            digest = hashlib.sha256()
            git_blob = hashlib.sha1(f"blob {entry['size']}\0".encode())
            received = 0
            with (
                client.stream(
                    "GET", f"https://huggingface.co/{repo}/resolve/{revision}/" + entry["path"]
                ) as result,
                partial.open("wb") as stream,
            ):
                result.raise_for_status()
                for block in result.iter_bytes(1024 * 1024):
                    received += len(block)
                    if received > entry["size"]:
                        raise ValueError("Artifact exceeded its pinned size.")
                    digest.update(block)
                    git_blob.update(block)
                    stream.write(block)
                stream.flush()
                os.fsync(stream.fileno())
            actual = digest.hexdigest()
            if received != entry["size"] or (expected and actual != expected):
                raise ValueError(f"Artifact size/hash mismatch: {entry['path']}")
            if not expected and git_blob.hexdigest() != entry["oid"]:
                raise ValueError(f"Git blob mismatch: {entry['path']}")
            partial.replace(destination)
            entry["sha256"] = actual
            atomic_json(target / "download-manifest.json", manifest)
        verify(target)
    return target


def verify(target):
    target = Path(target)
    manifest = json.loads((target / "download-manifest.json").read_text(encoding="utf-8"))
    expected_repo, expected_revision, _, expected_licence = CANDIDATES[target.name]
    if (manifest.get("repository"), manifest.get("revision"), manifest.get("licence")) != (
        expected_repo,
        expected_revision,
        expected_licence,
    ):
        raise ValueError("Artifact manifest does not match the selected pinned candidate.")
    stem = CANDIDATES[target.name][2]
    required = {stem + ".xml", stem + ".bin", "tokenizer.json"}
    paths = [entry.get("path") for entry in manifest["files"]]
    if not required <= set(paths) or len(paths) != len(set(paths)):
        raise ValueError(
            "Artifact manifest omits required model/tokenizer files or repeats a path."
        )
    for entry in manifest["files"]:
        if entry["path"] not in SMALL_FILES | required:
            raise ValueError("Artifact path is not in the selected download set.")
        path = target / entry["path"]
        if (
            not entry.get("sha256")
            or path.stat().st_size != entry["size"]
            or (sha256(path) != entry["sha256"])
        ):
            raise ValueError(f"Unverified model artifact: {entry['path']}")
        if entry["path"] == stem + ".bin" and entry["sha256"] != WEIGHT_SHA256[target.name]:
            raise ValueError("Artifact does not match the pinned publisher weight hash.")
    return manifest
