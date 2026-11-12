"""Parse captured versions without requiring the caller's source file to survive."""

import hashlib
import json
import os
import re
import sqlite3
import tempfile
from pathlib import Path, PurePosixPath

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from .source_registry import NATIVE_PARSERS, SCOPES, atomic_bytes
from .store import now


def configure_parsers(home):
    path = Path(home) / "parsers.local.json"
    if not path.exists():
        return None
    config = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(config, dict) and set(config) == {"offline_docling"}:
        descriptor = config["offline_docling"]
        names = {"runtime_root", "distribution", "runtime_manifest_sha256", "model_manifest_sha256"}
        if not isinstance(descriptor, dict) or set(descriptor) != names:
            raise ValueError(
                "Offline parser configuration requires exact runtime/model pins and Linux runtime paths"
            )
        root = descriptor["runtime_root"]
        if (
            not isinstance(root, str)
            or not PurePosixPath(root).is_absolute()
            or PurePosixPath(root).as_posix() != root
            or ".." in PurePosixPath(root).parts
            or any(ord(character) < 32 for character in root)
        ):
            raise ValueError("Offline parser runtime_root must be a canonical absolute Linux path")
        if not isinstance(descriptor["distribution"], str) or not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", descriptor["distribution"]
        ):
            raise ValueError("Offline parser distribution must be an explicit local WSL name")
        if any(
            not isinstance(descriptor[name], str)
            or not re.fullmatch(r"[0-9a-f]{64}", descriptor[name])
            for name in ("runtime_manifest_sha256", "model_manifest_sha256")
        ):
            raise ValueError("Offline parser runtime/model pins must be exact SHA-256 hashes")
        return config
    names = {"docling_python": "DODO_DOCLING_PYTHON", "docling_models": "DODO_DOCLING_MODELS"}
    if not isinstance(config, dict) or set(config) != set(names):
        raise ValueError("parsers.local.json requires docling_python and docling_models only")
    for name, variable in names.items():
        value = config[name]
        if not isinstance(value, str) or not Path(value).is_absolute():
            raise ValueError(f"Parser configuration {name} must be an absolute local path")
        resolved = Path(value).resolve(strict=True)
        os.environ.setdefault(variable, str(resolved))
    return None


class Ingestor:
    def __init__(self, vault, registry):
        self.vault = vault
        self.registry = registry
        self.parser_config = configure_parsers(vault.home)

    def _cancelled(self, job_id):
        return bool(self.registry.job(job_id, SCOPES)["cancel_requested"])

    def _parse(self, original, job):
        parser = job["parser"]
        if parser in {"text-v1", "text-v2"}:
            if job["suffix"] not in {".md", ".txt"}:
                raise ValueError("Text parsers require captured Markdown or text")
            return original.read_text(encoding="utf-8"), {}
        if parser == "code-text-v1":
            from .repositories import decode_code

            provenance = job.get("repository_provenance")
            if job["kind"] != "code" or not isinstance(provenance, dict):
                raise ValueError("Code parsing requires captured repository provenance")
            content = original.read_bytes()
            text = decode_code(content, job["suffix"])
            if hashlib.sha256(content).hexdigest() != provenance["content_sha256"]:
                raise ValueError("Code original differs from its captured repository provenance")
            longest = max((len(run) for run in re.findall(chr(96) + "+", text)), default=0)
            fence = chr(96) * max(3, longest + 1)
            return (
                f"{fence}{job['suffix'][1:]}\n{text}\n{fence}",
                {
                    "repository-provenance.json": json.dumps(provenance, sort_keys=True).encode(
                        "utf-8"
                    )
                },
            )
        if parser == "pypdf-v1":
            if job["suffix"] != ".pdf":
                raise ValueError("PDF parsers require a captured PDF")
            pages = [page.extract_text() or "" for page in PdfReader(original).pages]
            text = "\n\n".join(f"## Page {i}\n\n{page}" for i, page in enumerate(pages, 1))
            if not any(page.strip() for page in pages):
                raise ValueError("PDF has no extracted text; select an OCR parser")
            return text, {}
        if parser == "pdf-annotations-v1":
            from .annotations import parse_pdf_annotations

            parsed = parse_pdf_annotations(original, cancelled=lambda: self._cancelled(job["id"]))
            if parsed.metadata["source_sha256"] != job["sha256"]:
                raise ValueError("Annotation parser read a different retained source version")
        elif parser in NATIVE_PARSERS:
            if job["scope"] != "private" or job["kind"] != "conversation":
                raise PermissionError(
                    "Native transcript parsing requires private conversation capture"
                )
            from .transcripts import parse_transcript

            parsed = parse_transcript(original, parser)
        else:
            try:
                from .parsing import DOCLING_PARSER, parse_document
            except ImportError as error:
                raise RuntimeError("Selected parser is not installed") from error

            with tempfile.TemporaryDirectory(prefix="parse-", dir=self.vault.state) as temporary:
                work_dir = Path(temporary) / "result"
                try:
                    if (
                        parser == DOCLING_PARSER
                        and self.parser_config
                        and "offline_docling" in self.parser_config
                    ):
                        from .offline_docling import parse_offline_document

                        parsed = parse_offline_document(
                            original,
                            work_dir=work_dir,
                            cancelled=lambda: self._cancelled(job["id"]),
                            **self.parser_config["offline_docling"],
                        )
                    else:
                        parsed = parse_document(
                            original,
                            parser=parser,
                            work_dir=work_dir,
                            cancelled=lambda: self._cancelled(job["id"]),
                        )
                finally:
                    log = work_dir / "conversion.log"
                    if log.is_file():
                        with log.open("rb") as stream:
                            stream.seek(max(0, log.stat().st_size - 1024**2))
                            atomic_bytes(
                                self.vault.state / "ingest-logs" / (job["id"] + ".log"),
                                stream.read(),
                            )
        artifacts = {
            **parsed.artifacts,
            "parse-metadata.json": json.dumps(parsed.metadata).encode("utf-8"),
        }
        return parsed.text, artifacts

    def run_next(self, job_id=None):
        job = self.registry.claim(job_id)
        if job is None:
            return False
        try:
            original = self.vault.home / job["original"]
            if original.is_symlink() or not original.resolve().is_relative_to(
                self.vault.home / "originals"
            ):
                raise ValueError("Captured originals must stay inside their retained directory")
            if hashlib.sha256(original.read_bytes()).hexdigest() != job["sha256"]:
                raise ValueError("Captured original failed its SHA-256 integrity check")
            from .resources import admit_storage

            with self.vault.connect() as db:
                db.execute("BEGIN IMMEDIATE")
                admit_storage(
                    self.vault.home,
                    0,
                    additional_entries=0,
                    temporary_bytes=2 * 1024**3,
                    temporary_entries=2000,
                    require_memory=True,
                )
            text, artifacts = self._parse(original, job)
            if not text.strip():
                raise ValueError("Parser produced no source text")
            self._publish(job, text, artifacts)
        except (
            OSError,
            UnicodeError,
            ValueError,
            PdfReadError,
            RuntimeError,
            MemoryError,
            sqlite3.DatabaseError,
        ) as error:
            # Job failure is durable; the original remains available for explicit diagnosis/reparse.
            from .resources import ResourceRefused

            reason = (
                "ResourceRefused:" + error.code
                if isinstance(error, ResourceRefused)
                else type(error).__name__
            )
            self.registry.finish(job["id"], error=reason)
        return True

    def _publish(self, job, text, artifacts):
        prefix = "private" if job["scope"] == "private" else "documents"
        relative = f"{prefix}/{job['source_id']}/{job['sha256']}/{job['parser']}.md"
        optional = "".join(
            f"{name}: {job[name]}\n" for name in ("project", "observed_at") if job[name] is not None
        )
        if "parse-metadata.json" in artifacts:
            extraction = json.loads(artifacts["parse-metadata.json"])
            for key, source_key in (
                ("extraction_uncertainty", "uncertainties"),
                ("figure_caption_coverage", "figure_caption_coverage"),
            ):
                if source_key in extraction:
                    encoded = json.dumps(extraction[source_key], ensure_ascii=True, allow_nan=False)
                    if len(encoded) > 16384:
                        raise ValueError(
                            "Parser uncertainty metadata exceeds its publication bound"
                        )
                    optional += f"{key}: {encoded}\n"
        record_kind = job["kind"]
        if job["parser"] == "pdf-annotations-v1":
            from .annotations import CITATION_LABEL

            record_kind = "pdf_annotation"
            optional += f"text_origin: pdf_annotation\ncitation_label: {CITATION_LABEL}\n"
        body = (
            f"---\nkind: {record_kind}\nstatus: reference\nsource: sha256:{job['sha256']}\n"
            f"source_id: {job['source_id']}\nsource_version: {job['sha256']}\n"
            f"ingest_job_id: {job['id']}\nrecord_id: {job['id']}\n"
            f"content_hash: {job['sha256']}\nschema_version: 1\n"
            f"parser: {job['parser']}\ncollection: {job['collection']}\n"
            f"recorded: {job['recorded_at']}\nassertion_type: {job['assertion_type']}\n{optional}"
            f"---\n# {Path(job['origin']).name}\n\n{text}\n"
        )
        with self.vault.connect() as db:
            # Cancellation and publication share one transaction, so a cancelled job is never indexed.
            db.execute("BEGIN IMMEDIATE")
            current = self.registry._job(db, job["id"], SCOPES)
            if current["cancel_requested"]:
                db.execute(
                    "UPDATE ingest_jobs SET status='cancelled',updated_at=? WHERE id=?",
                    (now(), job["id"]),
                )
                return
            from .resources import admit_storage

            encoded_bytes = len(body.encode("utf-8"))
            artifact_bytes = sum(len(content) for content in artifacts.values())
            admit_storage(
                self.vault.home,
                artifact_bytes + encoded_bytes * 8 + 8 * 1024**2,
                additional_entries=len(artifacts) * 4 + 12,
                temporary_bytes=artifact_bytes + encoded_bytes,
                temporary_entries=8,
                require_memory=True,
            )
            for name, content in artifacts.items():
                if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", name) or not isinstance(
                    content, bytes
                ):
                    raise ValueError("Parser artifacts require plain filenames and byte content")
                target = (
                    self.vault.home
                    / "derived"
                    / job["source_id"]
                    / job["sha256"]
                    / job["parser"]
                    / name
                )
                atomic_bytes(target, content)
            self.registry.register_artifacts(job, artifacts, db=db)
            atomic_bytes(self.vault.resolve(relative), body.encode("utf-8"))
            db.execute(
                """UPDATE ingest_jobs SET status='indexed',path=?,view_sha256=?,error=NULL,updated_at=?
                WHERE id=?""",
                (relative, hashlib.sha256(body.encode("utf-8")).hexdigest(), now(), job["id"]),
            )
            self.vault.upsert_document(relative, db=db)
