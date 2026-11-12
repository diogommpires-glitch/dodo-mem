import argparse
import json
import shutil
import sys
from pathlib import Path
from uuid import uuid4

from .coordinator import Coordinator
from .store import Vault, pin_core, read_config, read_intake_config, writer_lock


def main():
    parser = argparse.ArgumentParser(description="Dodo's local memory and agent coordinator")
    parser.add_argument("--home", type=Path, default=Path(__file__).resolve().parents[1])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("index")
    sub.add_parser("pin")
    sub.add_parser("doctor")
    owner = sub.add_parser(
        "owner-call", help="Call the running loopback gateway with the local owner credential"
    )
    owner.add_argument("tool")
    owner.add_argument(
        "arguments", type=Path, help="Local JSON object; explicit owner review remains required"
    )
    sub.add_parser(
        "retention-init",
        help="Offline owner maintenance: initialize the configured external revocation authority",
    )
    search = sub.add_parser("search")
    search.add_argument("query")
    search.add_argument("--private", action="store_true")
    search_scope = search.add_mutually_exclusive_group()
    search_scope.add_argument("--project")
    search_scope.add_argument("--broad", action="store_true")
    search.add_argument("--collection")
    search.add_argument("--mode", choices=["keyword", "exact", "dense", "hybrid"])
    get = sub.add_parser("get")
    get.add_argument("path")
    get.add_argument("--private", action="store_true")
    source = sub.add_parser("source")
    source.add_argument("path")
    source.add_argument("--page", type=int)
    source.add_argument("--lines", type=int, nargs=2, metavar=("START", "END"))
    source.add_argument("--messages", type=int, nargs=2, metavar=("START", "END"))
    source.add_argument("--conversation")
    source.add_argument("--branch")
    source.add_argument("--context", type=int, default=0)
    source.add_argument("--bbox", type=float, nargs=4, metavar=("LEFT", "BOTTOM", "RIGHT", "TOP"))
    source.add_argument("--private", action="store_true")
    load = sub.add_parser("import")
    load.add_argument("file", type=Path)
    load.add_argument("--kind", choices=["note", "paper", "conversation", "code"], default="note")
    load.add_argument("--private", action="store_true")
    capture = sub.add_parser(
        "capture", help="Durably capture a file before parsing; owner offline command"
    )
    capture.add_argument("file", type=Path)
    capture.add_argument(
        "--kind", choices=["note", "paper", "conversation", "code"], default="note"
    )
    capture.add_argument("--shared", action="store_true")
    capture.add_argument("--collection", default="references")
    capture.add_argument("--project")
    capture.add_argument("--source-id")
    capture.add_argument("--parser")
    capture.add_argument("--request-id")
    capture.add_argument("--observed-at")
    repository = sub.add_parser(
        "capture-repository", help="Offline owner capture from configured exact Git roots"
    )
    repository.add_argument("repository", type=Path)
    repository.add_argument("--path", action="append", dest="paths")
    repository.add_argument("--shared", action="store_true")
    repository.add_argument("--collection", default="code")
    repository.add_argument("--max-files", type=int, default=200)
    repository.add_argument("--max-bytes", type=int, default=16 * 1024**2)
    repository.add_argument("--max-file-bytes", type=int, default=1024**2)
    status = sub.add_parser("ingest-status")
    status.add_argument("job_id")
    status.add_argument("--private", action="store_true")
    ingest = sub.add_parser("ingest-run")
    ingest.add_argument("--max-jobs", type=int, default=20)
    sub.add_parser("ingest-pause")
    sub.add_parser("ingest-resume")
    cancel = sub.add_parser("ingest-cancel")
    cancel.add_argument("job_id")
    retry = sub.add_parser("ingest-retry")
    retry.add_argument("job_id")
    server = sub.add_parser("serve")
    server.add_argument("--port", type=int, default=8765)
    run = sub.add_parser("ask")
    run.add_argument("agent")
    run.add_argument("question")
    run.add_argument("--metadata-only", action="store_true")
    live = sub.add_parser("verify-live")
    live.add_argument("first")
    live.add_argument("peer")
    live.add_argument("--metadata-only", action="store_true")
    backup = sub.add_parser("backup")
    backup.add_argument("destination", type=Path)
    restore = sub.add_parser("restore")
    restore.add_argument("source", type=Path)
    restore.add_argument("destination", type=Path)
    restore.add_argument(
        "--authority", type=Path, help="Explicit current external authority for offline recovery"
    )
    restore.add_argument(
        "--backup-root",
        type=Path,
        action="append",
        help="Known external backup root; required with authority when no current policy exists",
    )
    args = parser.parse_args()
    if args.command == "owner-call":
        from .owner import call_owner

        with args.arguments.open("rb") as stream:
            content = stream.read(1024**2 + 1)
        if len(content) > 1024**2:
            parser.error("Owner arguments exceed 1 MiB")
        print(json.dumps(call_owner(args.home, args.tool, json.loads(content)), indent=2))
        return
    if args.command == "retention-init":
        from .ops import initialize_retention

        print(
            json.dumps(
                initialize_retention(args.home, actor_role="owner", scopes=("shared", "private")),
                indent=2,
            )
        )
        return
    if args.command == "restore":
        from .ops import restore_backup

        print(
            json.dumps(
                restore_backup(
                    args.source,
                    args.destination,
                    current_home=args.home,
                    authority=args.authority,
                    backup_roots=args.backup_root,
                ),
                indent=2,
            )
        )
        return
    vault = Vault(args.home)
    agents = read_config(args.home)
    if args.command == "doctor":
        print(
            json.dumps(
                {
                    "enabled_agents": list(agents),
                    "codex": shutil.which("codex"),
                    "claude": shutil.which("claude"),
                    "live_status": "not_tested",
                    "home": str(vault.home),
                },
                indent=2,
            )
        )
        return
    if args.command == "ingest-status":
        from .source_registry import SourceRegistry

        scopes = ("shared", "private") if args.private else ("shared",)
        result = SourceRegistry(vault).job(args.job_id, scopes)
    elif args.command in {"search", "get", "source"}:
        scopes = ("shared", "private", "archive", "inbox") if args.private else ("shared",)
        if args.command == "source":
            from .evidence import source_location

            print(
                json.dumps(
                    source_location(
                        vault,
                        args.path,
                        scopes,
                        args.page,
                        lines=args.lines,
                        messages=args.messages,
                        conversation=args.conversation,
                        branch=args.branch,
                        context=args.context,
                        bbox=args.bbox,
                    ),
                    indent=2,
                )
            )
            return
        result = (
            vault.search(
                args.query,
                scopes,
                project=args.project,
                collection=args.collection,
                broad=args.broad,
                mode=args.mode,
            )
            if args.command == "search"
            else vault.get(args.path, scopes)
        )
    else:
        with writer_lock(vault.home):
            if args.command == "pin":
                result = pin_core(vault)
            elif args.command == "index":
                result = vault.index()
            elif args.command == "import":
                result = vault.import_file(
                    args.file, args.kind, "private" if args.private else "shared"
                )
            elif args.command == "capture-repository":
                from .repositories import capture_repository
                from .source_registry import SourceRegistry

                intake = read_intake_config(vault.home)
                registry = SourceRegistry(
                    vault, intake["allowed_roots"], repository_roots=intake["repository_roots"]
                )
                result = capture_repository(
                    registry,
                    args.repository,
                    paths=args.paths,
                    scope="shared" if args.shared else "private",
                    collection=args.collection,
                    max_files=args.max_files,
                    max_bytes=args.max_bytes,
                    max_file_bytes=args.max_file_bytes,
                )
            elif args.command.startswith("ingest-") or args.command == "capture":
                from .ingest import Ingestor, configure_parsers
                from .source_registry import SourceRegistry

                registry = SourceRegistry(
                    vault,
                    [args.file.resolve(strict=True).parent] if args.command == "capture" else [],
                )
                if args.command == "capture":
                    result = registry.submit(
                        args.file,
                        kind=args.kind,
                        scope="shared" if args.shared else "private",
                        collection=args.collection,
                        project=args.project,
                        source_id=args.source_id,
                        parser=args.parser,
                        request_id=args.request_id,
                        observed_at=args.observed_at,
                    )
                elif args.command == "ingest-run":
                    if not 1 <= args.max_jobs <= 1000:
                        parser.error("--max-jobs must be between 1 and 1000")
                    configure_parsers(vault.home)
                    registry.recover()
                    worker, completed = Ingestor(vault, registry), 0
                    while completed < args.max_jobs and worker.run_next():
                        completed += 1
                    result = {
                        "processed": completed,
                        "note": "Check individual jobs for indexed/failed/cancelled state",
                    }
                elif args.command in {"ingest-cancel", "ingest-retry"}:
                    operation = (
                        registry.cancel if args.command == "ingest-cancel" else registry.retry
                    )
                    operation(args.job_id, ("shared", "private"))
                    result = registry.job(args.job_id, ("shared", "private"))
                else:
                    paused = args.command == "ingest-pause"
                    registry.pause(paused)
                    result = {"paused": paused}
            elif args.command == "backup":
                from .ops import snapshot_backup

                result = snapshot_backup(vault, args.destination)
            elif args.command == "serve":
                from .server import serve
                from .source_registry import atomic_bytes

                # Serving existing verified views must not require a model or rebuild vectors.
                atomic_bytes(
                    vault.state / "gateway-startup.json",
                    json.dumps(
                        {
                            "catalogue": "existing verified views",
                            "retrieval_default": vault.default_search_mode,
                            "vector_runtime": "not checked at startup",
                            "keyword_search": "available",
                            "indexing": "explicit index command; no startup rebuild",
                        },
                        indent=2,
                    ).encode(),
                )
                serve(vault, agents, args.port)
                return
            else:
                vault.index()
                coordinator = Coordinator(vault, agents)
                coordinator.recover()
                if args.command == "verify-live":
                    if args.first == args.peer or args.peer not in agents:
                        parser.error("verify-live needs two distinct enabled agents")
                    marker = "HANDOFF_" + uuid4().hex[:12]
                    question = (
                        f"Live handoff check. First delegate to {args.peer}, asking it to return "
                        "the private verification_challenge supplied to it by the coordinator "
                        "with delegate null. You have not received that challenge. After receiving that actual "
                        "peer reply, finish with an answer containing the marker and no delegation."
                    )
                else:
                    question = args.question
                job = coordinator.submit(
                    args.first if args.command == "verify-live" else args.agent, question
                )
                while coordinator.get(job["id"])["status"] == "queued":
                    coordinator.run_next(
                        job_id=job["id"],
                        peer_challenge=(args.peer, marker)
                        if args.command == "verify-live"
                        else None,
                    )
                result = coordinator.get(job["id"])
                if args.command == "verify-live":
                    sequence = [turn["agent"] for turn in result["transcript"]]
                    passed = (
                        result["status"] == "done"
                        and sequence == [args.first, args.peer, args.first]
                        and marker in result["transcript"][1]["answer"]
                        and marker in result["answer"]
                    )
                    result["live_handoff_passed"] = passed
                    result["evidence_note"] = (
                        "Real configured providers only; no fixture or cloud fallback"
                    )
                    (vault.state / "live-check.json").write_text(
                        json.dumps(result, indent=2), encoding="utf-8"
                    )
                records = vault.state / "results"
                records.mkdir(exist_ok=True)
                record = records / (job["id"] + ".json")
                record.write_text(
                    json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                if args.metadata_only:
                    result = {key: result[key] for key in ["id", "status", "agent", "error"]}
                    result["record"] = str(record)
                    if args.command == "verify-live":
                        result["live_handoff_passed"] = passed
                print(json.dumps(result, ensure_ascii=False, indent=2))
                if result["status"] != "done" or args.command == "verify-live" and not passed:
                    sys.exit(1)
                return
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
