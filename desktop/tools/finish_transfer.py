"""Record completion and optionally shut down after the desktop and catalogue pass."""

import argparse
import json
import os
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

from dodo_memory_desktop.whole_index import Catalogue


def app_qualified(receipt):
    if (
        receipt.get("qt_platform") != "windows"
        or receipt.get("shortcut_verified") is not True
        or receipt.get("same_application_at_start_and_end") is not True
        or receipt.get("launcher_probe_verified") is not True
    ):
        return False
    results = receipt["results"]
    search, graph, updates = (results[name] for name in ("search", "graph", "updates"))
    live = receipt["live_updates"]
    timing = (
        (search["first_pass"], 1000),
        (search["repeat_pass"], 300),
        (graph["wheel_input_to_paint"], 33),
        (graph["cached_expansion"], 250),
    )
    return (
        all(part["pass"] is True for part in (search, graph, updates))
        and search["filename_index_ready_during_both_passes"] is True
        and search["source_tasks"] == 12
        and 11 <= search["verified_source_tasks"] <= 12
        and all(
            result["samples"] >= 5 and 0 <= result["p95_ms"] <= target for result, target in timing
        )
        and all(
            updates[key] is True
            for key in (
                "stale_source_denied",
                "revoked_source_denied",
                "protected_source_denied",
                "reference_source_verified",
                "restart_source_verified",
            )
        )
        and all(0 <= updates[key] <= 30_000 for key in ("created_ms", "changed_ms", "removed_ms"))
        and live["pass"] is True
        and live["stale_source_denied"] is True
        and all(0 <= live[key] <= 30_000 for key in ("created_ms", "changed_ms", "removed_ms"))
    )


def catalogue_completion(catalogue):
    status = catalogue.status(summary=True)
    if (
        status["scanning"]
        or status["content_pending"]
        or status["inventory_pending"]
        or not status["inventory_finished"]
        or status["message"] != "Local catalogue ready."
    ):
        return None
    graph = catalogue.graph_projection(include_inferred=True)
    revision = status["generation"]
    if (
        graph["generation"] != revision
        or graph["counts"]["nodes"] != status["nodes"]
        or not graph["nodes"]
        or not graph["edges"]
        or catalogue.revision() != revision
    ):
        return None
    return {"catalogue": status, "graph": graph["counts"]}


def main():
    from tools.verify_live_desktop import application_digest

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("acceptance", type=Path, help="Native desktop acceptance receipt")
    parser.add_argument("--shutdown", action="store_true", help="Wait and shut down normally")
    args = parser.parse_args()
    acceptance = json.loads(args.acceptance.read_text(encoding="utf-8"))
    try:
        qualified = app_qualified(acceptance)
    except (KeyError, TypeError):
        qualified = False
    if not qualified:
        parser.exit(2, "Desktop acceptance is incomplete. Shutdown is disabled.\n")
    if acceptance.get("application_sha256") != application_digest():
        parser.exit(2, "The installed build changed after acceptance. Shutdown is disabled.\n")
    if args.shutdown and os.name != "nt":
        parser.error("Normal Windows shutdown is available on Windows only.")

    catalogue = Catalogue()
    if Path(acceptance.get("database", "")).resolve() != catalogue.database.resolve():
        parser.exit(2, "Acceptance used a different catalogue. Shutdown is disabled.\n")
    if args.shutdown:
        print("Waiting for the catalogue and graph to publish their final revision.", flush=True)
    while (completion := catalogue_completion(catalogue)) is None:
        if not args.shutdown:
            parser.exit(1, "The catalogue or its graph is still publishing.\n")
        time.sleep(30)
    if acceptance["application_sha256"] != application_digest():
        parser.exit(2, "The installed build changed while waiting. Shutdown is disabled.\n")

    receipt = args.acceptance.with_name("transfer-complete.json")
    temporary = receipt.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as output:
        json.dump(
            {
                **completion,
                "verified_at": datetime.now(UTC).isoformat(),
                "desktop_acceptance": str(args.acceptance.resolve()),
            },
            output,
            indent=2,
        )
        output.flush()
        os.fsync(output.fileno())
    temporary.replace(receipt)
    print(receipt, flush=True)
    if args.shutdown:
        shutdown = Path(os.environ["SYSTEMROOT"]) / "System32" / "shutdown.exe"
        subprocess.run(
            [str(shutdown), "/s", "/t", "0"],
            creationflags=subprocess.CREATE_NO_WINDOW,
            check=True,
        )


if __name__ == "__main__":
    main()
