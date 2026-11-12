"""Run configured owner checks before proposing an evidence-backed reusable procedure."""

import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4

from .evidence import _relative_note, source_location
from .native_process import linux_path, owned_wsl_process
from .parsing import _available_memory, _owned_process
from .reconciliation import _observe, _repository, _snapshot
from .repositories import _Cancelled, _relative
from .source_registry import single_line
from .store import now

MAX_OUTPUT_BYTES = 1024**2
CREDENTIAL_FLAG = re.compile(
    r"(?i)^--?(?:api[-_]?key|access[-_]?token|token|password|secret|credential)(?:=|$)"
)

WINDOWS_CHECK_BOOTSTRAP = r"""
import os
import sys
import time
from pathlib import Path
gate = Path(os.environ['DODO_DOCLING_START_GATE'])
deadline = time.monotonic() + 10
while not gate.is_file():
    if time.monotonic() >= deadline:
        raise TimeoutError('Owner verification launch gate was not assigned')
    time.sleep(.01)
import site
site.addsitedir(sys.argv[1])
sys.path.insert(0, sys.argv[2])
sys.argv = ['dodo_memory.milestones', '--worker', sys.argv[3]]
import runpy
runpy.run_module('dodo_memory.milestones', run_name='__main__')
"""


def _environment():
    names = {
        "PATH",
        "TEMP",
        "TMP",
        "USERPROFILE",
        "LOCALAPPDATA",
        "SystemRoot",
        "WINDIR",
        "ProgramFiles",
        "ProgramW6432",
        "OS",
        "PROCESSOR_ARCHITECTURE",
    }
    environment = {name: value for name, value in os.environ.items() if name in names}
    if os.name == "nt":
        environment.update(
            {
                "SystemRoot": "C:/Windows",
                "WINDIR": "C:/Windows",
                "ProgramFiles": "C:/Program Files",
                "ProgramW6432": "C:/Program Files",
                "OS": "Windows_NT",
                "PROCESSOR_ARCHITECTURE": "AMD64",
            }
        )
    environment["PYTHONUTF8"] = "1"
    return environment


def _resources(home):
    return _available_memory() >= 5 * 1024**3 and shutil.disk_usage(home).free >= 100 * 1024**3


def _worker(spec_path):
    # Only this trusted wrapper runs before assignment; configured commands wait behind the gate.
    gate = Path(os.environ["DODO_DOCLING_START_GATE"])
    deadline = time.monotonic() + 10
    while not gate.is_file():
        if time.monotonic() >= deadline:
            raise TimeoutError("Owner verification launch gate was not assigned")
        time.sleep(0.01)
    spec = json.loads(Path(spec_path).read_text(encoding="utf-8"))
    result = subprocess.run(
        spec["argv"], cwd=spec["repository"], env=os.environ, stdin=subprocess.DEVNULL, check=False
    )
    raise SystemExit(result.returncode)


@contextmanager
def _temporary_checks(state):
    temporary = tempfile.TemporaryDirectory(prefix="milestone-", dir=state)
    try:
        yield Path(temporary.name)
    finally:
        # Job termination can precede Windows releasing descendants' inherited log handles.
        for attempt in range(20):
            try:
                temporary.cleanup()
                break
            except PermissionError:
                if attempt == 19:
                    raise
                time.sleep(0.05)


class MilestoneProcedures:
    def __init__(self, vault, knowledge, registry, *, checks, timeout=60):
        if (
            type(timeout) not in {int, float}
            or not math.isfinite(timeout)
            or not 0 < timeout <= 120
        ):
            raise ValueError("Milestone timeout must be finite and at most 120 seconds")
        if not isinstance(checks, dict) or not 1 <= len(checks) <= 10:
            raise ValueError("Configure 1 to 10 explicit owner verification checks")
        configured = {}
        for name, descriptor in checks.items():
            single_line(name, "verification check name", 160)
            if (
                not isinstance(descriptor, dict)
                or set(descriptor) != {"argv", "shareable"}
                or type(descriptor["shareable"]) is not bool
            ):
                raise ValueError(
                    "Configured check requires exact argv and explicit shareable policy"
                )
            argv = descriptor["argv"]
            if not isinstance(argv, list) or not 1 <= len(argv) <= 64:
                raise ValueError("Configured check argv must contain 1 to 64 literal arguments")
            for value in argv:
                single_line(value, "verification argument", 4096)
                if CREDENTIAL_FLAG.match(value) or value.casefold().startswith("bearer "):
                    raise ValueError(
                        "Configured verification argv cannot contain credential arguments"
                    )
            if not Path(argv[0]).is_absolute() or sum(len(value) for value in argv) > 16384:
                raise ValueError("Verification executable must be absolute and argv bounded")
            wsl = None
            if Path(argv[0]).name.casefold() in {"wsl", "wsl.exe"}:
                expected = Path(os.environ.get("SYSTEMROOT", "C:/Windows")) / "System32/wsl.exe"
                if (
                    os.name != "nt"
                    or Path(argv[0]) != expected
                    or len(argv) < 5
                    or argv[1] != "-d"
                    or argv[3] != "--exec"
                ):
                    raise ValueError(
                        "WSL checks require exact System32/wsl.exe -d DIST --exec LinuxARGV"
                    )
                single_line(argv[2], "WSL distribution", 160)
                wsl = {"distribution": argv[2], "argv": tuple(argv[4:])}
            configured[name] = {
                "argv": tuple(argv),
                "shareable": descriptor["shareable"],
                "wsl": wsl,
            }
        self.vault, self.knowledge, self.registry = vault, knowledge, registry
        self.checks, self.timeout = configured, timeout

    def _code_evidence(self, evidence, root, scopes, deadline):
        found = False
        for reference in evidence:
            if self.vault.get(reference["path"], scopes).get("parser") != "code-text-v1":
                continue
            location = source_location(self.vault, reference["path"], scopes)
            context = location["repository"]
            if Path(context["repository"]) != root:
                raise ValueError("Milestone code evidence belongs to another configured repository")
            observed, reason, _ = _observe(root, _relative(context["relative_path"]), deadline)
            if reason or observed["sha256"] != location["source_version"]:
                raise ValueError("Milestone code evidence must match current live bytes")
            found = True
        if not found:
            raise ValueError(
                "Milestone requires at least one current registered code evidence reference"
            )

    def _run(self, name, root, temporary, deadline):
        configured = self.checks[name]
        started, reason = time.monotonic(), None
        # Anonymous/delete-on-close output avoids a durable raw log after a controller crash.
        with tempfile.TemporaryFile(mode="w+b", dir=temporary) as log:
            if configured["wsl"] is not None:
                check = configured["wsl"]
                command = ["/usr/bin/env", "-C", linux_path(root), *check["argv"]]
                process_scope = owned_wsl_process(
                    command,
                    distribution=check["distribution"],
                    directory=temporary,
                    log=log,
                    environment=_environment(),
                    stdin=subprocess.DEVNULL,
                )
            else:
                spec = temporary / "check.json"
                spec.write_text(
                    json.dumps({"argv": configured["argv"], "repository": str(root)}),
                    encoding="utf-8",
                )
                gate = temporary / "start.gate"
                gate.unlink(missing_ok=True)
                command = [sys.executable, "-m", "dodo_memory.milestones", "--worker", str(spec)]
                if os.name == "nt":
                    command = [
                        sys._base_executable,
                        "-I",
                        "-S",
                        "-c",
                        WINDOWS_CHECK_BOOTSTRAP,
                        sysconfig.get_path("purelib"),
                        str(Path(__file__).resolve().parents[1]),
                        str(spec),
                    ]
                process_scope = _owned_process(command, _environment(), log, gate)
            with process_scope as owned:
                process = owned.process if configured["wsl"] is not None else owned
                while process.poll() is None:
                    if time.monotonic() >= deadline:
                        reason = "timeout"
                    elif os.fstat(log.fileno()).st_size > MAX_OUTPUT_BYTES:
                        reason = "output_limit"
                    elif not _resources(self.vault.home):
                        reason = "resource_floor"
                    if reason:
                        break
                    time.sleep(0.02)
                exit_code = process.poll()
            size = os.fstat(log.fileno()).st_size
            if size > MAX_OUTPUT_BYTES:
                reason = "output_limit"
            digest = None
            if not reason:
                log.seek(0)
                digest = hashlib.file_digest(log, "sha256").hexdigest()
        receipt = {
            "name": name,
            "argv": list(configured["argv"]),
            "exit_code": exit_code,
            "elapsed_seconds": time.monotonic() - started,
            "output_bytes": size,
            "output_sha256": digest,
        }
        if configured["wsl"] is not None:
            receipt["linux_cleanup"] = owned.cleanup
        return receipt, reason or ("nonzero_exit" if exit_code else None)

    def propose(
        self,
        entry_key,
        *,
        repository,
        check_names,
        preconditions,
        evidence,
        expected_revision,
        writer,
        scope="private",
        scopes=("shared",),
        admin=False,
    ):
        if admin is not True:
            raise PermissionError("Milestone verification requires the owner's local credential")
        if scope not in {"shared", "private"} or scope not in scopes:
            raise PermissionError("Milestone procedure scope is not granted")
        if self.registry.vault is not self.vault or self.knowledge.vault is not self.vault:
            raise ValueError("Milestone verification must use one vault authority")
        root = Path(repository)
        if not root.is_absolute() or root not in self.registry.repository_roots:
            raise PermissionError("Milestone requires an exact owner-configured repository root")
        if (
            not isinstance(check_names, list)
            or not 1 <= len(check_names) <= 10
            or len(set(check_names)) != len(check_names)
            or any(name not in self.checks for name in check_names)
        ):
            raise ValueError("Select 1 to 10 distinct configured verification checks")
        if scope == "shared" and any(not self.checks[name]["shareable"] for name in check_names):
            raise PermissionError(
                "Shared procedures require explicitly shareable credential-free check commands"
            )
        if sum(len(value) for name in check_names for value in self.checks[name]["argv"]) > 32768:
            raise ValueError(
                "Selected verification commands exceed their aggregate argument budget"
            )
        single_line(entry_key, "procedure key", 160)
        single_line(writer, "procedure writer", 160)
        if type(expected_revision) is not int or expected_revision < 0:
            raise ValueError("Procedure review requires an explicit non-negative expected revision")
        if not isinstance(preconditions, list) or not 1 <= len(preconditions) <= 20:
            raise ValueError("Procedure requires 1 to 20 stated preconditions")
        preconditions = [
            single_line(value, "procedure precondition", 2048) for value in preconditions
        ]
        if not isinstance(evidence, list):
            raise TypeError("Milestone requires exact evidence references")
        for reference in evidence:
            if not isinstance(reference, dict) or "path" not in reference:
                raise ValueError("Milestone evidence must identify exact source paths")
            _relative_note(reference["path"], scopes)
        evidence = self.knowledge._evidence(evidence, scope, scopes)
        deadline = time.monotonic() + self.timeout
        receipt = {
            "schema_version": 1,
            "verification_id": uuid4().hex,
            "authority": "owner_local_observed_exit_codes",
            "started_at": now(),
            "repository": None,
            "checks": [],
        }
        failure = lambda reason: {
            "status": "verification_failed",
            "verification": {**receipt, "reason": reason},
        }
        try:
            _repository(root, deadline)
            self._code_evidence(evidence, root, scopes, deadline)
            before = _snapshot(root, deadline)
        except (TimeoutError, subprocess.TimeoutExpired, _Cancelled):
            return failure("timeout")
        receipt["repository"] = before
        if not _resources(self.vault.home):
            return failure("resource_floor")
        try:
            with _temporary_checks(self.vault.state) as directory:
                for name in check_names:
                    if time.monotonic() >= deadline:
                        return failure("timeout")
                    if not _resources(self.vault.home):
                        return failure("resource_floor")
                    checked, reason = self._run(name, root, directory, deadline)
                    receipt["checks"].append(checked)
                    if reason:
                        return failure(reason)
            if _snapshot(root, deadline) != before:
                return failure("repository_changed")
            self.knowledge._evidence(evidence, scope, scopes)
            self._code_evidence(evidence, root, scopes, deadline)
        except (TimeoutError, subprocess.TimeoutExpired, _Cancelled):
            return failure("timeout")
        except OSError:
            return failure("launch_or_resource_error")
        except ValueError:
            return failure("evidence_or_repository_changed")
        receipt["finished_at"] = now()
        commands = json.dumps(
            {name: self.checks[name]["argv"] for name in check_names}, ensure_ascii=False, indent=2
        )
        fence = chr(96) * max(
            3, max((len(run) for run in re.findall(chr(96) + "+", commands)), default=0) + 1
        )
        text = (
            "## Reusable procedure proposal\n\nDerived after the configured checks actually exited successfully. Generalisation and preconditions remain unreviewed; no command is executed by accepting or reading this proposal.\n\n## Preconditions\n\n"
            + "\n".join("- " + value for value in preconditions)
            + "\n\n## Commands\n\n"
            + fence
            + "json\n"
            + commands
            + "\n"
            + fence
            + "\n\n## Verification receipt\n\n"
            + json.dumps(receipt, ensure_ascii=False, sort_keys=True, indent=2)
            + "\n\n## Evidence\n\n"
            + json.dumps(evidence, ensure_ascii=False, sort_keys=True, indent=2)
        )
        proposal = self.knowledge.propose(
            entry_key,
            text,
            evidence=evidence,
            expected_revision=expected_revision,
            writer=writer,
            scope=scope,
            assertion_type="model_inference",
            scopes=scopes,
        )
        return {**proposal, "verification": receipt}


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] != "--worker":
        raise SystemExit("This module only exposes the internal configured-check worker")
    _worker(sys.argv[2])
