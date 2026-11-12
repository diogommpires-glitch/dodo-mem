"""Run the pinned CPU converter inside an unprivileged Linux network namespace."""

import argparse
import ctypes
import errno
import hashlib
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path, PurePosixPath

from .parsing import DOCLING_VERSION, ParsedDocument, _available_memory, _convert, _read_artifacts

MIN_DISK_BYTES = 100 * 1024**3
MIN_RAM_BYTES = 5 * 1024**3


def _digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _unlinked(path):
    if path.resolve(strict=True) != path.absolute() or any(
        part.is_symlink() or getattr(part, "is_junction", lambda: False)()
        for part in (path, *path.parents)
    ):
        raise ValueError("Offline document files must not follow links or redirections")


def verify_model_cache(cache, manifest_sha256):
    """Verify the bounded exact cache inventory before reading any model into memory."""
    cache = Path(cache).absolute()
    _unlinked(cache)
    manifest = cache / "manifest.json"
    _unlinked(manifest)
    if not 1 <= manifest.stat().st_size <= 256 * 1024 or _digest(manifest) != manifest_sha256:
        raise ValueError("Offline model manifest differs from its pinned hash")
    value = json.loads(manifest.read_text(encoding="utf-8"))
    files = value.get("files")
    if (
        value.get("docling_version") != DOCLING_VERSION
        or not isinstance(files, dict)
        or not 1 <= len(files) <= 200
    ):
        raise ValueError("Offline model manifest requires the pinned Docling revision and files")
    total = 0
    for name, digest in files.items():
        relative = PurePosixPath(name)
        if (
            not isinstance(name, str)
            or relative.is_absolute()
            or ".." in relative.parts
            or relative.as_posix() != name
            or ":" in name
            or "\\" in name
            or not isinstance(digest, str)
            or not re.fullmatch(r"[0-9a-f]{64}", digest)
        ):
            raise ValueError(
                "Offline model paths must be canonical relative names with SHA-256 hashes"
            )
        path = cache / name
        _unlinked(path)
        total += path.stat().st_size
        if total > 2 * 1024**3 or _digest(path) != digest:
            raise ValueError("Offline model file exceeds its bound or differs from its hash")
    actual = {
        path.relative_to(cache).as_posix()
        for path in cache.rglob("*")
        if path.is_file() and path != manifest
    }
    if actual != set(files):
        raise ValueError("Offline model cache contains files outside the pinned manifest")
    return {"manifest_sha256": manifest_sha256, "files": len(files), "bytes": total}


def verify_runtime(root, manifest_sha256):
    """Bind the executable, hash-locked requirements, package bytes and conversion source."""
    root = Path(root).absolute()
    _unlinked(root)
    path = root / "runtime-manifest.json"
    _unlinked(path)
    if not 1 <= path.stat().st_size <= 8 * 1024**2 or _digest(path) != manifest_sha256:
        raise ValueError("Offline runtime manifest differs from its pinned hash")
    value = json.loads(path.read_text())
    files = value.get("files")
    if not isinstance(files, dict) or not 1 <= len(files) <= 40_000:
        raise ValueError("Offline runtime file inventory is missing or oversized")
    for relative, digest in files.items():
        name = PurePosixPath(relative)
        if name.is_absolute() or ".." in name.parts or name.as_posix() != relative:
            raise ValueError("Offline runtime inventory requires canonical relative paths")
        target = root / relative
        _unlinked(target)
        if _digest(target) != digest:
            raise ValueError("Offline runtime file differs from its recorded hash")
    executable = Path(sys.executable).resolve(strict=True)
    if executable != root / value["python"] or _digest(executable) != files[value["python"]]:
        raise ValueError("Offline worker is using a different Python executable")
    return {
        "manifest_sha256": manifest_sha256,
        "files": len(files),
        "requirements_lock_sha256": value["requirements_lock_sha256"],
    }


def isolation_command(python, package_root, arguments):
    """Create a fresh namespace, then discard its administrative capabilities before exec."""
    for value in (python, package_root):
        if (
            not isinstance(value, str)
            or not PurePosixPath(value).is_absolute()
            or PurePosixPath(value).as_posix() != value
            or ".." in PurePosixPath(value).parts
            or any(ord(char) < 32 for char in value)
        ):
            raise ValueError(
                "Offline runtime paths must be explicit canonical absolute Linux paths"
            )
    if not isinstance(arguments, (list, tuple)) or any(
        not isinstance(value, str) or "\0" in value for value in arguments
    ):
        raise ValueError("Offline worker arguments must be a structured string vector")
    return [
        "/usr/bin/env",
        "-u",
        "WSL_INTEROP",
        "-u",
        "WSLENV",
        "PATH=/usr/bin:/bin",
        "PYTHONPATH=" + package_root,
        "PYTHONDONTWRITEBYTECODE=1",
        "HF_HUB_OFFLINE=1",
        "TRANSFORMERS_OFFLINE=1",
        "CUDA_VISIBLE_DEVICES=-1",
        "OMP_NUM_THREADS=2",
        "MKL_NUM_THREADS=2",
        "/usr/bin/unshare",
        "--user",
        "--map-root-user",
        "--net",
        "--",
        "/usr/bin/setpriv",
        "--bounding-set=-all",
        "--inh-caps=-all",
        "--ambient-caps=-all",
        "--no-new-privs",
        "--",
        python,
        "-B",
        "-m",
        "dodo_memory.offline_docling",
        *arguments,
    ]


def require_isolated_network():
    """Refuse a live network before source reads, imports or connection probes."""
    if sys.platform != "linux":
        raise RuntimeError("Offline Docling requires an isolated network on Linux")
    interfaces = sorted(name for _, name in socket.if_nameindex())
    routes = max(0, len(Path("/proc/net/route").read_text().splitlines()) - 1)
    status = dict(
        line.split(":", 1)
        for line in Path("/proc/self/status").read_text().splitlines()
        if ":" in line
    )
    if (
        interfaces != ["lo"]
        or routes
        or int(status["CapEff"].strip(), 16)
        or status["NoNewPrivs"].strip() != "1"
    ):
        raise PermissionError(
            "Worker requires an isolated network with no routes and no administrative capabilities"
        )
    return {
        "net_namespace": os.readlink("/proc/self/ns/net"),
        "interfaces": interfaces,
        "route_entries": routes,
        "effective_capabilities": status["CapEff"].strip(),
        "no_new_privileges": True,
    }


def network_probe():
    """Exercise Python and a direct libc connection without relying on a Python audit hook."""
    state = require_isolated_network()
    with socket.socket() as connection:
        connection.settimeout(0.2)
        try:
            connection.connect(("198.51.100.1", 443))
        except OSError as error:
            python_errno = error.errno
        else:
            raise RuntimeError("Isolated Python connection unexpectedly succeeded")
    libc = ctypes.CDLL(None, use_errno=True)

    class SockAddr(ctypes.Structure):
        _fields_ = [
            ("family", ctypes.c_ushort),
            ("port", ctypes.c_ushort),
            ("address", ctypes.c_ubyte * 4),
            ("padding", ctypes.c_ubyte * 8),
        ]

    descriptor = libc.socket(socket.AF_INET, socket.SOCK_STREAM, 0)
    if descriptor < 0:
        raise OSError(ctypes.get_errno(), "Native socket creation failed")
    try:
        address = SockAddr(
            socket.AF_INET,
            socket.htons(443),
            (ctypes.c_ubyte * 4)(198, 51, 100, 1),
            (ctypes.c_ubyte * 8)(),
        )
        result = libc.connect(descriptor, ctypes.byref(address), ctypes.sizeof(address))
        native_errno = ctypes.get_errno()
    finally:
        libc.close(descriptor)
    if python_errno != errno.ENETUNREACH or result != -1 or native_errno != errno.ENETUNREACH:
        raise RuntimeError("Kernel network denial differs from the required ENETUNREACH control")
    child = subprocess.run(
        [sys.executable, "-B", "-c", "import os;print(os.readlink('/proc/self/ns/net'))"],
        capture_output=True,
        text=True,
        check=True,
        timeout=5,
    )
    if child.stdout.strip() != state["net_namespace"]:
        raise RuntimeError("Subprocess escaped the isolated network namespace")
    return {
        **state,
        "python_errno": python_errno,
        "libc_errno": native_errno,
        "libc_result": result,
        "child_net_namespace": child.stdout.strip(),
    }


def _resource_state(directory):
    available, free = _available_memory(), shutil.disk_usage(directory).free
    if available < MIN_RAM_BYTES or free < MIN_DISK_BYTES:
        raise MemoryError("Offline Docling requires 5 GiB RAM and 100 GiB free disk")
    return {"available_ram": available, "disk_free": free}


def _probe_during_conversion(output):
    """The independent child starts before the converter installs its Python audit hook."""
    path, stop = output / "network-probes.jsonl", output / ".stop-probes"
    parent = Path("/proc") / str(os.getppid())
    created = int((parent / "stat").read_text().rsplit(") ", 1)[1].split()[19])
    nonce = os.environ.get("DODO_OWNED_INDEX_NONCE")
    try:
        with path.open("x", encoding="utf-8") as stream:
            while not stop.exists():
                row = {**network_probe(), **_resource_state(output), "monotonic": time.monotonic()}
                stream.write(json.dumps(row) + "\n")
                stream.flush()
                time.sleep(0.5)
    except (OSError, ValueError, RuntimeError, MemoryError) as error:
        (output / "network-failure.json").write_text(
            json.dumps({"error_type": type(error).__name__})
        )
        # Only the exact parent launched in the nonce-bound owned group may be stopped.
        if nonce is not None:
            try:
                current = (parent / "stat").read_text().rsplit(") ", 1)[1].split()
                marker = b"DODO_OWNED_INDEX_NONCE=" + nonce.encode()
                if int(current[19]) == created and marker in (
                    parent / "environ"
                ).read_bytes().split(b"\0"):
                    os.kill(int(parent.name), signal.SIGTERM)
            except (FileNotFoundError, ProcessLookupError):
                pass
        raise


def convert_worker(configuration):
    """Convert only after exact source/cache and kernel isolation qualification."""
    network = network_probe()
    if not re.fullmatch(r"[0-9a-f]{64}", os.environ.get("DODO_OWNED_INDEX_NONCE", "")):
        raise PermissionError("Offline conversion requires an owned Linux process nonce")
    runtime = verify_runtime(
        configuration["runtime_root"], configuration["runtime_manifest_sha256"]
    )
    original, output, cache = (Path(configuration[key]) for key in ("original", "output", "cache"))
    _unlinked(original)
    if original.suffix.lower() != ".pdf" or not 0 < original.stat().st_size <= 100 * 1024**2:
        raise ValueError("Offline conversion requires a retained PDF of 1 byte to 100 MiB")
    if _digest(original) != configuration["source_sha256"]:
        raise ValueError("Offline source differs from the frozen source hash")
    output.mkdir(mode=0o700, parents=True, exist_ok=True)
    _unlinked(output)
    if any(output.iterdir()):
        raise ValueError("Offline conversion output must be empty")
    resource = _resource_state(output)
    before = verify_model_cache(cache, configuration["model_manifest_sha256"])
    affinity = sorted(os.sched_getaffinity(0))[:2]
    if len(affinity) != 2:
        raise RuntimeError("Offline conversion requires two available logical CPUs")
    os.sched_setaffinity(0, affinity)
    import cv2
    import pypdfium2
    import torch

    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    cv2.setNumThreads(2)
    if torch.version.cuda is not None or torch.cuda.is_initialized():
        raise RuntimeError("Offline conversion requires the pinned CPU-only Torch build")
    geometry = {}
    with pypdfium2.PdfDocument(original) as source:
        if len(source) > 100:
            raise ValueError("Offline conversion requires at most 100 original pages")
        for index, page in enumerate(source, 1):
            geometry[index] = {
                "media_box": list(page.get_mediabox()),
                "crop_box": list(page.get_cropbox()),
                "size": list(page.get_size()),
                "rotation": page.get_rotation(),
            }
    temporary = output / "tmp"
    temporary.mkdir(mode=0o700)
    os.environ.update(
        TMPDIR=str(temporary),
        HF_HOME=str(temporary / "hf"),
        XDG_CACHE_HOME=str(temporary / "cache"),
    )
    probe = subprocess.Popen(
        [sys.executable, "-B", "-m", "dodo_memory.offline_docling", "--probe-loop", str(output)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    started = time.monotonic()
    try:
        _convert(original, output, cache)
    finally:
        (output / ".stop-probes").write_bytes(b"finished")
        try:
            probe.wait(timeout=5)
        finally:
            if probe.poll() is None:
                probe.terminate()
                probe.wait(timeout=5)
    if probe.returncode:
        raise RuntimeError("Independent kernel network/resource probe failed during conversion")
    after = verify_model_cache(cache, configuration["model_manifest_sha256"])
    if before != after or _digest(original) != configuration["source_sha256"]:
        raise ValueError("Source or model cache changed during offline conversion")
    probes = [json.loads(row) for row in (output / "network-probes.jsonl").read_text().splitlines()]
    if not probes or not any(row["monotonic"] >= started for row in probes):
        raise RuntimeError("No independent kernel denial was observed during conversion")
    result_path = output / "result.json"
    result = json.loads(result_path.read_text())
    from .figure_evidence import figure_manifest

    document = json.loads((output / "document.json").read_text(encoding="utf-8"))
    coverage = figure_manifest(document, configuration["source_sha256"], page_geometry=geometry)
    figure_path = output / "figure-evidence.json"
    figure_path.write_text(json.dumps(coverage, allow_nan=False), encoding="utf-8")
    result["artifacts"][figure_path.name] = {
        "bytes": figure_path.stat().st_size,
        "sha256": _digest(figure_path),
    }
    result["metadata"]["figure_caption_coverage"] = {
        "counts": coverage["counts"],
        "complete": coverage["complete"],
        "fidelity_verified_against_original": False,
        "artifact": figure_path.name,
    }
    if coverage["counts"]["failed"]:
        result["metadata"]["uncertainties"].append(
            "Figure captions are missing or contaminated; inspect figure-evidence.json and the registered original pages/regions."
        )
    result["metadata"]["os_isolation"] = {
        **network,
        "policy": "unprivileged_user_and_network_namespace_no_capabilities",
        "model_cache_before": before,
        "model_cache_after": after,
        "network_probe_samples": len(probes),
        "minimum_available_ram": min(
            resource["available_ram"], *(row["available_ram"] for row in probes)
        ),
        "minimum_disk_free": min(resource["disk_free"], *(row["disk_free"] for row in probes)),
        "source_sha256": configuration["source_sha256"],
        "runtime": runtime,
        "logical_cpu_affinity": affinity,
        "torch_intraop_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "opencv_threads": cv2.getNumThreads(),
        "torch_cuda_build": torch.version.cuda,
        "cuda_initialized": torch.cuda.is_initialized(),
    }
    result_path.write_text(json.dumps(result), encoding="utf-8")


def parse_offline_document(
    original,
    *,
    work_dir,
    runtime_manifest_sha256,
    model_manifest_sha256,
    runtime_root,
    distribution="Ubuntu-24.04",
    cancelled=lambda: False,
    timeout=180,
):
    """Host adapter for an owner-qualified Linux runtime; returned bodies remain local."""
    from .native_process import linux_path, owned_wsl_process

    if os.name != "nt":
        raise RuntimeError("This offline host adapter requires the qualified Windows/WSL platform")
    if type(timeout) not in (int, float) or not 0 < timeout <= 180:
        raise ValueError(
            "Offline conversion timeout must be greater than zero and at most 180 seconds"
        )
    for digest in (runtime_manifest_sha256, model_manifest_sha256):
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ValueError("Offline conversion requires explicit runtime and model SHA-256 pins")
    if cancelled():
        raise InterruptedError("Offline conversion cancelled before launch")
    original = Path(original).absolute()
    _unlinked(original)
    if original.suffix.lower() != ".pdf" or not 0 < original.stat().st_size <= 100 * 1024**2:
        raise ValueError("Offline conversion requires a retained PDF of 1 byte to 100 MiB")
    work_dir = Path(work_dir).absolute()
    work_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    _unlinked(work_dir)
    if any(work_dir.iterdir()):
        raise ValueError("Offline conversion work directory must be empty")
    root = PurePosixPath(runtime_root)
    command = isolation_command(
        str(root / "env/bin/python"),
        str(root / "runtime-code"),
        ["--worker", linux_path(work_dir / "worker.json")],
    )
    output = work_dir / "artifacts"
    configuration = {
        "original": linux_path(original),
        "output": linux_path(output),
        "cache": str(root / "models"),
        "runtime_root": str(root),
        "source_sha256": _digest(original),
        "runtime_manifest_sha256": runtime_manifest_sha256,
        "model_manifest_sha256": model_manifest_sha256,
    }
    (work_dir / "worker.json").write_text(json.dumps(configuration), encoding="utf-8")
    minimum = _resource_state(work_dir)["available_ram"]
    started = time.monotonic()
    with (
        (work_dir / "conversion.log").open("wb") as log,
        owned_wsl_process(
            command,
            distribution=distribution,
            directory=work_dir,
            log=log,
            stdin=subprocess.DEVNULL,
        ) as owned,
    ):
        while owned.process.poll() is None:
            minimum = min(minimum, _resource_state(work_dir)["available_ram"])
            if cancelled():
                raise InterruptedError("Offline conversion cancelled")
            if time.monotonic() - started > timeout:
                raise TimeoutError("Offline conversion exceeded its bounded process time")
            time.sleep(0.2)
    if owned.process.returncode:
        raise ValueError("Offline conversion failed; inspect its local metadata and conversion.log")
    if cancelled():
        raise InterruptedError("Offline conversion cancelled before publication")
    if _digest(original) != configuration["source_sha256"]:
        raise ValueError("Offline retained source changed during conversion")
    receipt = json.loads((output / "result.json").read_text())
    _resource_state(work_dir)
    artifacts = _read_artifacts(output, receipt["artifacts"])
    metadata = {
        **receipt["metadata"],
        "source_sha256": configuration["source_sha256"],
        "minimum_host_available_ram": minimum,
        "seconds": time.monotonic() - started,
        "owned_cleanup": owned.cleanup,
        "artifacts": receipt["artifacts"],
    }
    return ParsedDocument(artifacts["document.md"].decode("utf-8"), metadata, artifacts)


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--network-probe", action="store_true")
    group.add_argument("--probe-loop", type=Path)
    group.add_argument("--worker", type=Path)
    args = parser.parse_args()
    if args.network_probe:
        print(json.dumps(network_probe()))
    elif args.probe_loop is not None:
        _probe_during_conversion(args.probe_loop)
    else:
        convert_worker(json.loads(args.worker.read_text(encoding="utf-8")))


if __name__ == "__main__":
    main()
