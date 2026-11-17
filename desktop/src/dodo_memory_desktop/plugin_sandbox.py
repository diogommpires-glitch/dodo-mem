"""Launch a constrained, job-bound Windows worker for each Wasm invocation."""

import atexit
import ctypes
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import threading
import zipfile
from contextlib import ExitStack, nullcontext
from ctypes import wintypes as w
from pathlib import Path
from uuid import uuid4

from .plugin_worker import MAX_REQUEST, MAX_RESULT

PVOID = ctypes.c_void_p
SIZE_T = ctypes.c_size_t
DWORD_PTR = ctypes.c_size_t
CAPABILITIES_ATTRIBUTE = 0x00020009
HANDLES_ATTRIBUTE = 0x00020002
PROCESS_FLAGS = 0x08080004 | 0x00000400  # No window, extended startup, suspended, Unicode env.
TOKEN_QUERY = 0x0008
TOKEN_CAPABILITIES = 30
TOKEN_IS_APP_CONTAINER = 29
TOKEN_APP_CONTAINER_SID = 31
MAX_WORKER_MEMORY = 256 * 1024**2


class SandboxError(Exception):
    pass


class SecurityCapabilities(ctypes.Structure):
    _fields_ = [
        ("sid", PVOID),
        ("capabilities", PVOID),
        ("count", w.DWORD),
        ("reserved", w.DWORD),
    ]


class StartupInfo(ctypes.Structure):
    _fields_ = [
        ("size", w.DWORD),
        ("reserved", w.LPWSTR),
        ("desktop", w.LPWSTR),
        ("title", w.LPWSTR),
        ("x", w.DWORD),
        ("y", w.DWORD),
        ("width", w.DWORD),
        ("height", w.DWORD),
        ("columns", w.DWORD),
        ("rows", w.DWORD),
        ("fill", w.DWORD),
        ("flags", w.DWORD),
        ("show", w.WORD),
        ("reserved_size", w.WORD),
        ("reserved_bytes", PVOID),
        ("stdin", w.HANDLE),
        ("stdout", w.HANDLE),
        ("stderr", w.HANDLE),
    ]


class StartupInfoEx(ctypes.Structure):
    _fields_ = [("startup", StartupInfo), ("attributes", PVOID)]


class ProcessInfo(ctypes.Structure):
    _fields_ = [
        ("process", w.HANDLE),
        ("thread", w.HANDLE),
        ("pid", w.DWORD),
        ("tid", w.DWORD),
    ]


class JobLimits(ctypes.Structure):
    _fields_ = [
        ("process_time", ctypes.c_int64),
        ("job_time", ctypes.c_int64),
        ("flags", w.DWORD),
        ("min_working", SIZE_T),
        ("max_working", SIZE_T),
        ("process_count", w.DWORD),
        ("affinity", DWORD_PTR),
        ("priority", w.DWORD),
        ("schedule", w.DWORD),
    ]


class JobLimitsEx(ctypes.Structure):
    _fields_ = [
        ("limits", JobLimits),
        ("io_counters", ctypes.c_uint64 * 6),
        ("process_memory", SIZE_T),
        ("job_memory", SIZE_T),
        ("peak_process", SIZE_T),
        ("peak_job", SIZE_T),
    ]


class Windows:
    """Only the Win32 calls needed to establish and verify the process boundary."""

    def __init__(self):
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        security = ctypes.WinDLL("advapi32", use_last_error=True)
        userenv = ctypes.WinDLL("userenv", use_last_error=True)

        def api(dll, name, result, *arguments):
            fn = getattr(dll, name)
            fn.argtypes = arguments
            fn.restype = result
            return fn

        self.create_profile = api(
            userenv,
            "CreateAppContainerProfile",
            ctypes.c_long,
            w.LPCWSTR,
            w.LPCWSTR,
            w.LPCWSTR,
            PVOID,
            w.DWORD,
            ctypes.POINTER(PVOID),
        )
        self.delete_profile = api(userenv, "DeleteAppContainerProfile", ctypes.c_long, w.LPCWSTR)
        self.free_sid = api(security, "FreeSid", PVOID, PVOID)
        self.sid_string = api(
            security,
            "ConvertSidToStringSidW",
            w.BOOL,
            PVOID,
            ctypes.POINTER(w.LPWSTR),
        )
        self.local_free = api(kernel, "LocalFree", w.HLOCAL, w.HLOCAL)
        self.init_attributes = api(
            kernel,
            "InitializeProcThreadAttributeList",
            w.BOOL,
            PVOID,
            w.DWORD,
            w.DWORD,
            ctypes.POINTER(SIZE_T),
        )
        self.update_attribute = api(
            kernel,
            "UpdateProcThreadAttribute",
            w.BOOL,
            PVOID,
            w.DWORD,
            DWORD_PTR,
            PVOID,
            SIZE_T,
            PVOID,
            PVOID,
        )
        self.delete_attributes = api(kernel, "DeleteProcThreadAttributeList", None, PVOID)
        self.create_process = api(
            kernel,
            "CreateProcessW",
            w.BOOL,
            w.LPCWSTR,
            w.LPWSTR,
            PVOID,
            PVOID,
            w.BOOL,
            w.DWORD,
            PVOID,
            w.LPCWSTR,
            ctypes.POINTER(StartupInfoEx),
            ctypes.POINTER(ProcessInfo),
        )
        self.open_token = api(
            security,
            "OpenProcessToken",
            w.BOOL,
            w.HANDLE,
            w.DWORD,
            ctypes.POINTER(w.HANDLE),
        )
        self.token_info = api(
            security,
            "GetTokenInformation",
            w.BOOL,
            w.HANDLE,
            ctypes.c_int,
            PVOID,
            w.DWORD,
            ctypes.POINTER(w.DWORD),
        )
        self.equal_sid = api(security, "EqualSid", w.BOOL, PVOID, PVOID)
        self.create_job = api(kernel, "CreateJobObjectW", w.HANDLE, PVOID, w.LPCWSTR)
        self.job_info = api(
            kernel, "SetInformationJobObject", w.BOOL, w.HANDLE, ctypes.c_int, PVOID, w.DWORD
        )
        self.assign_job = api(kernel, "AssignProcessToJobObject", w.BOOL, w.HANDLE, w.HANDLE)
        self.resume = api(kernel, "ResumeThread", w.DWORD, w.HANDLE)
        self.wait = api(kernel, "WaitForSingleObject", w.DWORD, w.HANDLE, w.DWORD)
        self.exit_code = api(
            kernel,
            "GetExitCodeProcess",
            w.BOOL,
            w.HANDLE,
            ctypes.POINTER(w.DWORD),
        )
        self.terminate = api(kernel, "TerminateProcess", w.BOOL, w.HANDLE, w.UINT)
        self.close = api(kernel, "CloseHandle", w.BOOL, w.HANDLE)

    @staticmethod
    def check(value, action):
        if not value:
            code = ctypes.get_last_error()
            raise SandboxError(f"{action} failed (Windows error {code}).")
        return value

    def token_buffer(self, token, kind):
        needed = w.DWORD()
        self.token_info(token, kind, None, 0, ctypes.byref(needed))
        self.check(needed.value, "Measuring sandbox token")
        buffer = ctypes.create_string_buffer(needed.value)
        self.check(
            self.token_info(token, kind, buffer, needed, ctypes.byref(needed)),
            "Reading sandbox token",
        )
        return buffer

    def assert_sandbox(self, process, sid):
        token = w.HANDLE()
        self.check(
            self.open_token(process, TOKEN_QUERY, ctypes.byref(token)), "Opening sandbox token"
        )
        try:
            is_container = self.token_buffer(token, TOKEN_IS_APP_CONTAINER)
            actual_sid = self.token_buffer(token, TOKEN_APP_CONTAINER_SID)
            capabilities = self.token_buffer(token, TOKEN_CAPABILITIES)
            if (
                not w.DWORD.from_buffer(is_container).value
                or not self.equal_sid(PVOID.from_buffer(actual_sid), sid)
                or w.DWORD.from_buffer(capabilities).value
            ):
                raise SandboxError(
                    "Worker did not receive the required capability-free AppContainer."
                )
        finally:
            self.close(token)


def _stage_runtime(root):
    base = Path(sys.base_prefix)
    library = Path(sysconfig.get_path("stdlib"))
    package = importlib.util.find_spec("wasmtime")
    if not package or not package.submodule_search_locations:
        raise SandboxError("Install the locked Wasmtime runtime to enable executable plugins.")
    prefix = f"python{sys.version_info.major}{sys.version_info.minor}"
    for path in [base / "python.exe", *base.glob("*.dll")]:
        shutil.copyfile(path, root / path.name)
    (root / f"{prefix}._pth").write_text(f"{prefix}.zip\nDLLs\n.\n", encoding="ascii")
    dlls = root / "DLLs"
    dlls.mkdir()
    for path in (base / "DLLs").iterdir():
        if path.suffix in {".pyd", ".dll"} and not path.name.startswith("_test"):
            shutil.copyfile(path, dlls / path.name)
    excluded = {
        "__pycache__",
        "ensurepip",
        "idlelib",
        "site-packages",
        "test",
        "tests",
        "tkinter",
        "venv",
    }
    with zipfile.ZipFile(root / f"{prefix}.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for path in library.rglob("*.py"):
            relative = path.relative_to(library)
            if not excluded.intersection(relative.parts):
                archive.write(path, relative.as_posix())
    shutil.copytree(
        Path(next(iter(package.submodule_search_locations))),
        root / "wasmtime",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    shutil.copyfile(Path(__file__).with_name("plugin_worker.py"), root / "runner.py")


class Sandbox:
    def __init__(self):
        self._lock = threading.Lock()
        self._cleanup = None
        self._closed = False
        atexit.register(self.close)

    def close(self):
        with self._lock:
            self._closed = True
            if self._cleanup:
                self._cleanup.close()
                self._cleanup = None
            atexit.unregister(self.close)

    def _prepare(self):
        if self._closed:
            raise SandboxError("The executable plugin host is closed.")
        if self._cleanup is not None:
            return
        if os.name != "nt":
            raise SandboxError("Executable plugins require Windows AppContainer isolation.")
        api = Windows()
        with ExitStack() as cleanup:
            directory = cleanup.enter_context(tempfile.TemporaryDirectory(prefix="dodo-wasm-"))
            root = Path(directory)
            _stage_runtime(root)
            name = f"Dodo.Plugin.{uuid4().hex}"
            sid = PVOID()
            result = api.create_profile(
                name, "Dodo plugin worker", "Temporary Wasm worker", None, 0, ctypes.byref(sid)
            )
            if result or not sid:
                raise SandboxError(
                    f"Creating AppContainer failed (HRESULT 0x{result & 0xFFFFFFFF:08x})."
                )
            cleanup.callback(api.delete_profile, name)
            cleanup.callback(api.free_sid, sid)
            string_sid = w.LPWSTR()
            api.check(api.sid_string(sid, ctypes.byref(string_sid)), "Reading AppContainer SID")
            try:
                os_root = Path(os.environ.get("SYSTEMROOT", "C:/Windows"))
                grant = subprocess.run(
                    [
                        str(os_root / "System32/icacls.exe"),
                        str(root),
                        "/grant",
                        f"*{string_sid.value}:(OI)(CI)(RX)",
                        "/T",
                        "/Q",
                    ],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    stdin=subprocess.DEVNULL,
                    creationflags=subprocess.CREATE_NO_WINDOW,
                    check=False,
                    timeout=15,
                )
            finally:
                api.local_free(ctypes.cast(string_sid, PVOID))
            if grant.returncode:
                raise SandboxError("Granting read-only access to the worker runtime failed.")
            self._api, self._root, self._sid = api, root, sid
            self._cleanup = cleanup.pop_all()

    def execute(self, request, *, authorise=nullcontext):
        payload = json.dumps(request, ensure_ascii=True, allow_nan=False).encode("ascii")
        if len(payload) > MAX_REQUEST:
            raise SandboxError("Plugin input exceeds its limit.")
        with self._lock:
            try:
                self._prepare()
                return self._launch(payload, authorise)
            except (OSError, subprocess.SubprocessError) as error:
                raise SandboxError("The isolated plugin worker could not start.") from error

    def _launch(self, payload, authorise):
        import msvcrt

        api, root, sid = self._api, self._root, self._sid
        with ExitStack() as cleanup:
            request_read, request_write = os.pipe()
            request_input = cleanup.enter_context(os.fdopen(request_read, "rb"))
            request_output = cleanup.enter_context(os.fdopen(request_write, "wb"))
            response_read, response_write = os.pipe()
            response_input = cleanup.enter_context(os.fdopen(response_read, "rb"))
            response_output = cleanup.enter_context(os.fdopen(response_write, "wb"))
            errors = cleanup.enter_context(open(os.devnull, "wb"))
            handles = (w.HANDLE * 3)(
                msvcrt.get_osfhandle(request_input.fileno()),
                msvcrt.get_osfhandle(response_output.fileno()),
                msvcrt.get_osfhandle(errors.fileno()),
            )
            for handle in handles:
                os.set_handle_inheritable(handle, True)
            size = SIZE_T()
            api.init_attributes(None, 2, 0, ctypes.byref(size))
            api.check(size.value, "Measuring sandbox process attributes")
            attributes = ctypes.create_string_buffer(size.value)
            api.check(
                api.init_attributes(attributes, 2, 0, ctypes.byref(size)),
                "Creating sandbox attributes",
            )
            cleanup.callback(api.delete_attributes, attributes)
            capabilities = SecurityCapabilities(sid, None, 0, 0)
            for key, value in (
                (CAPABILITIES_ATTRIBUTE, capabilities),
                (HANDLES_ATTRIBUTE, handles),
            ):
                api.check(
                    api.update_attribute(
                        attributes, 0, key, ctypes.byref(value), ctypes.sizeof(value), None, None
                    ),
                    "Configuring sandbox process attributes",
                )
            startup = StartupInfoEx()
            startup.startup.size = ctypes.sizeof(startup)
            startup.startup.flags = 0x00000100  # STARTF_USESTDHANDLES
            startup.startup.stdin, startup.startup.stdout, startup.startup.stderr = handles
            startup.attributes = ctypes.cast(attributes, PVOID)
            job = api.check(api.create_job(None, None), "Creating plugin job")
            cleanup.callback(api.close, job)
            limits = JobLimitsEx()
            limits.limits.flags = 0x00002108  # Kill on close, process memory, one process.
            limits.limits.process_count = 1
            limits.process_memory = MAX_WORKER_MEMORY
            api.check(
                api.job_info(job, 9, ctypes.byref(limits), ctypes.sizeof(limits)),
                "Limiting plugin job",
            )
            executable = str(root / "python.exe")
            command = ctypes.create_unicode_buffer(
                subprocess.list2cmdline([executable, "-I", "-S", "-B", str(root / "runner.py")])
            )
            windows = os.environ.get("SYSTEMROOT", "C:/Windows")
            local = os.environ["LOCALAPPDATA"]
            architecture = os.environ.get("PROCESSOR_ARCHITECTURE", "")
            if architecture not in {"AMD64", "ARM64", "x86"}:
                raise SandboxError("The Windows plugin worker architecture is unsupported.")
            # Windows derives the package-local profile from this one input path.
            environment = ctypes.create_unicode_buffer(
                f"LOCALAPPDATA={local}\0PROCESSOR_ARCHITECTURE={architecture}\0"
                f"SystemRoot={windows}\0TEMP={root}\0TMP={root}\0WINDIR={windows}\0\0"
            )
            process = ProcessInfo()
            try:
                api.check(
                    api.create_process(
                        executable,
                        command,
                        None,
                        None,
                        True,
                        PROCESS_FLAGS,
                        environment,
                        str(root),
                        ctypes.byref(startup),
                        ctypes.byref(process),
                    ),
                    "Launching AppContainer",
                )
            finally:
                request_input.close()
                response_output.close()
                errors.close()
            cleanup.callback(api.close, process.process)
            cleanup.callback(api.close, process.thread)
            try:
                api.assert_sandbox(process.process, sid)
                api.check(api.assign_job(job, process.process), "Assigning sandbox process to job")
                with authorise():
                    if api.resume(process.thread) == 0xFFFFFFFF:
                        api.check(False, "Starting isolated worker")
                result = self._exchange(
                    api, process.process, request_output, response_input, payload
                )
            finally:
                if api.wait(process.process, 0) != 0:
                    api.terminate(process.process, 1)
                    api.wait(process.process, 1000)
            return result

    @staticmethod
    def _exchange(api, process, request_output, response_input, payload):
        response = []
        failures = []

        def send():
            try:
                with request_output:
                    request_output.write(payload)
            except OSError:
                failures.append("input")

        def receive():
            try:
                response.append(response_input.read(MAX_RESULT * 2 + 1))
            except OSError:
                failures.append("output")

        threads = [threading.Thread(target=action, daemon=True) for action in (send, receive)]
        for thread in threads:
            thread.start()
        outcome = api.wait(process, 6000)
        if outcome != 0:
            api.terminate(process, 1)
            api.wait(process, 1000)
        for thread in threads:
            thread.join(1)
        code = w.DWORD()
        api.check(api.exit_code(process, ctypes.byref(code)), "Reading plugin exit status")
        if (
            outcome != 0
            or code.value
            or failures
            or not response
            or len(response[0]) > MAX_RESULT * 2
        ):
            raise SandboxError("The isolated plugin worker did not complete within its limits.")
        try:
            reply = json.loads(response[0])
        except (ValueError, UnicodeError) as error:
            raise SandboxError("The isolated plugin returned an invalid response.") from error
        if not isinstance(reply, dict) or reply.get("ok") is not True or "result" not in reply:
            raise SandboxError("The isolated plugin rejected the module or exceeded a limit.")
        return reply["result"]
