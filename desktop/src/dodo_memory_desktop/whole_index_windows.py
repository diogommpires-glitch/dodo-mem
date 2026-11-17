"""Native handles, drive discovery and directory change notifications."""

import ctypes as c
import hashlib
import msvcrt
import os
import struct
import threading
from contextlib import contextmanager
from ctypes import wintypes as w
from pathlib import Path
from uuid import UUID

from .whole_index import PLACEHOLDER, REPARSE, path_key

kernel = c.WinDLL("kernel32", use_last_error=True)
security = c.WinDLL("advapi32", use_last_error=True)
INVALID_HANDLE = c.c_void_p(-1).value


def native(library, name, restype, *arguments):
    fn = getattr(library, name)
    fn.argtypes, fn.restype = arguments, restype
    return fn


def desktop_directory() -> Path | None:
    shell = c.WinDLL("shell32")
    ole = c.WinDLL("ole32")
    get_folder = native(
        shell, "SHGetKnownFolderPath", c.c_long, w.LPVOID, w.DWORD, w.HANDLE, c.POINTER(c.c_void_p)
    )
    free = native(ole, "CoTaskMemFree", None, w.LPVOID)
    desktop = (c.c_ubyte * 16).from_buffer_copy(
        UUID("B4BFCC3A-DB2C-424C-B029-7FE99A87C641").bytes_le
    )
    result = c.c_void_p()
    try:
        # DONT_VERIFY resolves the configured folder without contacting a redirected location.
        if get_folder(desktop, 0x4000, None, c.byref(result)) == 0 and result.value:
            return Path(c.wstring_at(result))
        return None
    finally:
        free(result)


close_handle = native(kernel, "CloseHandle", w.BOOL, w.HANDLE)
create_file = native(
    kernel,
    "CreateFileW",
    w.HANDLE,
    w.LPCWSTR,
    w.DWORD,
    w.DWORD,
    w.LPVOID,
    w.DWORD,
    w.DWORD,
    w.HANDLE,
)
final_path = native(
    kernel,
    "GetFinalPathNameByHandleW",
    w.DWORD,
    w.HANDLE,
    w.LPWSTR,
    w.DWORD,
    w.DWORD,
)


class BasicJobLimits(c.Structure):
    _fields_ = [
        ("process_time", c.c_int64),
        ("job_time", c.c_int64),
        ("flags", w.DWORD),
        ("minimum_working_set", c.c_size_t),
        ("maximum_working_set", c.c_size_t),
        ("active_processes", w.DWORD),
        ("affinity", c.c_size_t),
        ("priority", w.DWORD),
        ("scheduling", w.DWORD),
    ]


class ExtendedJobLimits(c.Structure):
    _fields_ = [
        ("basic", BasicJobLimits),
        ("io_counts", c.c_uint64 * 6),
        ("process_memory", c.c_size_t),
        ("job_memory", c.c_size_t),
        ("peak_process_memory", c.c_size_t),
        ("peak_job_memory", c.c_size_t),
    ]


@contextmanager
def limit_parser_process(handle):
    create = native(kernel, "CreateJobObjectW", w.HANDLE, w.LPVOID, w.LPCWSTR)
    set_info = native(
        kernel, "SetInformationJobObject", w.BOOL, w.HANDLE, c.c_int, w.LPVOID, w.DWORD
    )
    assign = native(kernel, "AssignProcessToJobObject", w.BOOL, w.HANDLE, w.HANDLE)
    job = create(None, None)
    if not job:
        raise c.WinError(c.get_last_error())
    try:
        limits = ExtendedJobLimits()
        # PROCESS_MEMORY limits decompression. KILL_ON_JOB_CLOSE cleans up after worker exit.
        limits.basic.flags = 0x100 | 0x2000
        limits.process_memory = 512 * 1024**2
        if not set_info(job, 9, c.byref(limits), c.sizeof(limits)) or not assign(job, handle):
            raise c.WinError(c.get_last_error())
        yield
    finally:
        close_handle(job)


class AttributeTag(c.Structure):
    _fields_ = [("attributes", w.DWORD), ("tag", w.DWORD)]


file_info = native(
    kernel,
    "GetFileInformationByHandleEx",
    w.BOOL,
    w.HANDLE,
    c.c_int,
    w.LPVOID,
    w.DWORD,
)


def open_verified(path: Path):
    # OPEN_REPARSE_POINT makes replacement of a cloud placeholder or link fail before read.
    handle = create_file(str(path), 0x80000000, 7, None, 3, 0x00200000, None)
    if handle == INVALID_HANDLE:
        raise c.WinError(c.get_last_error())
    try:
        name = c.create_unicode_buffer(32768)
        count = final_path(handle, name, len(name), 0)
        if (
            not count
            or count >= len(name)
            or path_key(name.value.removeprefix("\\\\?\\")) != path_key(path)
        ):
            raise PermissionError("Source handle resolved to a different path")
        attrs = AttributeTag()
        if not file_info(handle, 9, c.byref(attrs), c.sizeof(attrs)):
            raise c.WinError(c.get_last_error())
        if attrs.attributes & (REPARSE | PLACEHOLDER | 0x10):
            raise PermissionError("Source is unavailable")
        fd = msvcrt.open_osfhandle(handle, os.O_RDONLY | os.O_BINARY)
    except BaseException:
        close_handle(handle)
        raise
    return os.fdopen(fd, "rb")


@contextmanager
def index_lock(database: Path):
    create_mutex = native(kernel, "CreateMutexW", w.HANDLE, w.LPVOID, w.BOOL, w.LPCWSTR)
    identifier = hashlib.sha256(path_key(database).encode()).hexdigest()
    handle = create_mutex(None, False, f"Local\\DodoWholeLibrary-{identifier}")
    if not handle:
        raise c.WinError(c.get_last_error())
    existed = c.get_last_error() == 183
    try:
        yield not existed
    finally:
        close_handle(handle)


def lower_index_priority():
    get_process = native(kernel, "GetCurrentProcess", w.HANDLE)
    set_priority = native(kernel, "SetPriorityClass", w.BOOL, w.HANDLE, w.DWORD)
    if not set_priority(get_process(), 0x4000):  # BELOW_NORMAL_PRIORITY_CLASS
        raise c.WinError(c.get_last_error())


@contextmanager
def current_sid():
    get_process = native(kernel, "GetCurrentProcess", w.HANDLE)
    open_token = native(
        security, "OpenProcessToken", w.BOOL, w.HANDLE, w.DWORD, c.POINTER(w.HANDLE)
    )
    get_token = native(
        security,
        "GetTokenInformation",
        w.BOOL,
        w.HANDLE,
        c.c_int,
        w.LPVOID,
        w.DWORD,
        c.POINTER(w.DWORD),
    )
    token = w.HANDLE()
    if not open_token(get_process(), 8, c.byref(token)):
        raise c.WinError(c.get_last_error())
    try:
        size = w.DWORD()
        get_token(token, 1, None, 0, c.byref(size))
        data = c.create_string_buffer(size.value)
        if not get_token(token, 1, data, size, c.byref(size)):
            raise c.WinError(c.get_last_error())
        yield c.cast(data, c.POINTER(c.c_void_p))[0]
    finally:
        close_handle(token)


def owned_paths(paths):
    get_security = native(
        security,
        "GetNamedSecurityInfoW",
        w.DWORD,
        w.LPCWSTR,
        c.c_int,
        w.DWORD,
        c.POINTER(c.c_void_p),
        w.LPVOID,
        w.LPVOID,
        w.LPVOID,
        c.POINTER(c.c_void_p),
    )
    equal_sid = native(security, "EqualSid", w.BOOL, w.LPVOID, w.LPVOID)
    free = native(kernel, "LocalFree", w.HLOCAL, w.HLOCAL)
    with current_sid() as user:
        for path in paths:
            owner, descriptor = c.c_void_p(), c.c_void_p()
            code = get_security(
                str(path), 1, 1, c.byref(owner), None, None, None, c.byref(descriptor)
            )
            try:
                if code == 0 and owner and equal_sid(owner, user):
                    yield path
            finally:
                if descriptor:
                    free(descriptor)


class DirectoryWatch:
    def __init__(self, root: Path, notify):
        self.root = root
        self.notify = notify
        self.thread = threading.Thread(target=self._run, name="dodo-directory-watch", daemon=True)

    def start(self):
        self.thread.start()

    def _run(self):
        changes = native(
            kernel,
            "ReadDirectoryChangesW",
            w.BOOL,
            w.HANDLE,
            w.LPVOID,
            w.DWORD,
            w.BOOL,
            w.DWORD,
            c.POINTER(w.DWORD),
            w.LPVOID,
            w.LPVOID,
        )
        handle = create_file(str(self.root), 1, 7, None, 3, 0x02200000, None)
        if handle == INVALID_HANDLE:
            return
        buffer = c.create_string_buffer(64 * 1024)
        try:
            while True:
                amount = w.DWORD()
                if not changes(
                    handle, buffer, len(buffer), True, 0x11F, c.byref(amount), None, None
                ):
                    self.notify(self.root)
                    return
                if not amount.value:
                    self.notify(self.root)
                    continue
                position = 0
                while position + 12 <= amount.value:
                    following, _, name_size = struct.unpack_from("<III", buffer, position)
                    end = position + 12 + name_size
                    if end > amount.value or name_size % 2:
                        self.notify(self.root)
                        break
                    try:
                        name = buffer[position + 12 : end].decode("utf-16-le")
                    except UnicodeError:
                        self.notify(self.root)
                        break
                    relative = Path(name)
                    if relative.is_absolute() or ".." in relative.parts:
                        self.notify(self.root)
                    else:
                        self.notify(self.root / relative)
                    if not following:
                        break
                    if following < 12 or position + following >= amount.value:
                        self.notify(self.root)
                        break
                    position += following
        finally:
            close_handle(handle)
