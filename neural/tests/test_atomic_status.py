import ctypes
import json
import os
import threading

import pytest

from dodo_mem_neural.artifacts import atomic_json


@pytest.mark.skipif(os.name != "nt", reason="Windows sharing semantics")
def test_atomic_json_retries_a_brief_reader_without_delete_sharing(tmp_path):
    path = tmp_path / "progress.json"
    path.write_text('{"old": true}')
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateFileW.restype = ctypes.c_void_p
    handle = kernel.CreateFileW(str(path), 0x80000000, 1, None, 3, 0, None)
    assert handle != ctypes.c_void_p(-1).value
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    release = threading.Timer(0.05, lambda: kernel.CloseHandle(handle))
    release.start()
    try:
        atomic_json(path, {"new": True})
        assert json.loads(path.read_text()) == {"new": True}
    finally:
        release.join()
