"""Trusted worker staged in a capability-free Windows AppContainer."""

import base64
import json
import os
import sys

MAX_REQUEST = 1024**2
MAX_CONTEXT = 48 * 1024
MAX_RESULT = 16 * 1024
MAX_MEMORY = 4 * 1024**2


def wasm_result(request):
    import wasmtime

    config = wasmtime.Config()
    config.consume_fuel = True
    config.wasm_threads = False
    engine = wasmtime.Engine(config)
    binary = base64.b64decode(request["module"], validate=True)
    if not binary.startswith(b"\0asm\x01\0\0\0"):
        raise ValueError("Expected a WebAssembly core module")
    module = wasmtime.Module(engine, binary)
    if module.imports:
        raise ValueError("Plugin modules cannot import any host or WASI functions")
    store = wasmtime.Store(engine)
    store.set_limits(memory_size=MAX_MEMORY, table_elements=4096, instances=1, tables=1, memories=1)
    store.set_fuel(1_000_000)
    exports = wasmtime.Instance(store, module, []).exports(store)
    memory, run = exports.get("memory"), exports.get("run")
    if not isinstance(memory, wasmtime.Memory) or not isinstance(run, wasmtime.Func):
        raise TypeError("Plugin must export memory and run")
    signature = run.type(store)
    if signature.params != [wasmtime.ValType.i32()] * 2 or signature.results != [
        wasmtime.ValType.i64()
    ]:
        raise ValueError("Plugin run must use the v1 signature (i32,i32)->i64")
    payload = json.dumps(request["context"], ensure_ascii=True, separators=(",", ":")).encode()
    if len(payload) > MAX_CONTEXT or memory.data_len(store) < len(payload):
        raise ValueError("Plugin input is outside its memory limit")
    memory.write(store, payload, 0)
    packed = run(store, 0, len(payload)) & ((1 << 64) - 1)
    offset, length = packed >> 32, packed & 0xFFFFFFFF
    if not 1 <= length <= MAX_RESULT or offset + length > memory.data_len(store):
        raise ValueError("Plugin returned an invalid result pointer")
    return json.loads(memory.read(store, offset, offset + length))


def isolation_probe(request):
    """Probe the host boundary using only paths and ports supplied by isolated tests."""
    import socket

    try:
        with open(request["path"], "rb"):
            readable = True
    except OSError:
        readable = False
    try:
        with socket.create_connection(("127.0.0.1", request["port"]), timeout=0.5):
            connected = True
    except OSError:
        connected = False
    return {
        "external_file_readable": readable,
        "loopback_connectable": connected,
        "environment": sorted(os.environ),
    }


def main():
    try:
        payload = sys.stdin.buffer.read(MAX_REQUEST + 1)
        if len(payload) > MAX_REQUEST:
            raise ValueError("Request exceeds the worker limit")
        request = json.loads(payload)
        if request["operation"] == "run":
            from wasmtime import Trap, WasmtimeError

            try:
                result = wasm_result(request)
            except (Trap, WasmtimeError) as error:
                raise ValueError("Wasm failed validation or execution") from error
        elif request["operation"] == "probe":
            result = isolation_probe(request)
        else:
            raise ValueError("Unknown worker operation")
        response = json.dumps({"ok": True, "result": result}, ensure_ascii=True, allow_nan=False)
    except (ImportError, KeyError, OSError, RuntimeError, TypeError, ValueError):
        response = '{"ok":false}'
    if len(response) > MAX_RESULT * 2:
        response = '{"ok":false}'
    sys.stdout.buffer.write(response.encode("ascii"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
