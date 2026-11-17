import importlib.util
from pathlib import Path


def harness():
    path = Path(__file__).parents[1] / "scripts/qualify_clients.py"
    spec = importlib.util.spec_from_file_location("qualify_clients", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_client_capture_rejects_unpaired_or_duplicate_tool_results():
    valid = [
        {"role": "assistant", "tool_calls": [{"id": "read"}]},
        {"role": "tool", "tool_call_id": "read"},
    ]
    module = harness()
    assert module.tool_pairs_valid(valid)
    assert not module.tool_pairs_valid(valid + [valid[-1]])
    assert not module.tool_pairs_valid([valid[0]])
    assert not module.tool_pairs_valid([{**valid[-1], "tool_call_id": "unknown"}])
