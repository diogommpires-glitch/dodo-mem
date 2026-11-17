import json
import subprocess
import sys
from pathlib import Path

import psutil
import pytest

from dodo_mem_neural import organisation_global


def test_cold_interpreter_rejects_a_foreign_base_prefix(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "base_prefix", str(tmp_path))
    with pytest.raises(ValueError, match="validated base"):
        organisation_global.cold_interpreter()


def test_cold_process_has_direct_pid_pinned_dependencies_and_no_cloud_credentials(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-credential")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "synthetic-credential")
    executable = organisation_global.cold_interpreter()
    environment = organisation_global.cold_environment(tmp_path)
    assert "OPENAI_API_KEY" not in environment and "ANTHROPIC_API_KEY" not in environment
    code = (
        "import json,os,time,numpy,igraph,dodo_memory,dodo_memory_desktop,dodo_mem_neural;"
        "from pathlib import Path;import dodo_mem_neural.organisation_global;"
        "print(json.dumps({'pid':os.getpid(),'numpy':numpy.__version__,"
        "'igraph':igraph.__version__,'origins':[str(Path(m.__file__).resolve()) "
        "for m in (dodo_memory,dodo_memory_desktop,dodo_mem_neural)]}),flush=True);time.sleep(120)"
    )
    process = subprocess.Popen(
        [str(executable), "-S", "-c", code],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=environment,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    try:
        result = json.loads(process.stdout.readline())
        assert result["pid"] == process.pid
        assert result["numpy"] == "2.5.4" and result["igraph"] == "1.0.0"
        root = Path(__file__).resolve().parents[2]
        assert [Path(value) for value in result["origins"]] == [
            root / "dodo_memory/__init__.py",
            root / "desktop/src/dodo_memory_desktop/__init__.py",
            root / "neural/src/dodo_mem_neural/__init__.py",
        ]
        owned = psutil.Process(process.pid)
        assert Path(owned.exe()) == executable
        assert all(
            child.name().casefold() == "conhost.exe" for child in owned.children(recursive=True)
        )
    finally:
        process.terminate()
        process.wait(timeout=5)
    assert not psutil.pid_exists(result["pid"])
