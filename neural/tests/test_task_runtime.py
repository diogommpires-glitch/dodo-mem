"""Task policy and journal tests never register or run a Windows task."""

import json
import subprocess
from pathlib import Path
from xml.etree import ElementTree as ET

import pytest

from dodo_mem_neural import task_runtime

COMMAND = ["C:/synthetic/pythonw.exe", "-m", "dodo_mem_neural.cli"]
DIRECTORY = Path("C:/synthetic/repository")


class Scheduler:
    def __init__(self):
        self.xml = None
        self.state = 3
        self.calls = []
        self.effective_override = None

    def __call__(self, action="query", xml=None, *, expected_xml=None):
        self.calls.append(action)
        if expected_xml is not None and expected_xml != self.xml:
            raise ValueError("Scheduled task changed before its action.")
        if action == "register":
            assert self.xml is None
            self.xml = xml
        elif action == "run":
            self.state = 4
        elif action == "delete":
            if self.state in {2, 4}:
                raise RuntimeError("Stop the service normally first.")
            self.xml = None
        effective = None
        if self.xml:
            root = ET.fromstring(self.xml)
            find = lambda path: root.findtext(path, namespaces={"t": task_runtime.NAMESPACE})
            effective = {
                "user": "S-1-5-21-1000000001-1000000002-1000000003-1001",
                "logon_type": 3,
                "run_level": 0,
                "actions": 1,
                "triggers": 0,
                "action_type": 0,
                "command": find("t:Actions/t:Exec/t:Command"),
                "arguments": find("t:Actions/t:Exec/t:Arguments"),
                "cwd": find("t:Actions/t:Exec/t:WorkingDirectory"),
            }
            for key in task_runtime.SETTINGS:
                value = find("t:Settings/t:" + key)
                effective[key] = value == "true" if value in {"true", "false"} else value
            effective.update(MultipleInstancesPolicy=2, Priority=7)
        return {
            "sid": "S-1-5-21-1000000001-1000000002-1000000003-1001",
            "exists": self.xml is not None,
            "xml": self.xml,
            "state": self.state,
            "effective": self.effective_override or effective,
        }


def test_task_runs_as_current_user_without_wake_timeout_or_forced_stop(tmp_path, monkeypatch):
    scheduler = Scheduler()
    monkeypatch.setattr(task_runtime, "_scheduler", scheduler)
    ticket = task_runtime.request_start(tmp_path, COMMAND, DIRECTORY)
    assert scheduler.calls.count("register") == scheduler.calls.count("run") == 1
    assert "InteractiveToken" in scheduler.xml and "LeastPrivilege" in scheduler.xml
    assert (
        task_runtime.verify_registered(tmp_path, COMMAND, DIRECTORY)["xml_sha256"]
        == ticket["task_xml_sha256"]
    )
    assert task_runtime.request_start(tmp_path, COMMAND, DIRECTORY) == ticket
    assert scheduler.calls.count("run") == 1
    with pytest.raises(RuntimeError, match="normally"):
        task_runtime.remove_owned_task(tmp_path, COMMAND, DIRECTORY)
    scheduler.state = 3
    task_runtime.remove_owned_task(tmp_path, COMMAND, DIRECTORY)
    assert scheduler.xml is None


def test_foreign_or_edited_task_is_preserved(tmp_path, monkeypatch):
    scheduler = Scheduler()
    scheduler.xml = task_runtime.definition(
        COMMAND, DIRECTORY, "S-1-5-21-1000000001-1000000002-1000000003-1001", "foreign"
    )
    monkeypatch.setattr(task_runtime, "_scheduler", scheduler)
    with pytest.raises(ValueError, match="unrecognised"):
        task_runtime.request_start(tmp_path, COMMAND, DIRECTORY)
    assert scheduler.calls == ["query"]
    scheduler.xml = None
    task_runtime.ensure_registered(tmp_path, COMMAND, DIRECTORY)
    scheduler.xml = scheduler.xml.replace("<Hidden>true</Hidden>", "<Hidden>false</Hidden>")
    with pytest.raises(ValueError, match="policy|changed"):
        task_runtime.request_start(tmp_path, COMMAND, DIRECTORY)
    assert "run" not in scheduler.calls


def test_registration_recovers_from_crash_before_binding_write(tmp_path, monkeypatch):
    scheduler = Scheduler()
    monkeypatch.setattr(task_runtime, "_scheduler", scheduler)
    write = task_runtime.atomic_json

    def crash_before_binding(path, value):
        if path.name == "task-binding.json":
            raise OSError("Synthetic interruption after Windows registered the task.")
        write(path, value)

    monkeypatch.setattr(task_runtime, "atomic_json", crash_before_binding)
    with pytest.raises(OSError):
        task_runtime.ensure_registered(tmp_path, COMMAND, DIRECTORY)
    assert scheduler.xml is not None
    monkeypatch.setattr(task_runtime, "atomic_json", write)
    task_runtime.ensure_registered(tmp_path, COMMAND, DIRECTORY)
    assert scheduler.calls.count("register") == 1
    assert (tmp_path / "task-binding.json").is_file()


def test_scheduler_timeout_is_visible_without_stopping_any_task(monkeypatch):
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired("synthetic scheduler helper", 20)

    monkeypatch.setattr(task_runtime.subprocess, "run", timeout)
    with pytest.raises(RuntimeError, match="did not respond"):
        task_runtime._scheduler()


def test_captured_normalised_windows_task_recovers_using_effective_com_values(
    tmp_path, monkeypatch
):
    captured = json.loads(
        (Path(__file__).parent / "fixtures/scheduled-task-normalised.json").read_text()
    )
    scheduler = Scheduler()
    scheduler.xml = captured["xml"].replace("C:/synthetic/repository", str(DIRECTORY))
    scheduler.effective_override = {**captured["effective"], "cwd": str(DIRECTORY)}
    monkeypatch.setattr(task_runtime, "_scheduler", scheduler)
    task_runtime.atomic_json(
        tmp_path / "task-journal.json",
        {
            "state": "registering",
            "nonce": "a" * 32,
            "sid": "S-1-5-21-1000000001-1000000002-1000000003-1001",
            "before": None,
        },
    )
    task_runtime.ensure_registered(tmp_path, COMMAND, DIRECTORY)
    assert "register" not in scheduler.calls and "run" not in scheduler.calls
    task_runtime.verify_registered(tmp_path, COMMAND, DIRECTORY)
    scheduler.effective_override["WakeToRun"] = True
    with pytest.raises(ValueError, match="effective COM policy"):
        task_runtime.verify_registered(tmp_path, COMMAND, DIRECTORY)
