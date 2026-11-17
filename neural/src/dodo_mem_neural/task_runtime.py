"""Ask Windows Task Scheduler to own the service beyond the calling app's job."""

import base64
import hashlib
import json
import subprocess
import time
import uuid
from pathlib import Path
from xml.etree import ElementTree as ET

from .artifacts import atomic_json

TASK_NAME = "Dodo Mem Neural"
NAMESPACE = "http://schemas.microsoft.com/windows/2004/02/mit/task"
SETTINGS = {
    "MultipleInstancesPolicy": "IgnoreNew",
    "DisallowStartIfOnBatteries": "false",
    "StopIfGoingOnBatteries": "false",
    "AllowHardTerminate": "false",
    "StartWhenAvailable": "true",
    "RunOnlyIfNetworkAvailable": "false",
    "AllowStartOnDemand": "true",
    "Enabled": "true",
    "Hidden": "true",
    "RunOnlyIfIdle": "false",
    "WakeToRun": "false",
    "ExecutionTimeLimit": "PT0S",
    "Priority": "7",
}


def _quote(value):
    return "'" + value.replace("'", "''") + "'"


def _scheduler(action="query", xml=None, *, expected_xml=None):
    if action not in {"query", "register", "run", "delete"}:
        raise ValueError("Unsupported scheduled service operation.")
    script = (
        "$ErrorActionPreference='Stop';"
        "$identity=[Security.Principal.WindowsIdentity]::GetCurrent();$sid=$identity.User.Value;"
        "$service=New-Object -ComObject Schedule.Service;$service.Connect();"
        "$folder=$service.GetFolder('\\');$task=$null;$instance=$null;"
        f"try{{$task=$folder.GetTask({_quote(TASK_NAME)})}}catch{{"
        "if($_.Exception.HResult -ne -2147024894){throw}};"
    )
    if expected_xml is not None:
        script += f"if(!$task -or $task.Xml -cne {_quote(expected_xml)}){{throw 'The scheduled service changed.'}};"
    if action == "register":
        script += (
            "if($task){throw 'The scheduled service name is already in use.'};"
            f"$task=$folder.RegisterTask({_quote(TASK_NAME)},{_quote(xml)},2,$sid,$null,3,$null);"
        )
    elif action == "run":
        script += (
            "if(!$task){throw 'The scheduled service is missing.'};$instance=$task.Run($null);"
        )
    elif action == "delete":
        script += (
            "if(!$task){throw 'The scheduled service is missing.'};"
            "if($task.State -in @(2,4)){throw 'Stop the service normally before removing its task.'};"
            f"$folder.DeleteTask({_quote(TASK_NAME)},0);$task=$null;"
        )
    script += (
        "$policy=$null;if($task){$definition=$task.Definition;$settings=$definition.Settings;"
        "$principalUser=$definition.Principal.UserId;"
        "$verifiedUser=if($principalUser -ieq $sid -or $principalUser -ieq $identity.Name -or "
        "$identity.Name -ieq ($env:COMPUTERNAME+'\\'+$principalUser)){$sid}else{$principalUser};"
        "$first=$definition.Actions.Item(1);$policy=[ordered]@{"
        "user=$verifiedUser;logon_type=[int]$definition.Principal.LogonType;"
        "run_level=[int]$definition.Principal.RunLevel;actions=[int]$definition.Actions.Count;"
        "triggers=[int]$definition.Triggers.Count;action_type=[int]$first.Type;"
        "command=$first.Path;arguments=$first.Arguments;cwd=$first.WorkingDirectory;"
        "MultipleInstancesPolicy=[int]$settings.MultipleInstances;"
        "DisallowStartIfOnBatteries=[bool]$settings.DisallowStartIfOnBatteries;"
        "StopIfGoingOnBatteries=[bool]$settings.StopIfGoingOnBatteries;"
        "AllowHardTerminate=[bool]$settings.AllowHardTerminate;"
        "StartWhenAvailable=[bool]$settings.StartWhenAvailable;"
        "RunOnlyIfNetworkAvailable=[bool]$settings.RunOnlyIfNetworkAvailable;"
        "AllowStartOnDemand=[bool]$settings.AllowDemandStart;"
        "Enabled=[bool]$settings.Enabled;Hidden=[bool]$settings.Hidden;"
        "RunOnlyIfIdle=[bool]$settings.RunOnlyIfIdle;WakeToRun=[bool]$settings.WakeToRun;"
        "ExecutionTimeLimit=$settings.ExecutionTimeLimit;Priority=[int]$settings.Priority}};"
    )
    script += (
        "$result=if($task){[pscustomobject]@{sid=$sid;exists=$true;xml=$task.Xml;state=[int]$task.State;"
        "effective=$policy;"
        "instance_guid=$(if($instance){$instance.InstanceGuid});engine_pid=$(if($instance){$instance.EnginePID})}}"
        "else{[pscustomobject]@{sid=$sid;exists=$false;xml=$null;state=$null}}"
        ";[Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes(($result|ConvertTo-Json -Compress -Depth 4)))"
    )
    encoded = base64.b64encode(script.encode("utf-16-le")).decode("ascii")
    try:
        result = subprocess.run(
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-EncodedCommand", encoded],
            check=False,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=20,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
    except subprocess.TimeoutExpired as error:
        raise RuntimeError(
            "Windows Task Scheduler did not respond; preserve the launch journal."
        ) from error
    if result.returncode:
        raise RuntimeError(
            "Windows refused the scheduled service operation; no other task was changed."
        )
    if len(result.stdout) > 64 * 1024:
        raise ValueError("Scheduled service metadata exceeded its bound.")
    return json.loads(base64.b64decode(result.stdout.strip(), validate=True).decode("utf-8"))


def definition(command, cwd, sid, nonce):
    ET.register_namespace("", NAMESPACE)
    root = ET.Element(f"{{{NAMESPACE}}}Task", version="1.4")

    def add(parent, name, value=None, **attributes):
        child = ET.SubElement(parent, f"{{{NAMESPACE}}}{name}", attributes)
        child.text = value
        return child

    registration = add(root, "RegistrationInfo")
    add(registration, "Description", f"Dodo Mem owned hidden neural service. Registration {nonce}")
    add(root, "Triggers")
    principal = add(add(root, "Principals"), "Principal", id="DodoMem")
    add(principal, "UserId", sid)
    add(principal, "LogonType", "InteractiveToken")
    add(principal, "RunLevel", "LeastPrivilege")
    settings = add(root, "Settings")
    for key, value in SETTINGS.items():
        add(settings, key, value)
    action = add(add(root, "Actions", Context="DodoMem"), "Exec")
    add(action, "Command", str(command[0]))
    add(action, "Arguments", subprocess.list2cmdline(command[1:]))
    add(action, "WorkingDirectory", str(cwd))
    return ET.tostring(root, encoding="unicode")


def _fingerprint(xml):
    return hashlib.sha256(ET.canonicalize(xml_data=xml, strip_text=True).encode()).hexdigest()


def _validate_policy(xml, command, cwd, sid, nonce, effective):
    root = ET.fromstring(xml)
    find = lambda path: root.findtext(path, namespaces={"t": NAMESPACE})
    if (
        len(root.findall("t:Principals/t:Principal", {"t": NAMESPACE})) != 1
        or len(root.findall("t:Actions/*", {"t": NAMESPACE})) != 1
        or root.findall("t:Triggers/*", {"t": NAMESPACE})
        or find("t:Principals/t:Principal/t:UserId") != sid
        or find("t:Principals/t:Principal/t:LogonType") != "InteractiveToken"
        or find("t:Actions/t:Exec/t:Command") != str(command[0])
        or find("t:Actions/t:Exec/t:Arguments") != subprocess.list2cmdline(command[1:])
        or find("t:Actions/t:Exec/t:WorkingDirectory") != str(cwd)
        or find("t:RegistrationInfo/t:Description")
        != f"Dodo Mem owned hidden neural service. Registration {nonce}"
    ):
        raise ValueError("Scheduled service action, user or persistence policy changed.")
    expected = {
        "user": sid,
        "logon_type": 3,
        "run_level": 0,
        "actions": 1,
        "triggers": 0,
        "action_type": 0,
        "command": str(command[0]),
        "arguments": subprocess.list2cmdline(command[1:]),
        "cwd": str(cwd),
        **{
            key: value == "true" if value in {"true", "false"} else value
            for key, value in SETTINGS.items()
        },
        "MultipleInstancesPolicy": 2,
        "Priority": 7,
    }
    if not isinstance(effective, dict) or any(
        type(effective.get(key)) is not type(value) or effective[key] != value
        for key, value in expected.items()
    ):
        raise ValueError("Scheduled service effective COM policy is missing or changed.")


def _read(path):
    if path.resolve(strict=True) != path.absolute() or path.stat().st_size > 64 * 1024:
        raise ValueError("Scheduled service ownership record is redirected or oversized.")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("Scheduled service ownership record must be an object.")
    return value


def verify_registered(home, command, cwd):
    binding = _read(Path(home) / "task-binding.json")
    current = _scheduler()
    if not current["exists"]:
        raise ValueError("The owned scheduled service was removed; preserve its recovery journal.")
    _validate_policy(
        current["xml"], command, cwd, current["sid"], binding["nonce"], current.get("effective")
    )
    if current["sid"] != binding["sid"] or _fingerprint(current["xml"]) != binding["xml_sha256"]:
        raise ValueError("Scheduled service definition changed; its current content was preserved.")
    return {**binding, "_xml": current["xml"], "_state": current["state"]}


def ensure_registered(home, command, cwd):
    home = Path(home)
    binding_path, journal_path = home / "task-binding.json", home / "task-journal.json"
    if binding_path.exists():
        return verify_registered(home, command, cwd)
    current = _scheduler()
    if current["exists"]:
        if not journal_path.exists():
            raise ValueError("The scheduled service name belongs to an unrecognised task.")
        intent = _read(journal_path)
        if intent.get("state") != "registering" or intent["sid"] != current["sid"]:
            raise ValueError("The scheduled service has no matching creation intent.")
        _validate_policy(
            current["xml"], command, cwd, current["sid"], intent["nonce"], current.get("effective")
        )
    else:
        intent = {
            "state": "registering",
            "nonce": uuid.uuid4().hex,
            "sid": current["sid"],
            "before": None,
        }
        atomic_json(journal_path, intent)
        current = _scheduler("register", definition(command, cwd, current["sid"], intent["nonce"]))
        _validate_policy(
            current["xml"], command, cwd, current["sid"], intent["nonce"], current.get("effective")
        )
    binding = {
        "task": TASK_NAME,
        "sid": current["sid"],
        "nonce": intent["nonce"],
        "xml_sha256": _fingerprint(current["xml"]),
    }
    atomic_json(binding_path, binding)
    atomic_json(journal_path, {"state": "registered", **binding})
    return binding


def request_start(home, command, cwd):
    binding = ensure_registered(home, command, cwd)
    current = verify_registered(home, command, cwd)
    intent_path = Path(home) / "task-start-intent.json"
    if current["_state"] in {2, 4}:
        ticket = _read(intent_path)
        if ticket["task_xml_sha256"] != binding["xml_sha256"]:
            raise ValueError("The running scheduled service has no matching launch intent.")
        return ticket
    ticket = {"requested_at": time.time(), "task_xml_sha256": binding["xml_sha256"]}
    atomic_json(intent_path, ticket)
    result = _scheduler("run", expected_xml=current["_xml"])
    if not result["exists"]:
        raise RuntimeError("Windows did not retain the scheduled service.")
    for field in ("instance_guid", "engine_pid"):
        if result.get(field) is not None:
            ticket["task_" + field] = result[field]
    atomic_json(intent_path, ticket)
    return ticket


def remove_owned_task(home, command, cwd):
    current = verify_registered(home, command, cwd)
    atomic_json(Path(home) / "task-journal.json", {"state": "removing"})
    _scheduler("delete", expected_xml=current["_xml"])
    (Path(home) / "task-binding.json").unlink()
    atomic_json(Path(home) / "task-journal.json", {"state": "removed"})
