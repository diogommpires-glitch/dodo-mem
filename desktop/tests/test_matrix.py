"""Synthetic Matrix runs never invoke providers or change a real project."""

import copy
import threading
import time

import pytest
from desktop_fixture import FixtureGateway
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication, QDialog

from dodo_memory_desktop.app import HOME, Desktop
from dodo_memory_desktop.gateway import GatewayError
from dodo_memory_desktop.matrix import checked_run


def until(predicate, seconds=3):
    deadline = time.monotonic() + seconds
    while not predicate():
        assert time.monotonic() < deadline
        QApplication.processEvents()
        time.sleep(0.005)


class MatrixGateway(FixtureGateway):
    def __init__(self):
        super().__init__()
        self.matrix_calls = []
        self.owner_calls = []
        self.runs = {}
        self.status_wait = None
        self.session_id = "f" * 32
        self.external_runs = {}

    def call(self, name, arguments=None):
        if name != "connect_agent_matrix" and not name.startswith("matrix_"):
            return super().call(name, arguments)
        if not self.connected:
            raise GatewayError("Matrix offline.", connection_lost=True)
        args = arguments or {}
        self.matrix_calls.append((name, copy.deepcopy(args), threading.get_ident()))
        if name == "connect_agent_matrix":
            return {
                "state": "ready",
                "requires_local_approval": True,
                "session_id": self.session_id,
            }
        assert args["session_id"] == self.session_id
        if name == "matrix_list":
            return {"runs": copy.deepcopy(list(self.runs.values()))}
        if name == "matrix_submit":
            run = {
                "run_id": f"run-{len(self.runs) + 1}",
                "state": "awaiting_approval",
                "request_sha256": "a" * 64,
                "request": {
                    key: value
                    for key, value in args.items()
                    if key not in {"request_id", "session_id"}
                },
            }
            self.runs[run["run_id"]] = run
            return copy.deepcopy(run)
        run = copy.deepcopy(self.runs[args["run_id"]])
        if name == "matrix_status" and self.status_wait:
            identity, started, release = self.status_wait
            if identity == run["run_id"]:
                self.status_wait = None
                started.set()
                assert release.wait(3)
        if name == "matrix_control":
            run["state"] = {"pause": "paused", "resume": "queued", "stop": "stopped"}[
                args["action"]
            ]
            self.runs[run["run_id"]] = copy.deepcopy(run)
        return run

    def matrix_owner(self, operation, arguments):
        self.owner_calls.append((operation, copy.deepcopy(arguments), threading.get_ident()))
        runs = {**self.runs, **self.external_runs}
        if operation == "matrix_list":
            return {"runs": copy.deepcopy(list(runs.values()))}
        run = runs[arguments["run_id"]]
        if operation == "matrix_status":
            return copy.deepcopy(run)
        if operation == "matrix_control":
            run["state"] = {"pause": "paused", "resume": "queued", "stop": "stopped"}[
                arguments["action"]
            ]
            return copy.deepcopy(run)
        assert arguments["expected_request_sha256"] == run["request_sha256"]
        run["state"] = "queued"
        run["result"] = {
            "workers": [{"role": "planner", "model": "fixture-model", "effort": "high"}],
            "answer": "Synthetic answer, no provider invoked.",
        }
        return copy.deepcopy(run)


@pytest.fixture
def workspace(tmp_path):
    gateway = MatrixGateway()
    gateway.home = tmp_path / "gateway"
    settings = QSettings(str(tmp_path / "matrix.ini"), QSettings.Format.IniFormat)
    window = Desktop(gateway, settings, tmp_path)
    window.show()
    until(lambda: window.online)
    assert gateway.matrix_calls == []
    window.show_page(window.matrix_index)
    view = window.matrix
    assert gateway.matrix_calls == []
    yield window, view, gateway
    window.close()
    assert window.pool.waitForDone(5000)
    QApplication.processEvents()


def prepare(view, gateway, *, uncertain=False):
    view.connect_matrix.click()
    until(lambda: view.connected and not view.busy)
    view.question.setPlainText("Plan the synthetic project.\nKeep the exact request.")
    view.uncertain.setChecked(uncertain)
    view.prepare.click()
    until(lambda: view.current is not None and not view.busy)
    return copy.deepcopy(gateway.runs[view.current["run_id"]])


def test_connect_and_prepare_never_start_workers_and_poll_only_on_the_visible_page(workspace):
    window, view, gateway = workspace
    run = prepare(view, gateway)
    assert run["state"] == "awaiting_approval" and not gateway.owner_calls
    assert run["request"]["providers"] == ["codex", "claude"]
    assert run["request"]["token_budget"] == 128000
    assert run["request"]["budget_seconds"] == 28800
    assert run["request"]["allow_uncertain_usage"] is False
    assert all(thread != threading.get_ident() for _name, _args, thread in gateway.matrix_calls)
    assert all(
        args["session_id"] == gateway.session_id
        for name, args, _thread in gateway.matrix_calls
        if name != "connect_agent_matrix"
    )
    assert view.timer.isActive() and view.timer.interval() == 2000
    assert view.approve.isEnabled() and not view.pause.isEnabled()
    window.show_page(HOME)
    assert not view.timer.isActive()
    before = len(gateway.matrix_calls)
    view.timer.timeout.emit()
    QApplication.processEvents()
    assert len(gateway.matrix_calls) == before


def test_approval_reviews_server_request_and_requires_uncertain_spending_acknowledgement(workspace):
    _window, view, gateway = workspace
    run = prepare(view, gateway, uncertain=True)
    view.question.setPlainText("A later unsent edit must not be approved.")
    view.approve.click()
    dialog = view.approval_dialog
    assert run["request"]["question"] in dialog.request_text.toPlainText()
    assert "A later unsent edit" not in dialog.request_text.toPlainText()
    assert "128000" in dialog.request_text.toPlainText()
    assert "28800" in dialog.request_text.toPlainText()
    assert "codex" in dialog.request_text.toPlainText()
    assert "claude" in dialog.request_text.toPlainText()
    assert not dialog.start.isEnabled()
    dialog.spending.setChecked(True)
    assert dialog.start.isEnabled()
    dialog.start.click()
    until(lambda: bool(gateway.owner_calls) and not view.busy)
    operation, arguments, thread = gateway.owner_calls[0]
    assert operation == "matrix_approve" and thread != threading.get_ident()
    assert arguments == {"run_id": run["run_id"], "expected_request_sha256": run["request_sha256"]}
    assert view.current["state"] == "queued"
    assert view.workers.item(0, 1).text() == "planner"
    assert view.workers.item(0, 2).text() == "fixture-model"
    assert view.workers.item(0, 3).text() == "high"
    assert "Synthetic answer" in view.output.toPlainText()


def test_controls_and_disconnect_clear_output_and_stop_polling(workspace):
    _window, view, gateway = workspace
    prepare(view, gateway)
    view.approve.click()
    view.approval_dialog.start.click()
    until(lambda: view.current["state"] == "queued" and not view.busy)
    for widget, state in ((view.pause, "paused"), (view.resume, "queued"), (view.stop, "stopped")):
        assert widget.isEnabled()
        widget.click()
        until(lambda expected=state: view.current["state"] == expected and not view.busy)
    gateway.connected = False
    view.refresh.click()
    until(lambda: not view.connected)
    assert view.current is None and not view.run_list.count()
    assert not view.output.toPlainText() and not view.workers.rowCount()
    assert not view.timer.isActive() and not view.busy
    session_id = view.session_id
    gateway.connected = True
    view.connect_matrix.click()
    until(lambda: view.connected and not view.busy and view.run_list.count() == 1)
    assert view.session_id == session_id
    assert sum(name == "connect_agent_matrix" for name, _args, _thread in gateway.matrix_calls) == 1


def test_stale_status_cannot_replace_another_selected_run(workspace):
    _window, view, gateway = workspace
    first = prepare(view, gateway)
    second = copy.deepcopy(first)
    second["run_id"] = "run-2"
    second["request"]["question"] = "Another synthetic request."
    gateway.runs[second["run_id"]] = second
    view.refresh.click()
    until(lambda: view.run_list.count() == 2)
    started, release = threading.Event(), threading.Event()
    gateway.status_wait = first["run_id"], started, release
    view.refresh.click()
    until(started.is_set)
    try:
        view.run_list.setCurrentRow(1)
        until(lambda: view.current["run_id"] == second["run_id"])
    finally:
        release.set()
    until(lambda: "matrix-status" not in view.desktop.pending)
    assert view.current["run_id"] == second["run_id"]
    assert "Another synthetic request" in view.run_detail.toPlainText()


def test_rejecting_local_review_makes_no_owner_call(workspace):
    _window, view, gateway = workspace
    prepare(view, gateway)
    view.approve.click()
    view.approval_dialog.done(QDialog.DialogCode.Rejected)
    assert view.approval_dialog is None and not gateway.owner_calls


def test_status_started_before_approval_cannot_restore_the_old_state(workspace):
    _window, view, gateway = workspace
    run = prepare(view, gateway)
    started, release = threading.Event(), threading.Event()
    gateway.status_wait = run["run_id"], started, release
    view.refresh.click()
    until(started.is_set)
    try:
        view.approve.click()
        view.approval_dialog.start.click()
        until(lambda: view.current["state"] == "queued" and not view.busy)
    finally:
        release.set()
    assert view.desktop.pool.waitForDone(3000)
    QApplication.processEvents()
    assert view.current["state"] == "queued"


def test_delayed_list_cannot_erase_a_newer_selected_run_result(workspace):
    _window, view, gateway = workspace
    run = prepare(view, gateway)
    gateway.runs[run["run_id"]]["state"] = "queued"
    captured, release_list, release_status = threading.Event(), threading.Event(), threading.Event()
    call = gateway.call

    def delayed(name, arguments=None):
        if name == "matrix_list":
            value = call(name, arguments)
            captured.set()
            assert release_list.wait(3)
            return value
        if name == "matrix_status":
            assert release_status.wait(3)
        return call(name, arguments)

    gateway.call = delayed
    view.refresh.click()
    until(captured.is_set)
    try:
        gateway.runs[run["run_id"]].update(
            state="completed", result={"answer": "A newer synthetic answer."}
        )
        release_status.set()
        until(lambda: view.current["state"] == "completed")
    finally:
        release_status.set()
        release_list.set()
    assert view.desktop.pool.waitForDone(3000)
    QApplication.processEvents()
    assert view.current["state"] == "completed"
    assert view.output.toPlainText() == "A newer synthetic answer."


@pytest.mark.parametrize("field,invalid", [("mode", {}), ("providers", [[]])])
def test_invalid_run_metadata_fails_as_a_gateway_error(workspace, field, invalid):
    _window, view, gateway = workspace
    run = prepare(view, gateway)
    run["request"][field] = invalid
    with pytest.raises(GatewayError, match="metadata"):
        checked_run(run)


def test_saved_native_session_restores_only_the_same_gateway_runs(workspace, tmp_path):
    window, view, gateway = workspace
    run = prepare(view, gateway)
    gateway.runs[run["run_id"]]["state"] = "paused"
    settings = QSettings(window.settings.fileName(), QSettings.Format.IniFormat)
    restored = Desktop(gateway, settings, tmp_path / "restored")
    try:
        assert restored.matrix.session_id == view.session_id
        restored.show_page(restored.matrix_index)
        restored.show()
        until(lambda: restored.matrix.current is not None)
        assert restored.matrix.current["state"] == "paused"
        assert (
            sum(name == "connect_agent_matrix" for name, _args, _thread in gateway.matrix_calls)
            == 1
        )
    finally:
        restored.close()
        assert restored.pool.waitForDone(5000)
    other_gateway = MatrixGateway()
    other_gateway.home = tmp_path / "another-gateway"
    other = Desktop(other_gateway, window.settings, tmp_path / "other")
    try:
        assert other.matrix.session_id is None and not other_gateway.matrix_calls
    finally:
        other.close()
        assert other.pool.waitForDone(5000)


def test_ui_bounds_and_stop_match_the_gateway_contract(workspace):
    _window, view, gateway = workspace
    run = prepare(view, gateway)
    assert view.token_budget.maximum() == 1000000
    assert view.seconds.maximum() == 28800
    gateway.runs[run["run_id"]]["state"] = "pause_requested"
    view.refresh.click()
    until(lambda: view.current["state"] == "pause_requested")
    assert view.stop.isEnabled()
    view.stop.click()
    until(lambda: view.current["state"] == "stopped" and not view.busy)


def test_other_principal_requests_require_explicit_local_owner_review(workspace):
    _window, view, gateway = workspace
    native = prepare(view, gateway)
    external = copy.deepcopy(native)
    external["run_id"] = "external-agent-run"
    external["request"]["question"] = "A synthetic request from another client."
    gateway.external_runs[external["run_id"]] = external
    view.refresh.click()
    until(lambda: "matrix-list" not in view.desktop.pending)
    assert view.run_list.count() == 1 and not gateway.owner_calls
    view.review_agents.click()
    until(lambda: view.run_list.count() == 2)
    view.run_list.setCurrentRow(1)
    until(lambda: view.current["run_id"] == external["run_id"])
    assert view.summary.text().startswith("Owner review")
    assert not view.prepare.isEnabled()
    view.approve.click()
    assert external["request"]["question"] in view.approval_dialog.request_text.toPlainText()
    view.approval_dialog.start.click()
    until(lambda: view.current["state"] == "queued" and not view.busy)
    approval = next(args for tool, args, _thread in gateway.owner_calls if tool == "matrix_approve")
    assert approval == {
        "run_id": external["run_id"],
        "expected_request_sha256": external["request_sha256"],
    }
    view.pause.click()
    until(lambda: view.current["state"] == "paused" and not view.busy)
    assert any(tool == "matrix_control" for tool, _args, _thread in gateway.owner_calls)
    assert all("session_id" not in args for _tool, args, _thread in gateway.owner_calls)
    view.review_agents.click()
    until(lambda: view.run_list.count() == 1 and view.current is not None)
    assert view.current["run_id"] == native["run_id"]
    assert not view.summary.text().startswith("Owner review")
