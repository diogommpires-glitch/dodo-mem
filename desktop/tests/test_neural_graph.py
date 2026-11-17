import threading
import time

import httpx
import pytest
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from dodo_memory_desktop.neural_client import NeuralClient, NeuralError
from dodo_memory_desktop.neural_status import NeuralStatus
from dodo_memory_desktop.whole_graph import INFERRED_KINDS, WholeLibraryGraph


def until(predicate):
    deadline = time.monotonic() + 4
    while not predicate() and time.monotonic() < deadline:
        QApplication.processEvents()
        QTest.qWait(10)
    assert predicate()


def test_neural_client_authenticates_locally_and_rejects_redirects(tmp_path):
    token = "x" * 48
    (tmp_path / "broker.token").write_text(token, encoding="utf-8")
    requests = []

    def receive(request):
        requests.append(request)
        if request.url.path == "/topics":
            return httpx.Response(200, json={"generation": 3, "topics": []})
        return httpx.Response(302, headers={"Location": "https://example.invalid/private"})

    client = NeuralClient(tmp_path, transport=httpx.MockTransport(receive))
    assert client.topics() == {"generation": 3, "topics": []}
    assert requests[0].headers["Authorization"] == f"Bearer {token}"
    assert str(requests[0].url) == "http://127.0.0.1:8770/topics"
    with pytest.raises(NeuralError, match="service error"):
        client.status()
    assert len(requests) == 2


@pytest.mark.parametrize("grouped", [True, False])
def test_neural_client_accepts_unscored_groups_and_only_real_edge_weights(tmp_path, grouped):
    (tmp_path / "broker.token").write_text("x" * 48, encoding="utf-8")
    value = {"edges": [{"source": "a", "target": "b", "weight": 0.81}]}
    if grouped:
        value["topics"] = [
            {"id": "t", "label": "Representative source", "members": [{"node_id": "a"}]}
        ]
    else:
        value["members"] = [{"source_id": "a", "community_id": "t"}]
    client = NeuralClient(
        tmp_path, transport=httpx.MockTransport(lambda _: httpx.Response(200, json=value))
    )
    result = client.topics()
    assert result["topics"][0]["members"] == [{"node_id": "a"}]
    assert result["edges"] == value["edges"]


class Catalogue:
    def __init__(self):
        self.nodes = {
            "a": {"id": "a", "name": "Decision", "kind": "file", "category": "document"},
            "b": {"id": "b", "name": "Report", "kind": "file", "category": "document"},
        }

    def status(self):
        return {"generation": 2, "nodes": len(self.nodes)}

    def node(self, identity):
        return self.nodes.get(identity)

    def neighbours(self, *_args, **_kwargs):
        return []

    def projection(self, _scope, **_options):
        return {"generation": 2, "nodes": list(self.nodes.values()), "edges": [], "total": 2}


class Neural:
    def __init__(self):
        self.worker_threads = []

    def topics(self):
        self.worker_threads.append(threading.get_ident())
        return {
            "generation": 7,
            "topics": [
                {
                    "id": "calibration",
                    "label": "Calibration",
                    "members": [
                        {"node_id": "a", "score": 0.92},
                        {"node_id": "b", "score": 0.81},
                        {"node_id": "revoked", "score": 0.9},
                    ],
                }
            ],
            "edges": [
                {
                    "source": "a",
                    "target": "b",
                    "score": 0.81,
                    "evidence": "Mutual nearest neighbours from current source centroids",
                }
            ],
        }


def test_topics_are_real_current_members_and_refresh_keeps_positions():
    neural = Neural()
    graph = WholeLibraryGraph(Catalogue(), neural=neural)
    graph.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
    graph.show()
    graph.grouping.setCurrentIndex(1)
    until(lambda: graph.diagram is not None)
    assert graph.inferred.isChecked()
    assert {node["name"] for node in graph.diagram.nodes.values()} == {
        "Calibration",
        "Decision",
        "Report",
    }
    assert {edge["kind"] for edge in graph.diagram.edges} <= INFERRED_KINDS
    assert len(graph.diagram.edges) == 1
    assert "2 current topic members" in graph.summary.text()
    assert "Dashed" in graph.legend.text()
    assert threading.get_ident() not in neural.worker_threads
    graph._select("a")
    saved = graph.diagram.positions["a"]
    graph.refresh()
    until(lambda: not graph._running)
    assert graph.diagram.positions["a"] == saved
    graph._inspect_edge(graph.diagram.edges[0])
    assert "Similarity" in graph.detail.text()
    assert "0.81" in graph.relationships.item(0).text()
    graph.close()


def test_projects_overlay_similarity_and_remove_it_when_disabled():
    graph = WholeLibraryGraph(Catalogue(), neural=Neural())
    graph.refresh()
    until(lambda: graph.diagram is not None)
    assert len(graph.diagram.edges) == 1
    assert graph.diagram.edges[0]["kind"] == "semantic"
    graph.inferred.setChecked(False)
    until(lambda: not graph._running)
    assert graph.diagram.edges == []
    graph.close()


def test_shared_topic_membership_does_not_fabricate_similarity_edges():
    class Unconnected(Neural):
        def topics(self):
            return {**super().topics(), "edges": []}

    graph = WholeLibraryGraph(Catalogue(), neural=Unconnected())
    graph.refresh()
    until(lambda: graph.diagram is not None)
    assert graph.diagram.edges == []
    graph.grouping.setCurrentIndex(1)
    until(lambda: graph.mode == "topics" and not graph._running)
    assert len(graph.diagram.nodes) == 3
    assert graph.diagram.edges == []
    graph.close()


def test_source_focus_selects_only_after_the_new_projection_arrives():
    catalogue = Catalogue()
    graph = WholeLibraryGraph(catalogue)
    graph.refresh()
    until(lambda: graph.diagram is not None)
    graph.show_node(catalogue.nodes["a"])
    assert graph.selection is None
    until(lambda: graph.selection == "a")
    assert graph.diagram.scope["id"] == "a"
    graph.close()


def test_neural_status_shows_degradation_without_blocking_catalogue():
    class Offline:
        def status(self):
            raise NeuralError("Local neural service is unavailable.")

    widget = NeuralStatus(Offline())
    widget.refresh()
    until(lambda: "unavailable" in widget.summary.text())
    assert "Catalogue graph remains available" in widget.summary.text()
    widget.close()
