import json
from pathlib import Path

import pytest
from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from dodo_memory_desktop.canvas import Canvas
from dodo_memory_desktop.canvas_store import read_board


@pytest.fixture
def canvas(tmp_path):
    app = QApplication.instance() or QApplication([])
    board = Canvas(tmp_path / "boards")
    board.resize(1160, 760)
    board.show()
    app.processEvents()
    yield board
    board.close()
    app.processEvents()


def test_sample_has_image_and_url_and_exports_portable_asset(canvas, tmp_path):
    assert canvas.board["title"] == "Philosophy notebook"
    assert canvas.boards.currentText() == "Philosophy notebook"
    assert {
        node["text"].splitlines()[0] for node in canvas.board["nodes"] if node["type"] == "text"
    } == {
        "Rationalists",
        "Empiricists",
        "Cogito, ergo sum",
    }
    assert len(canvas.connections) == 2
    assert any(not item.pixmap.isNull() for item in canvas.cards.values())
    assert any(
        node.get("url") == "https://plato.stanford.edu/entries/descartes-epistemology/"
        for node in canvas.board["nodes"]
    )
    portable = tmp_path / "Philosophy.canvas"
    canvas.export_board(portable)
    raw = json.loads(portable.read_bytes())
    assert set(raw) == {"nodes", "edges"}
    image = next(node["file"] for node in raw["nodes"] if node["type"] == "file")
    assert (portable.parent / image).is_file()

    imported = canvas.import_board(portable)
    board, _digest = read_board(imported)
    original = read_board(portable)[0]
    for incoming, recreated in zip(original["nodes"], board["nodes"], strict=True):
        assert [incoming[key] for key in ("id", "x", "y", "width", "height")] == [
            recreated[key] for key in ("id", "x", "y", "width", "height")
        ]
    assert original["edges"] == board["edges"]


def test_add_example_preserves_existing_canvas_and_switches_selector(canvas):
    original = canvas.current
    original_bytes = original.read_bytes()
    canvas._add_example()
    assert canvas.current != original
    assert original.read_bytes() == original_bytes
    assert canvas.boards.currentText() == "Philosophy example"
    canvas.boards.setCurrentIndex(canvas.boards.findData(str(original)))
    assert canvas.current == original


def test_move_connect_undo_and_reopen_canvas(canvas):
    path = canvas.create_board("Reading")
    canvas.add_note("notes/Philosophy.md", "Virtue")
    canvas.add_url("https://example.test/reference")
    first, second = [node["id"] for node in canvas.board["nodes"]]
    canvas.add_connection(first, second, "Compare")
    canvas.fit()
    QApplication.processEvents()
    item = canvas.cards[second]
    start = canvas.view.mapFromScene(item.sceneBoundingRect().center())
    viewport = canvas.view.viewport()
    before = canvas.board["nodes"][1]["x"]
    QTest.mousePress(viewport, Qt.MouseButton.LeftButton, pos=start)
    QTest.mouseMove(viewport, start + QPoint(58, 0))
    QTest.mouseRelease(viewport, Qt.MouseButton.LeftButton, pos=start + QPoint(58, 0))
    QApplication.processEvents()
    after = canvas.board["nodes"][1]["x"]
    assert after > before
    canvas.undo.undo()
    assert canvas.board["nodes"][1]["x"] == before
    canvas.undo.redo()
    assert canvas.board["nodes"][1]["x"] == after
    assert canvas.save()

    reopened = Canvas(canvas.storage)
    reopened.open_board(path)
    assert reopened.board["nodes"][1]["x"] == after
    assert reopened.board["edges"][0]["label"] == "Compare"
    reopened.close()


def test_import_rejects_traversal_without_replacing_current_canvas(canvas, tmp_path):
    original = canvas.current
    original_bytes = original.read_bytes()
    malformed = tmp_path / "bad.canvas"
    malicious_card = {
        "id": "image",
        "type": "file",
        "file": "../secret.png",
        "x": 0,
        "y": 0,
        "width": 200,
        "height": 100,
    }
    malformed.write_text(json.dumps({"nodes": [malicious_card]}), encoding="utf-8")
    with pytest.raises(ValueError, match="relative paths"):
        canvas.import_board(malformed)
    assert canvas.current == original
    assert original.read_bytes() == original_bytes
    assert list(canvas.storage.glob("*.canvas")) == [original]


def test_dragging_group_moves_contained_cards_and_resize_keeps_their_positions(canvas):
    group, *cards = canvas.board["nodes"]
    graphic = canvas.cards[group["id"]]
    viewport = canvas.view.viewport()
    anchor = canvas.view.mapFromScene(graphic.mapToScene(QPointF(22, 20)))
    destination = anchor + QPoint(32, 14)
    QTest.mousePress(viewport, Qt.MouseButton.LeftButton, pos=anchor)
    QTest.mouseMove(viewport, destination)
    QTest.mouseRelease(viewport, Qt.MouseButton.LeftButton, pos=destination)
    QApplication.processEvents()
    moved, *moved_cards = canvas.board["nodes"]
    dx = moved["x"] - group["x"]
    dy = moved["y"] - group["y"]
    assert dx > 0 and dy > 0
    assert [(card["x"], card["y"]) for card in moved_cards] == [
        (card["x"] + dx, card["y"] + dy) for card in cards
    ]
    graphic = canvas.cards[group["id"]]
    handle = graphic.mapToScene(graphic.rect().bottomRight() - QPointF(6, 6))
    anchor = canvas.view.mapFromScene(handle)
    destination = anchor + QPoint(22, 18)
    QTest.mousePress(viewport, Qt.MouseButton.LeftButton, pos=anchor)
    QTest.mouseMove(viewport, destination)
    QTest.mouseRelease(viewport, Qt.MouseButton.LeftButton, pos=destination)
    QApplication.processEvents()
    resized, *same_cards = canvas.board["nodes"]
    assert resized["width"] > moved["width"]
    assert resized["height"] > moved["height"]
    assert same_cards == moved_cards


def test_note_cards_require_fresh_gateway_result_for_open_and_export(canvas, tmp_path):
    canvas.create_board("Reading")
    opened = []
    requests = []
    canvas.note_requested.connect(lambda path, heading: opened.append((path, heading)))
    canvas.note_refs_requested.connect(requests.append)
    note = "notes/Private name.md"
    canvas.add_note(note, "Evidence")
    assert requests == [[note]]
    canvas.open_selected()
    assert opened == []
    assert canvas.card_list.item(0).text() == "Unavailable note"
    canvas.set_note_cards({note: {"title": "Authorised title"}})
    canvas.open_selected()
    assert opened == [(note, "Evidence")]

    canvas.clear_note_cards()
    path = Path(tmp_path) / "Reading.canvas"
    canvas.export_board(path)
    exported = json.loads(path.read_bytes())
    assert exported["nodes"][0]["text"] == "Unavailable note"
    assert "file" not in exported["nodes"][0]
