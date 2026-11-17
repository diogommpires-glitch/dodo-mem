import os

import pytest
from PySide6.QtCore import QCoreApplication, QEvent, QThreadPool
from PySide6.QtWidgets import QApplication

from dodo_memory_desktop.neural_activity import NeuralActivity


@pytest.fixture(scope="session", autouse=True)
def application():
    """Drain asynchronous widget reads before the test application's teardown."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    app.setQuitOnLastWindowClosed(False)
    yield app
    for widget in app.allWidgets():
        if isinstance(widget, NeuralActivity):
            assert widget.shutdown()
    assert QThreadPool.globalInstance().waitForDone(5000)
    for widget in app.topLevelWidgets():
        # Native popup windows still belong to their combo box, menu or owner.
        if widget.parent() is None:
            widget.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    app.processEvents()
    assert not app.allWidgets()
