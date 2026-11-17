"""Connect the private catalogue to the desktop without changing shared-vault routes."""

from functools import partial

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QTabWidget, QVBoxLayout, QWidget

from .ask_dodo import AskDodo
from .host_plugins import HostPlugins
from .neural_activity import NeuralActivity
from .neural_client import NeuralClient
from .neural_status import NeuralStatus
from .ui import button, label
from .whole_graph import WholeLibraryGraph
from .whole_library import FOLDERS, WholeLibrary


class WholeWorkspace:
    def __init__(self, desktop, catalogue, pages, indices):
        self.desktop = desktop
        self.catalogue = catalogue
        (
            self.library_index,
            self.graph_index,
            self.canvas_index,
            self.plugins_index,
            self.ask_index,
        ) = indices
        self.selected = None
        self.ask_revision = None
        self.library = WholeLibrary(catalogue)
        self.neural = NeuralClient()
        self.neural_status = NeuralStatus(self.neural)
        self.graph = WholeLibraryGraph(catalogue, neural=self.neural)
        self.network = NeuralActivity(self.neural)
        self.network.view_mode.setCurrentIndex(self.network.view_mode.findData("overview"))
        self.ask = AskDodo(catalogue, profile=desktop.personal_profile)
        self.plugins = HostPlugins(catalogue, desktop.request)
        self.library_tabs = self._tabs(
            self.library, "Whole library", desktop.library, "Shared vault"
        )
        self.library_tabs.currentChanged.connect(self.refresh)
        self.graph_tabs = self._tabs(self.graph, "Whole library", desktop.graph, "Shared vault")
        self.graph_tabs.setTabText(0, "Library relationships")
        self.graph_tabs.insertTab(0, self.network, "E5 network")
        self.graph_tabs.setCurrentWidget(self.network)
        self.graph_tabs.currentChanged.connect(desktop._refresh_graph_tab)
        self.graph_page = QWidget()
        graph_layout = QVBoxLayout(self.graph_page)
        graph_layout.setContentsMargins(0, 0, 0, 0)
        graph_layout.addWidget(self.neural_status)
        graph_layout.addWidget(self.graph_tabs, 1)
        generated = QWidget()
        panel = QVBoxLayout(generated)
        panel.setContentsMargins(24, 22, 24, 22)
        panel.addWidget(label("Generated Canvas", "title"))
        self.generated_scope = label("Whole library", "section")
        panel.addWidget(self.generated_scope)
        preview = button("Open live Canvas", role="primary")
        preview.clicked.connect(self.show_canvas)
        panel.addWidget(preview)
        panel.addStretch()
        self.canvas_tabs = self._tabs(
            desktop.canvas, "Editable boards", generated, "Generated from the library"
        )
        self.plugins_tabs = self._tabs(
            self.plugins, "Dodo Kit and host add-ins", desktop.plugins, "Dodo add-ins"
        )
        pages[self.library_index] = self.library_tabs
        pages[self.graph_index] = self.graph_page
        pages[self.canvas_index] = self.canvas_tabs
        pages[self.plugins_index] = self.plugins_tabs
        pages.append(self.ask)
        self.library.source_requested.connect(self.open_source)
        self.library.selection_changed.connect(self.select)
        self.library.graph_requested.connect(self.show_graph)
        self.library.map_requested.connect(self.show_map)
        self.library.canvas_requested.connect(self.show_canvas)
        self.library.ask_requested.connect(self.show_ask)
        self.graph.source_requested.connect(self.open_source)
        self.graph.scope_changed.connect(self._select_graph_node)
        self.graph.canvas_requested.connect(self.save_canvas)
        self.plugins.skill_requested.connect(self.use_skill)
        self.plugins.pack_requested.connect(self.install_pack)
        self.timer = QTimer(desktop, interval=3000)
        self.timer.timeout.connect(self.refresh)
        self.timer.start()

    @staticmethod
    def _tabs(first, first_name, second, second_name):
        tabs = QTabWidget()
        tabs.addTab(first, first_name)
        tabs.addTab(second, second_name)
        return tabs

    def show_library(self):
        self.library_tabs.setCurrentWidget(self.library)
        self.desktop.show_page(self.library_index)

    def refresh(self, _index=None):
        self.neural_status.refresh()
        index = self.desktop.stack.currentIndex()
        if index == self.library_index and self.library_tabs.currentWidget() is self.library:
            self.library.refresh()
        graph_visible = index == self.graph_index and self.graph_tabs.currentWidget() is self.graph
        self.network.set_active(
            index == self.graph_index
            and self.graph_tabs.currentWidget() is self.network
            and self.desktop.isVisible()
        )
        if graph_visible:
            self.graph.refresh()
        self.graph.refresh_previews(graph_visible=graph_visible)
        if index == self.plugins_index and self.plugins_tabs.currentWidget() is self.plugins:
            self.plugins.refresh_packages()
        if index == self.ask_index:
            self.library.reads.request(
                "ask-revision",
                self.catalogue.revision,
                self._ask_revision_ready,
                self.ask.invalidate_evidence,
            )

    def _ask_revision_ready(self, revision):
        if revision != self.ask_revision:
            self.ask_revision = revision
            self.ask.refresh_visible()

    def open_source(self, node):
        self.select(node)
        self.show_library()
        self.library.open_source(node)

    def open_node(self, node):
        if node["kind"] in FOLDERS:
            self.select(node)
            self.show_library()
            self.library.open_folder(node)
        else:
            self.open_source(node)

    def project_scope(self):
        node = self.selected
        if node:
            return node.get("project_id") or (
                node["id"] if node["kind"] in FOLDERS else node.get("parent_id")
            )
        return None

    def search_home(self, query, *, project):
        home = self.desktop.home
        scope = self.project_scope() if project else None
        if project and scope is None:
            home.result_title.setText("Select a project first")
            return
        home.result_title.setText("Searching project…" if project else "Searching this PC…")
        self.library.reads.request(
            "home-search",
            partial(self.catalogue.search, query, 30, scope=scope),
            home.set_local_results,
            lambda: home.result_title.setText("Local search is unavailable"),
            replace=True,
        )

    def _select_graph_node(self, node):
        if (
            self.desktop.stack.currentIndex() == self.graph_index
            and self.graph_tabs.currentWidget() is self.graph
        ):
            self.select(node)

    def select(self, node):
        if isinstance(node, dict) and node.get("id"):
            self.selected = node
            location = str(node.get("path") or node["name"])
            self.generated_scope.setText(location)
            self.desktop.home.project_context.setText(f"Project around {location}")
            self.ask.set_context(node_id=node["id"], node_name=node["name"])

    def show_graph(self, node=None):
        self.graph_tabs.setCurrentWidget(self.graph)
        if isinstance(node, dict):
            self.graph.show_node(node)
            self.select(node)
        self.desktop.show_page(self.graph_index)

    def show_map(self, node=None):
        self.show_graph(node)
        self.graph.show_mind_map()

    def show_canvas(self, node=None):
        self.show_graph(node if isinstance(node, dict) else self.selected)
        self.graph.show_generated_canvas()

    def show_ask(self, node=None):
        self.select(node)
        self.desktop.show_page(self.ask_index)

    def use_skill(self, title, markdown):
        self.ask.set_skill(title, markdown)
        self.show_ask()

    def install_pack(self, path):
        self.desktop.plugins.install_path(path)
        self.plugins_tabs.setCurrentWidget(self.desktop.plugins)

    def save_canvas(self, board):
        try:
            self.desktop.canvas.create_board(board.get("title", "Library Canvas"), board)
        except (OSError, ValueError) as error:
            self.desktop.status.setText(str(error))
            return
        self.canvas_tabs.setCurrentWidget(self.desktop.canvas)
        self.desktop.show_page(self.canvas_index)
