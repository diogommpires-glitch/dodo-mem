from PySide6.QtWidgets import QApplication

from dodo_memory_desktop import host_plugins
from dodo_memory_desktop.plugin_ui import Plugins
from dodo_memory_desktop.plugins import EXAMPLES, PluginStore
from dodo_memory_desktop.whole_index import Catalogue
from dodo_memory_desktop.whole_index_worker import Indexer


def test_host_skill_is_local_guidance_only_after_selection(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    cache = tmp_path / "cache"
    base = cache / "catalogue" / "sample" / "1.0.0"
    manifest = base / ".claude-plugin" / "plugin.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text('{"name":"Sample"}', encoding="utf-8")
    skill = base / "skills" / "sample-skill" / "SKILL.md"
    skill.parent.mkdir(parents=True)
    skill.write_text("# Sample guidance", encoding="utf-8")
    discover = host_plugins.discover
    monkeypatch.setattr(
        host_plugins, "discover", lambda: discover([("Claude Code", cache, ".claude-plugin")])
    )

    browser = host_plugins.HostPlugins()
    chosen = []
    browser.skill_requested.connect(lambda title, body: chosen.append((title, body)))
    browser.refresh()
    assert not browser.use.isEnabled() and not chosen
    browser.tree.setCurrentItem(browser.tree.topLevelItem(0).child(0))
    assert browser.use.isEnabled() and not chosen
    browser.use.click()
    assert chosen == [("sample-skill", "# Sample guidance")]
    browser.close()
    app.processEvents()


def test_indexed_pack_requires_an_install_action_and_remains_disabled(tmp_path, monkeypatch):
    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(host_plugins, "discover", list)
    root = tmp_path / "library"
    root.mkdir()
    bundled = EXAMPLES / "research-starter-1.0.0.dodo-pack.zip"
    pack = root / bundled.name
    pack.write_bytes(bundled.read_bytes())
    database = tmp_path / "index.sqlite3"
    worker = Indexer(database)
    try:
        worker.reconcile([root])
    finally:
        worker.close()

    def request(_lane, read, done, _failed=None):
        done(read())

    browser = host_plugins.HostPlugins(Catalogue(database), request)
    installed = Plugins(PluginStore(tmp_path / "installed"))
    browser.pack_requested.connect(installed.install_path)
    browser.refresh()
    assert not installed.store.versions()

    item = browser.tree.topLevelItem(0).child(0)
    browser.tree.setCurrentItem(item)
    browser.install.click()
    versions = installed.store.versions()
    assert len(versions) == 1 and not versions[0].enabled
    assert installed.enable.isEnabled()
    browser.close()
    installed.close()
    app.processEvents()
