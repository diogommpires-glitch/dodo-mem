from dodo_memory_desktop.whole_index import Catalogue
from dodo_memory_desktop.whole_index_worker import Indexer


def test_overview_draws_a_direct_link_between_deep_unsampled_files(tmp_path):
    root = tmp_path / "library"
    folder = root / "Project"
    nested = folder / "a" / "b" / "c" / "d"
    nested.mkdir(parents=True)
    origin = nested / "00.md"
    target = nested / "zz-reference.md"
    origin.write_text("# Plan\n[Reference](zz-reference.md)\n", encoding="utf-8")
    target.write_text("# Reference\nA verified destination.\n", encoding="utf-8")
    for number in range(50):
        (folder / f"{number:02}.txt").write_text(f"Unrelated {number}", encoding="utf-8")
    db = tmp_path / "catalogue.sqlite3"
    indexer = Indexer(db)
    reader = Catalogue(db)
    try:
        indexer.reconcile([root], extract_now=False)
        start = reader.resolve_path(origin)["id"]
        end = reader.resolve_path(target)["id"]
        assert not {start, end} & {node["id"] for node in reader.graph_projection()["nodes"]}
        indexer.reconcile([root])
        overview = reader.graph_projection()
        link = next(edge for edge in overview["edges"] if edge["kind"] == "reference")
        assert (link["source"], link["target"]) == (start, end)
        assert {start, end} <= {node["id"] for node in overview["nodes"]}
        assert overview["counts"]["bundled"] == 0
        assert reader.graph_evidence(link)[0]["source"]["locator"]["start"] == 2
        target.unlink()
        assert reader.graph_evidence(link) == []
        indexer.refresh_paths({target}, [root])
        assert all(edge["kind"] != "reference" for edge in reader.graph_projection()["edges"])
    finally:
        indexer.close()


def test_graph_connects_inventory_and_rechecks_exact_bundle_evidence(tmp_path):
    root = tmp_path / "library"
    alpha = root / "Alpha"
    nested = alpha / "z-nested"
    deeper = nested / "deeper" / "last"
    beta = root / "Beta"
    deeper.mkdir(parents=True)
    beta.mkdir()
    for number in range(50):
        (alpha / f"{number:02}.txt").write_text(f"Unique text {number}", encoding="utf-8")
    origin = deeper / "note.md"
    target = beta / "report.md"
    origin.write_text(
        "# Shared project heading\n[report](../../../../Beta/report.md)\n", encoding="utf-8"
    )
    target.write_text(
        "# Shared project heading\n[note](../Alpha/z-nested/deeper/last/note.md)\n",
        encoding="utf-8",
    )
    database = tmp_path / "catalogue.sqlite3"
    indexer = Indexer(database)
    catalogue = Catalogue(database)
    try:
        indexer.reconcile([root], extract_now=False)
        overview = catalogue.graph_projection()
        alpha_id = catalogue.resolve_path(alpha)["id"]
        beta_id = catalogue.resolve_path(beta)["id"]
        nested_id = catalogue.resolve_path(nested)["id"]
        deeper_id = catalogue.resolve_path(deeper)["id"]
        origin_id = catalogue.resolve_path(origin)["id"]
        target_id = catalogue.resolve_path(target)["id"]
        assert overview["hubs"] == [alpha_id, beta_id]
        assert overview["counts"] == {"nodes": 58, "containment": 57, "bundled": 0}
        assert overview["edges"]
        assert all(edge["kind"] == "contains" for edge in overview["edges"])
        overview_ids = {row["id"] for row in overview["nodes"]}
        assert len(overview_ids) < overview["counts"]["nodes"]
        assert deeper_id in overview_ids and origin_id not in overview_ids

        expanded = catalogue.graph_projection(expanded=(nested_id,))
        assert expanded["hubs"] == overview["hubs"]
        assert overview_ids <= {row["id"] for row in expanded["nodes"]}
        assert any(
            edge["source"] == deeper_id and edge["target"] == origin_id
            for edge in expanded["edges"]
        )
        assert next(row for row in expanded["nodes"] if row["id"] == origin_id)["cluster_id"] == (
            alpha_id
        )
        focused = catalogue.graph_projection(origin_id)
        assert focused["hubs"] == [origin_id]
        assert focused["counts"]["nodes"] == 1

        indexer.reconcile([root])
        links = catalogue.graph_projection()
        assert links["counts"]["bundled"] == 2
        assert not any(edge["kind"] == "topic" for edge in links["edges"])
        assert any(
            edge["kind"] == "topic"
            for edge in catalogue.graph_projection(include_inferred=True)["edges"]
        )
        references = [edge for edge in links["edges"] if edge["kind"] == "reference"]
        bundles = [edge for edge in references if edge.get("bundle")]
        assert len(bundles) == 1 and bundles[0]["count"] == 2
        assert {
            (edge["source"], edge["target"]) for edge in references if not edge.get("bundle")
        } == {
            (origin_id, target_id),
            (target_id, origin_id),
        }
        bundle = bundles[0]
        evidence = catalogue.graph_evidence(bundle)
        assert {row["node"]["id"] for row in evidence} == {origin_id, target_id}
        assert all(row["source"]["locator"]["start"] == 2 for row in evidence)
        incoming = next(
            edge
            for edge in catalogue.neighbours(target_id)
            if edge["kind"] == "reference" and edge["source"] == origin_id
        )
        assert catalogue.graph_evidence(incoming)[0]["source"]["path"] == str(origin)

        origin.unlink()
        assert catalogue.graph_evidence(bundle) == []
        indexer.reconcile([root])
        assert not any(
            edge["kind"] == "reference" for edge in catalogue.graph_projection()["edges"]
        )
    finally:
        indexer.close()
