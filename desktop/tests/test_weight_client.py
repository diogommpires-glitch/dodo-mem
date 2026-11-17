import copy
import json
import threading
from concurrent.futures import ThreadPoolExecutor

import httpx
import pytest

from dodo_memory_desktop.neural_client import NeuralClient, NeuralError


def descriptor():
    return {
        "block": 11,
        "projection": "ff_down",
        "input_dim": 1536,
        "output_dim": 384,
        "total_weight_count": 1536 * 384,
        "orientation": "output_input",
        "source": {"operator_id": "900", "operation": "gelu", "semantics": "GELU output"},
        "target": {"operator_id": "911", "operation": "dense", "semantics": "Before residual"},
        "provenance": {
            "prepared_manifest_sha256": "a" * 64,
            "xml_sha256": "b" * 64,
            "bin_sha256": "c" * 64,
            "bin_verified": True,
        },
    }


def weight_response():
    return {
        **descriptor(),
        "schema": 1,
        "static": True,
        "dtype": "f16",
        "model_revision": "pinned",
        "model_space": "e5-qualified",
        "edges": [{"source": 23, "target": 7, "weight": -0.25}],
        "shown_count": 1,
        "sampling": "top_absolute_incoming_weight",
        "output_index": 7,
    }


def client_for(tmp_path, receive):
    (tmp_path / "broker.token").write_text("x" * 48)
    return NeuralClient(tmp_path, transport=httpx.MockTransport(receive))


def test_static_weight_request_preserves_real_matrix_coordinates_and_selection(tmp_path):
    seen = []

    def receive(request):
        seen.append(request)
        return httpx.Response(200, json=weight_response())

    result = client_for(tmp_path, receive).activity_weights(
        11, "ff_down", limit=256, output_index=7
    )
    assert result["edges"] == [{"source": 23, "target": 7, "weight": -0.25}]
    assert dict(seen[0].url.params) == {
        "block": "11",
        "projection": "ff_down",
        "limit": "256",
        "output_index": "7",
    }


@pytest.mark.parametrize(
    "invalid", ["coordinates", "finite", "proof", "axes", "count", "selection"]
)
def test_static_weight_client_rejects_misleading_or_unverified_edges(tmp_path, invalid):
    value = weight_response()
    if invalid == "coordinates":
        value["edges"][0]["source"] = 1536
    elif invalid == "finite":
        value["edges"][0]["weight"] = float("inf")
    elif invalid == "proof":
        value["provenance"]["bin_verified"] = False
    elif invalid == "axes":
        value["orientation"] = "input_output"
    elif invalid == "count":
        value["shown_count"] = 3
    else:
        value["edges"][0]["target"] = 8
    client = client_for(
        tmp_path,
        lambda _: httpx.Response(
            200, content=json.dumps(value), headers={"Content-Type": "application/json"}
        ),
    )
    with pytest.raises(NeuralError):
        client.activity_weights(11, "ff_down", output_index=7)


def test_metadata_requires_all_seventy_two_unique_e5_projections(tmp_path):
    value = weight_response()
    partial = {
        key: value[key] for key in ("schema", "static", "dtype", "model_revision", "model_space")
    }
    partial["projections"] = [copy.deepcopy(descriptor())] * 72
    client = client_for(tmp_path, lambda _: httpx.Response(200, json=partial))
    with pytest.raises(NeuralError, match="projection"):
        client.activity_projections()


def test_overview_requires_declared_output_stratification(tmp_path):
    value = weight_response()
    value.update(sampling="stratified_by_output", output_index=None)
    client = client_for(tmp_path, lambda _: httpx.Response(200, json=value))
    assert client.activity_weights(11, "ff_down")["edges"] == value["edges"]
    value["sampling"] = "top_absolute_weight"
    with pytest.raises(NeuralError, match="selection"):
        client.activity_weights(11, "ff_down")


def test_parameter_resource_reason_remains_visible_without_retry(tmp_path):
    calls = []

    def receive(request):
        calls.append(request)
        return httpx.Response(
            503, json={"detail": "Background work paused: less than 5 GiB available RAM."}
        )

    client = client_for(tmp_path, receive)
    with pytest.raises(NeuralError, match="5 GiB"):
        client.activity_weights(11, "ff_down")
    assert len(calls) == 1


def test_shared_weight_reads_are_serialised_across_view_workers(tmp_path):
    entered, release, second_started = threading.Event(), threading.Event(), threading.Event()
    guard = threading.Lock()
    active = maximum = 0
    arrivals = []

    def receive(_request):
        nonlocal active, maximum
        with guard:
            active += 1
            maximum = max(maximum, active)
            arrivals.append(1)
            first = len(arrivals) == 1
        try:
            if first:
                entered.set()
                assert release.wait(2)
            return httpx.Response(200, json=weight_response())
        finally:
            with guard:
                active -= 1

    client = client_for(tmp_path, receive)

    def detail_read():
        second_started.set()
        return client.activity_weights(11, "ff_down", output_index=7)

    with ThreadPoolExecutor(max_workers=2) as workers:
        overview = workers.submit(client.activity_weights, 11, "ff_down", output_index=7)
        assert entered.wait(1)
        detail = workers.submit(detail_read)
        assert second_started.wait(1)
        try:
            detail_done_before_release = threading.Event()
            detail.add_done_callback(lambda _: detail_done_before_release.set())
            assert not detail_done_before_release.wait(0.05)
        finally:
            release.set()
        assert overview.result()["shown_count"] == detail.result()["shown_count"] == 1
    assert maximum == 1 and len(arrivals) == 2
