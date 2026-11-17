import numpy as np
import pytest

from dodo_mem_neural.resources import ResourcePressure, enforce_limits
from dodo_mem_neural.runtime import _submit, normalise, pool_embeddings


def test_masked_pooling_ignores_padding_and_normalises():
    hidden = np.array([[[3, 0], [0, 4], [999, 999]]], dtype=np.float32)
    result = pool_embeddings(hidden, np.array([[1, 1, 0]]), dimensions=2)
    np.testing.assert_allclose(result, [[0.6, 0.8]], atol=1e-6)
    with pytest.raises(ValueError, match="zero"):
        normalise(np.zeros((1, 2)))


def test_resource_gate_includes_pending_allocation_and_never_evicts():
    gib = 1024**3
    enforce_limits(
        available_ram=6 * gib, free_disk=102 * gib, cache_bytes=18 * gib, additional_disk=gib
    )
    with pytest.raises(ResourcePressure, match="quota"):
        enforce_limits(
            available_ram=6 * gib, free_disk=102 * gib, cache_bytes=20 * gib, additional_disk=1
        )
    with pytest.raises(ResourcePressure, match="RAM"):
        enforce_limits(available_ram=4 * gib, free_disk=102 * gib, cache_bytes=0)


def test_inference_queue_survives_operation_failure():
    def unavailable():
        raise OSError("disconnected")

    with pytest.raises(OSError, match="disconnected"):
        _submit(unavailable)
    assert _submit(lambda: 42) == 42
