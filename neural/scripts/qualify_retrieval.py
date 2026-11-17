"""Calculate offline qualification gates; live model evaluation is excluded."""


def activity_gates(totals, p95, captures, spaces, retained, total):
    off, on = p95["neural_off"], p95["neural_on"]
    dimensions = {"block.11": 384, "pooled": 384, "output": 384}
    space = spaces["neural_off"]
    observed = sum(item.get("captured", False) for item in captures)
    return {
        "quality": {"passed": total == 24 and all(value == total for value in totals.values())},
        "latency": {
            "passed": off > 0 and on <= off * 1.05,
            "off_p95_seconds": off,
            "on_p95_seconds": on,
            "ratio": on / off if off > 0 else None,
            "maximum_ratio": 1.05,
        },
        "embedding_space": {
            "passed": bool(space)
            and all(value == space for value in spaces.values())
            and all(item.get("model_space") == space for item in captures),
            "space": space,
        },
        "actual_captures": {
            "passed": retained == total == len(captures) == observed
            and all(
                item.get("dimensions") == dimensions
                and item.get("level") == "final"
                and item.get("source") == "interactive"
                and item.get("token_is_padding") is False
                for item in captures
            ),
            "observed": observed,
            "retained": retained,
            "expected": total,
            "expected_dimensions": dimensions,
        },
    }
