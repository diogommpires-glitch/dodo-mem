"""Perfect source recall cannot hide an activity latency or capture regression."""

from scripts.qualify_retrieval import activity_gates


def test_activity_gate_rejects_slow_or_missing_measurements_despite_perfect_retrieval():
    totals = {"baseline": 24, "neural_off": 24, "neural_on": 24}
    spaces = dict.fromkeys(totals, "pinned-e5-space")
    captures = [
        {
            "captured": True,
            "model_space": "pinned-e5-space",
            "level": "final",
            "source": "interactive",
            "token_is_padding": False,
            "dimensions": {"block.11": 384, "pooled": 384, "output": 384},
        }
        for _ in range(24)
    ]
    gates = activity_gates(
        totals, {"neural_off": 0.1, "neural_on": 0.104}, captures, spaces, 24, 24
    )
    assert all(gate["passed"] for gate in gates.values())
    slow = activity_gates(totals, {"neural_off": 0.1, "neural_on": 0.106}, captures, spaces, 24, 24)
    assert slow["quality"]["passed"] is True and slow["latency"]["passed"] is False
    captures[-1]["captured"] = False
    missing = activity_gates(
        totals, {"neural_off": 0.1, "neural_on": 0.104}, captures, spaces, 23, 24
    )
    assert missing["actual_captures"]["observed"] == 23
    assert missing["actual_captures"]["passed"] is False
