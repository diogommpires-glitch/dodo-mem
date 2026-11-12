import hashlib

import pytest

from dodo_memory.quantities import validate_quantities
from dodo_memory.review import AcceptedKnowledge
from dodo_memory.store import Vault


def quantity(kind="system_ram", value=64, unit="GiB", device="system"):
    return {
        "name": "reported capacity",
        "kind": kind,
        "value": value,
        "unit": unit,
        "device": {"kind": device, "name": "Synthetic workstation"},
    }


def test_ram_vram_and_decimal_binary_units_remain_distinct():
    records = [
        quantity(),
        quantity("gpu_vram", 24, "GB", "gpu"),
        quantity("scientific", -9.81, "m/s^2", "instrument"),
    ]
    assert validate_quantities(records) == records
    unknown = {**quantity(value=None), "device": {"kind": "unknown", "name": None}}
    assert validate_quantities([unknown]) == [unknown]


@pytest.mark.parametrize(
    "record",
    [
        quantity("system_ram", 64, "GiB", "gpu"),
        quantity("gpu_vram", 24, "GiB", "system"),
        quantity(value=True),
        quantity(value=float("nan")),
        quantity(value=float("inf")),
        quantity(value=-1),
        quantity(unit="GiB/s"),
        quantity(unit="gib"),
        quantity(unit="unknown"),
        {**quantity(), "device": {"kind": "unknown", "name": "invented identity"}},
        {**quantity(), "unit": None},
        {**quantity(), "kind": "memory"},
    ],
)
def test_ambiguous_or_mismatched_quantities_are_rejected(record):
    with pytest.raises((ValueError, TypeError)):
        validate_quantities([record])


def test_typed_quantities_survive_explicit_review_restart_and_history(tmp_path):
    vault = Vault(tmp_path)
    source = vault.root / "documents/evidence.md"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"# Observation\nSynthetic RAM and VRAM capacity observation.\n")
    evidence = [
        {
            "path": "documents/evidence.md",
            "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "heading": "# Observation",
        }
    ]
    knowledge = AcceptedKnowledge(vault)
    records = [quantity(), quantity("gpu_vram", 24, "GB", "gpu")]
    proposal = knowledge.propose(
        "synthetic-capacity",
        "Synthetic capacity facts.",
        evidence=evidence,
        expected_revision=0,
        writer="fixture",
        quantities=records,
        assertion_type="verified_observation",
    )
    assert knowledge.get_proposal(proposal["proposal_id"])["quantities"] == records
    accepted = knowledge.accept(
        proposal["proposal_id"],
        actor_role="owner",
        reviewed_by="fixture-owner",
        review_note="Exact units and devices verified in synthetic evidence.",
        expected_revision=0,
    )
    assert accepted["quantities"] == records
    reopened = AcceptedKnowledge(Vault(tmp_path))
    assert reopened.read("synthetic-capacity")["quantities"] == records
    assert reopened.history("synthetic-capacity")["records"][0]["quantities"] == records
