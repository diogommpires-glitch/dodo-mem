"""Synthetic workspace used for keyboard checks and screenshot review."""

from copy import deepcopy

from dodo_memory_desktop.gateway import GatewayError

PATH = "notes/lab/thermal-calibration.md"
SENSOR = "notes/lab/sensor-placement.md"
METHOD = "notes/lab/chamber-protocol.md"
ISOLATED = "notes/independent.md"
DIGEST = "b" * 64
NOTE = """# Thermal calibration

Calibration notes for the compact chamber and its four thermocouple channels.
The reference probe stays in the centre of the sample plane.

## Offset and drift

At each set point, wait until the reference changes by less than 0.05 °C per minute.
Take the median of the final 60 seconds before moving to the next set point.

| Set point | Mean offset | Repeatability |
| --- | --- | --- |
| 20 °C | +0.12 °C | 0.04 °C |
| 35 °C | +0.19 °C | 0.06 °C |
| 50 °C | +0.26 °C | 0.08 °C |

Keep the raw readings and record which probe was used. Apply a correction only
within the measured interval.

## Repeating the run

See [Sensor placement](sensor-placement.md) before mounting the next batch.
Use the [chamber protocol](chamber-protocol.md) to reproduce the timing.
Read [[Chamber#Method|the chamber procedure]] for the original method.

Log the time of each change in chamber set point and keep the same sampling rate
across channels.

## Units

Report temperatures in degrees Celsius, offsets in kelvin, and time in seconds.
Check the probe against the ice reference before repeating this run.
"""
ITEMS = [
    {
        "path": path,
        "title": title,
        "collection": collection,
        "project": project,
        "kind": "note",
        "sha256": DIGEST,
    }
    for path, title, collection, project in [
        (PATH, "Thermal calibration", "Lab notebook", "Chamber study"),
        (SENSOR, "Sensor placement", "Lab notebook", "Chamber study"),
        (METHOD, "Chamber protocol", "Field references", None),
        (ISOLATED, "Independent note", "Field references", None),
    ]
]
PROPOSAL = {
    "proposal_id": "fixture-review",
    "entry_key": "Chamber settling criterion",
    "expected_revision": 1,
    "status": "pending_review",
    "text": "Wait for drift below 0.05 °C per minute before starting the sampling window. "
    "Record the set point, probe ID, and duration.",
    "evidence": [{"path": PATH, "heading": "Offset and drift", "sha256": DIGEST}],
}


class FixtureGateway:
    def __init__(self):
        self.connected = True
        self.approved = False
        self.revoked = False
        self.saved = None

    def call(self, name, arguments=None):
        if not self.connected:
            raise GatewayError("Memory is offline.", connection_lost=True)
        args = arguments or {}
        items = [item for item in ITEMS if not self.revoked or item["path"] != PATH]
        match name:
            case "browse_notes":
                return {
                    "items": deepcopy(items),
                    "next_cursor": None,
                    "projects": ["Chamber study"],
                    "collections": ["Lab notebook", "Field references"],
                }
            case "search":
                if self.revoked:
                    return []
                return [
                    {
                        **items[0],
                        "heading": "Offset and drift",
                        "body": "Reference offsets and the measured chamber drift.",
                    }
                ]
            case "get_note":
                item = next((entry for entry in items if entry["path"] == args["path"]), None)
                if item is None:
                    raise GatewayError("This note is no longer available.")
                text = NOTE if item["path"] == PATH else f"# {item['title']}\n\n## Method"
                if self.saved and self.saved[0] == item["path"]:
                    text = self.saved[1]
                return {
                    **item,
                    "text": text,
                    "source": f"sha256:{DIGEST}",
                    "status": "indexed",
                    "editable": item["path"] == METHOD,
                }
            case "read_note_for_edit":
                return self.call("get_note", args)
            case "suggest_note_links":
                return {
                    "items": [
                        {**item, "heading": "## Method"} for item in items if item["path"] == METHOD
                    ]
                    if args["query"] == "Chamber#Meth"
                    else []
                }
            case "note_card_refs":
                return {
                    "items": [
                        {key: item[key] for key in ("path", "title", "sha256")}
                        for item in items
                        if item["path"] in args["paths"]
                    ]
                }
            case "resolve_note_link":
                return {
                    "status": "ambiguous",
                    "choices": [
                        {**item, "heading": "Method"}
                        for item in items
                        if item["path"] in (SENSOR, METHOD)
                    ],
                }
            case "ordinary_note_history":
                return {"records": [], "truncated": False}
            case "related_notes" | "graph_snapshot":
                edges = []
                if not self.revoked and (name == "graph_snapshot" or args["path"] == PATH):
                    edges = [
                        {
                            "source": PATH,
                            "target": target,
                            "kind": "link",
                            "label": "Repeating the run",
                            "evidence": {"path": PATH, "heading": "Repeating the run"},
                        }
                        for target in (SENSOR, METHOD)
                    ]
                return {
                    "nodes": deepcopy(items),
                    "edges": edges,
                    "count": len(edges),
                    "truncated": False,
                }
            case "open_source":
                if self.revoked:
                    raise GatewayError("Original unavailable or changed.")
                assert args["preview"] is True
                return {
                    "sha256": DIGEST,
                    "locator": {"kind": "lines", "start": 8, "end": 13},
                    "preview_text": "Set point, offset, repeatability\n20 °C, +0.12 °C, 0.04 °C\n"
                    "35 °C, +0.19 °C, 0.06 °C\n50 °C, +0.26 °C, 0.08 °C",
                }
            case "list_knowledge_proposals":
                return [] if self.approved else [deepcopy(PROPOSAL)]
            case "read_knowledge_proposal":
                return deepcopy(PROPOSAL)
            case "read_working_knowledge":
                return {
                    "revision": 1,
                    "text": "Allow the chamber reading to stabilise before sampling.",
                }
            case "working_knowledge_history":
                return {"records": [], "truncated": False}
            case _:
                raise AssertionError(name)

    def approve(self, proposal_id, expected_revision, review_note):
        assert (proposal_id, expected_revision) == ("fixture-review", 1)
        assert review_note
        self.approved = True

    def save_note(self, path, text, expected_sha256):
        assert path == METHOD and expected_sha256 == DIGEST
        self.saved = (path, text, expected_sha256)
