"""Validate reported quantities without converting units or inventing device identity."""

import math
import re

from .source_registry import single_line

MEMORY_UNITS = frozenset({"B", "kB", "MB", "GB", "TB", "KiB", "MiB", "GiB", "TiB"})
UNIT_SYMBOLS = MEMORY_UNITS | frozenset(
    {
        "1",
        "%",
        "ppm",
        "m",
        "cm",
        "mm",
        "um",
        "µm",
        "nm",
        "km",
        "kg",
        "g",
        "mg",
        "s",
        "ms",
        "us",
        "µs",
        "min",
        "h",
        "A",
        "K",
        "mol",
        "cd",
        "Hz",
        "N",
        "Pa",
        "J",
        "W",
        "C",
        "V",
        "F",
        "Ω",
        "ohm",
        "S",
        "Wb",
        "T",
        "H",
        "lm",
        "lx",
        "Bq",
        "Gy",
        "Sv",
        "kat",
        "rad",
        "sr",
        "eV",
        "L",
        "mL",
        "token",
        "tokens",
        "sample",
        "samples",
    }
)


def _unit(value):
    single_line(value, "quantity unit", 80)
    if any(not part for part in re.split(r"[*/]", value)):
        raise ValueError("Quantity unit must use explicit symbols and products or quotients")
    for part in re.split(r"[*/]", value):
        match = re.fullmatch(r"([^\^]+)(?:\^(-?\d{1,2}))?", part)
        if (
            match is None
            or match[1] not in UNIT_SYMBOLS
            or (match[2] and not -12 <= int(match[2]) <= 12)
        ):
            raise ValueError("Unsupported quantity unit symbol or exponent; no unit is inferred")
    return value


def validate_quantities(values):
    if not isinstance(values, list) or len(values) > 64:
        raise ValueError("Quantities must be a list of at most 64 explicit records")
    checked = []
    for record in values:
        if not isinstance(record, dict) or set(record) != {
            "name",
            "kind",
            "value",
            "unit",
            "device",
        }:
            raise ValueError("Quantity needs name, kind, value, unit and device")
        single_line(record["name"], "quantity name", 160)
        if record["kind"] not in {"system_ram", "gpu_vram", "scientific"}:
            raise ValueError("Quantity kind must distinguish system_ram, gpu_vram or scientific")
        value = record["value"]
        if value is not None and (
            type(value) not in {int, float}
            or abs(value) > 10**308
            or (type(value) is float and not math.isfinite(value))
        ):
            raise ValueError(
                "Quantity value must be finite numeric data or explicit null for unknown"
            )
        unit = _unit(record["unit"])
        device = record["device"]
        if not isinstance(device, dict) or set(device) != {"kind", "name"}:
            raise ValueError("Quantity device needs explicit kind and name")
        if device["kind"] not in {"system", "gpu", "instrument", "unknown"}:
            raise ValueError("Unsupported quantity device kind")
        if device["kind"] == "unknown":
            if device["name"] is not None:
                raise ValueError("Unknown device cannot imply a named identity")
        else:
            single_line(device["name"], "quantity device name", 160)
        if record["kind"] in {"system_ram", "gpu_vram"}:
            expected = "system" if record["kind"] == "system_ram" else "gpu"
            if (
                unit not in MEMORY_UNITS
                or device["kind"] not in {expected, "unknown"}
                or (value is not None and value < 0)
            ):
                raise ValueError(
                    "RAM and VRAM need memory units and their correct distinct device kind"
                )
        checked.append({**record, "device": dict(device)})
    return checked
