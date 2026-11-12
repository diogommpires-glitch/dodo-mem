"""Keep uncertain figure captions separate from registered original page/region routes."""

import math
import re


def _box(value):
    if not isinstance(value, dict) or value.get("coord_origin") != "BOTTOMLEFT":
        return None
    numbers = [value.get(key) for key in ("l", "b", "r", "t")]
    if any(type(number) not in (int, float) or not math.isfinite(number) for number in numbers):
        return None
    return numbers if numbers[0] < numbers[2] and numbers[1] < numbers[3] else None


def _near(figure, text):
    caption = _box(text.get("bbox"))
    region = _box(figure.get("bbox"))
    if caption is None or region is None or text.get("page_no") != figure.get("page_no"):
        return False
    overlap = max(0, min(caption[2], region[2]) - max(caption[0], region[0]))
    gap = min(abs(caption[3] - region[1]), abs(caption[1] - region[3]))
    return overlap / min(caption[2] - caption[0], region[2] - region[0]) >= 0.25 and gap <= 72


def _original(provenance, source_sha256, geometry):
    page = provenance.get("page_no")
    if type(page) is not int or page < 1:
        raise ValueError("Figure provenance requires a positive source page")
    box = _box(provenance.get("bbox"))
    info = geometry.get(page, geometry.get(str(page), {}))
    size = info.get("size")
    ordinary_frame = (
        isinstance(size, (list, tuple))
        and len(size) == 2
        and all(
            type(value) in (int, float) and math.isfinite(value) and value > 0 for value in size
        )
        and info.get("rotation") == 0
        and info.get("media_box") == [0, 0, *size]
        and info.get("crop_box") == info.get("media_box")
    )
    validated = (
        ordinary_frame
        and box is not None
        and 0 <= box[0] < box[2] <= size[0]
        and 0 <= box[1] < box[3] <= size[1]
    )
    return {
        "source_sha256": source_sha256,
        "page": page,
        "bbox": box if validated else None,
        "coordinate_origin": "BOTTOMLEFT",
        "unit": "pdf_points" if validated else None,
        "region_status": "validated_coordinate_frame"
        if validated
        else "unverified_coordinate_transform",
        "region_content_verified": False,
    }


def figure_manifest(document, source_sha256, *, page_geometry=None, exclusions=None):
    """Enumerate every figure; unresolved captions expose references and an original route, never a caption fact."""
    if not isinstance(source_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", source_sha256):
        raise ValueError("Figure evidence requires the exact registered original SHA-256")
    pictures, texts = document.get("pictures"), document.get("texts")
    if (
        not isinstance(pictures, list)
        or len(pictures) > 2000
        or not isinstance(texts, list)
        or len(texts) > 100_000
    ):
        raise ValueError("Figure evidence inventory exceeds its bounded document shape")
    if any(not isinstance(value, dict) for value in (*pictures, *texts)):
        raise ValueError("Figure and caption inventories require structured records")
    geometry = {} if page_geometry is None else page_geometry
    exclusions = {} if exclusions is None else exclusions
    if not isinstance(geometry, dict) or not isinstance(exclusions, dict):
        raise TypeError("Figure geometry and explicit review exclusions must be dictionaries")
    identities = {picture.get("self_ref") for picture in pictures}
    if any(
        key not in identities or not isinstance(value, str) or not 1 <= len(value) <= 160
        for key, value in exclusions.items()
    ):
        raise ValueError(
            "Caption exclusions require an existing figure and a bounded review reason"
        )
    caption_pages = {}
    for text in texts:
        if text.get("label") == "caption":
            for provenance in text.get("prov", []):
                caption_pages.setdefault(provenance.get("page_no"), []).append((text, provenance))
    output = []
    counts = dict.fromkeys(("eligible", "processed", "failed", "excluded"), 0)
    for picture in pictures:
        identity = picture.get("self_ref")
        provenance = picture.get("prov")
        if (
            not isinstance(identity, str)
            or not re.fullmatch(r"#/pictures/[0-9]+", identity)
            or not isinstance(provenance, list)
            or not 1 <= len(provenance) <= 100
        ):
            raise ValueError(
                "Figure evidence requires bounded source-page provenance and identities"
            )
        originals = [_original(value, source_sha256, geometry) for value in provenance]
        uncertainty = []
        if any(_box(value.get("bbox")) is None for value in provenance):
            uncertainty.append("invalid_derived_region")
        if any(value["bbox"] is None for value in originals):
            uncertainty.append("use_original_page_until_coordinate_transform_is_verified")
        declared, clean = [], []
        for reference in picture.get("captions", []):
            ref = reference.get("$ref") if isinstance(reference, dict) else None
            if not isinstance(ref, str) or not re.fullmatch(r"#/texts/[0-9]+", ref):
                uncertainty.append("invalid_caption_reference")
                continue
            declared.append(ref)
            index = int(ref.rsplit("/", 1)[1])
            if index >= len(texts) or texts[index].get("self_ref") != ref:
                uncertainty.append("missing_caption_reference")
                continue
            text = texts[index]
            caption_text = text.get("text")
            if not isinstance(caption_text, str) or not re.match(
                r"^(?:Figure|Fig\.?|Figura)\s+[0-9]+\b", caption_text
            ):
                uncertainty.append("caption_text_precedes_or_lacks_figure_marker")
                continue
            if not any(
                _near(region, candidate)
                for region in provenance
                for candidate in text.get("prov", [])
            ):
                uncertainty.append("caption_source_region_is_not_near_figure")
                continue
            clean.append(ref)
        candidates = set(declared)
        for region in provenance:
            for text, candidate in caption_pages.get(region["page_no"], []):
                if _near(region, candidate) and isinstance(text.get("self_ref"), str):
                    candidates.add(text["self_ref"])
        complete = bool(clean) and len(clean) == len(declared) and not uncertainty
        if identity in exclusions:
            status, reason = "excluded", exclusions[identity]
        elif complete:
            status, reason = (
                "processed",
                "structurally_linked_caption_requires_original_transcription_review",
            )
        else:
            status, reason = "failed", "caption_missing_or_contaminated_use_registered_original"
        counts["eligible"] += int(status != "excluded")
        counts[status] += 1
        output.append(
            {
                "figure_ref": identity,
                "status": status,
                "reason": reason,
                "caption_status": "linked_unverified_transcription"
                if complete
                else "missing_or_contaminated",
                "caption_refs": clean if complete and status != "excluded" else [],
                "unverified_caption_refs": sorted(candidates - set(clean))
                if complete
                else sorted(candidates),
                "caption_verified_against_original": False,
                "original": originals[0],
                "additional_original_regions": originals[1:],
                "derived_regions": [
                    {
                        "page": value["page_no"],
                        "bbox": _box(value.get("bbox")),
                        "coordinate_origin": "BOTTOMLEFT" if _box(value.get("bbox")) else "unknown",
                    }
                    for value in provenance
                ],
                "uncertainty": sorted(set(uncertainty)),
            }
        )
    return {
        "schema_version": 1,
        "source_sha256": source_sha256,
        "figures": output,
        "counts": counts,
        "complete": counts["failed"] == 0,
        "fidelity_verified_against_original": False,
        "processing": "structural figure-caption coverage; original transcription remains unverified",
    }
