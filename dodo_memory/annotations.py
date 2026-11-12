"""Extract PDF annotations as attributed source claims, separate from author text."""

import hashlib
import json
import math
import re
from collections import Counter
from io import BytesIO

from pypdf import PdfReader, __version__

from .parsing import ParsedDocument

PARSER = "pdf-annotations-v1"
MAX_PDF_BYTES = 100 * 1024**2
MAX_PAGES = 2000
MAX_ANNOTATIONS = 2000
MAX_TEXT_CHARS = 16384
MAX_TOTAL_TEXT_BYTES = 2 * 1024**2
MAX_OUTPUT_BYTES = 4 * 1024**2
SUPPORTED = {"/Text", "/Highlight", "/Underline", "/Squiggly", "/StrikeOut", "/FreeText"}
CITATION_LABEL = "PDF annotation (claimed authorship; unverified)"


class NoAnnotationsFound(ValueError):
    """The retained PDF has no supported annotation records."""


class AnnotationFormatError(ValueError):
    """The retained annotation structure does not match the PDF schema."""


def _number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    try:
        number = float(value)
    except OverflowError:
        return None
    return number if math.isfinite(number) else None


def _rectangle(value):
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    numbers = [_number(item) for item in value]
    if any(number is None for number in numbers):
        return None
    left, bottom, right, top = numbers
    return numbers if left < right and bottom < top else None


def _inside(rectangle, box):
    return (
        rectangle[0] >= box[0]
        and rectangle[1] >= box[1]
        and rectangle[2] <= box[2]
        and rectangle[3] <= box[3]
    )


def _claimed_text(value, name, uncertainty, limit=MAX_TEXT_CHARS):
    if value is None:
        return None
    if not isinstance(value, str):
        uncertainty.append(name + "_not_text")
        return None
    if len(value) > limit:
        raise ValueError(f"PDF annotation {name} exceeds its {limit}-character limit")
    return value


def parse_pdf_annotations(original, *, cancelled=lambda: False):
    """Read bounded comments/highlights without executing actions or changing the PDF."""
    if not 0 < original.stat().st_size <= MAX_PDF_BYTES:
        raise ValueError("Annotation PDF must contain 1 byte to 100 MiB")
    if cancelled():
        raise RuntimeError("Annotation parsing cancelled")
    with original.open("rb") as stream:
        source_bytes = stream.read(MAX_PDF_BYTES + 1)
    if not 0 < len(source_bytes) <= MAX_PDF_BYTES:
        raise ValueError("Annotation PDF must contain 1 byte to 100 MiB")
    reader = PdfReader(BytesIO(source_bytes), strict=True)
    if reader.is_encrypted:
        raise ValueError(
            "Encrypted annotation PDFs require an explicitly decrypted retained source"
        )
    if len(reader.pages) > MAX_PAGES:
        raise ValueError("Annotation PDF exceeds the 2000-page limit")
    annotations, excluded, examined, text_bytes = [], Counter(), 0, 0
    for page_number, page in enumerate(reader.pages, 1):
        if cancelled():
            raise RuntimeError("Annotation parsing cancelled")
        objects = page.get("/Annots", [])
        objects = objects.get_object() if hasattr(objects, "get_object") else objects
        if not isinstance(objects, (list, tuple)):
            raise AnnotationFormatError("PDF page annotation inventory is not an array")
        examined += len(objects)
        if examined > MAX_ANNOTATIONS:
            raise ValueError("PDF annotation inventory exceeds the 2000-record limit")
        page_box = _rectangle(page.get("/MediaBox"))
        user_unit = _number(page.get("/UserUnit", 1))
        user_unit = user_unit if user_unit is not None and user_unit > 0 else None
        rotation = _number(page.get("/Rotate", 0))
        rotation = int(rotation) if rotation is not None and rotation % 90 == 0 else None
        for position, reference in enumerate(objects, 1):
            if cancelled():
                raise RuntimeError("Annotation parsing cancelled")
            annotation = reference.get_object() if hasattr(reference, "get_object") else reference
            if not isinstance(annotation, dict):
                raise AnnotationFormatError("PDF annotation record is not a dictionary")
            subtype = annotation.get("/Subtype")
            if not isinstance(subtype, str) or subtype not in SUPPORTED:
                excluded["unsupported_subtype"] += 1
                continue
            uncertainty = []
            if page_box is None:
                uncertainty.append("invalid_page_box")
            if user_unit is None:
                uncertainty.append("invalid_user_unit")
            if rotation is None:
                uncertainty.append("invalid_page_rotation")
            claimed_rect = annotation.get("/Rect")
            rectangle = _rectangle(claimed_rect)
            if rectangle is None:
                uncertainty.append("invalid_annotation_rect")
            elif page_box is None or not _inside(rectangle, page_box):
                rectangle = None
                uncertainty.append("annotation_rect_outside_known_page")
            quad_points = annotation.get("/QuadPoints")
            if quad_points is not None:
                if (
                    not isinstance(quad_points, (list, tuple))
                    or not 0 < len(quad_points) <= 256
                    or len(quad_points) % 8
                ):
                    quad_points = None
                    uncertainty.append("invalid_quad_points")
                else:
                    quad_points = [_number(value) for value in quad_points]
                    if any(value is None for value in quad_points):
                        quad_points = None
                        uncertainty.append("nonfinite_quad_points")
                    elif page_box is None or any(
                        not page_box[0] <= quad_points[index] <= page_box[2]
                        or not page_box[1] <= quad_points[index + 1] <= page_box[3]
                        for index in range(0, len(quad_points), 2)
                    ):
                        quad_points = None
                        uncertainty.append("quad_points_outside_known_page")
            contents = _claimed_text(annotation.get("/Contents"), "contents", uncertainty)
            author = _claimed_text(annotation.get("/T"), "claimed_author", uncertainty, 512)
            modified = _claimed_text(annotation.get("/M"), "claimed_modified_at", uncertainty, 256)
            text_bytes += sum(
                len(value.encode("utf-8")) for value in (contents, author, modified) if value
            )
            if text_bytes > MAX_TOTAL_TEXT_BYTES:
                raise ValueError("PDF annotation text exceeds the 2 MiB total limit")
            annotations.append(
                {
                    "id": f"page-{page_number}-annotation-{position}",
                    "page": page_number,
                    "subtype": str(subtype).removeprefix("/"),
                    "text_origin": "pdf_annotation",
                    "contents": contents,
                    "claimed_author": author,
                    "author_verified": False,
                    "claimed_modified_at": modified,
                    "rect": rectangle,
                    "claimed_rect": [str(value) for value in claimed_rect]
                    if isinstance(claimed_rect, (list, tuple)) and len(claimed_rect) <= 16
                    else None,
                    "coordinate_space": "pdf_user_space_bottom_left",
                    "unit": "pdf_user_unit",
                    "user_unit_scale_points": user_unit,
                    "page_box": page_box,
                    "rotation": rotation,
                    "quad_points": quad_points,
                    "uncertainty": uncertainty,
                }
            )
    if not annotations:
        raise NoAnnotationsFound("No supported PDF annotations were found")
    data = {
        "schema_version": 1,
        "parser": PARSER,
        "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "annotations": annotations,
        "examined_annotations": examined,
        "excluded": dict(excluded),
    }
    structured = json.dumps(data, ensure_ascii=False, allow_nan=False).encode("utf-8")
    parts = [
        "# PDF annotations",
        "Annotation contents are unverified source claims, separate from paper-author text.",
    ]
    for annotation in annotations:
        contents = (
            annotation["contents"] or "(No comment text extracted; annotation region retained.)"
        )
        longest = max((len(run) for run in re.findall(chr(96) + "+", contents)), default=0)
        fence = chr(96) * max(3, longest + 1)
        parts.extend(
            [
                f"## {annotation['id']}: {annotation['subtype']} on page {annotation['page']}",
                f"Claimed author (unverified): {json.dumps(annotation['claimed_author'])}",
                f"Uncertainty: {', '.join(annotation['uncertainty']) or 'none recorded'}",
                f"{fence}text\n{contents}\n{fence}",
            ]
        )
    text = "\n\n".join(parts) + "\n"
    if len(structured) + len(text.encode("utf-8")) > MAX_OUTPUT_BYTES:
        raise ValueError("PDF annotation output exceeds the 4 MiB limit")
    return ParsedDocument(
        text,
        {
            "parser": PARSER,
            "pypdf_version": __version__,
            "text_origin": "pdf_annotation",
            "citation_label": CITATION_LABEL,
            "annotation_count": len(annotations),
            "uncertain_annotations": sum(bool(item["uncertainty"]) for item in annotations),
            "source_sha256": data["source_sha256"],
        },
        {"annotations.json": structured},
    )
