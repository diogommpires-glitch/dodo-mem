import pytest

from dodo_memory.chunks import section_chunks
from dodo_memory.store import Vault


def test_indexed_tables_keep_units_headers_and_complete_cells(tmp_path):
    vault = Vault(tmp_path)
    path = vault.root / "documents/table.md"
    path.parent.mkdir(parents=True)
    rows = [f"| specimen-{index:04} | {index / 10:.1f} | stable |" for index in range(300)]
    header = "| Specimen | Force (N) | Status |\n| --- | ---: | --- |"
    original = (
        "# Paper\n\n## Measurements\n\nTable 3. Force in N, measured at 10 ms.\n\n"
        + header
        + "\n"
        + "\n".join(rows)
        + "\n"
    )
    path.write_text(original, encoding="utf-8")
    original_bytes = path.read_bytes()
    vault.index()
    hits = vault.search("specimen 0299")
    assert hits
    for hit in hits:
        if "| specimen-" in hit["body"]:
            assert header in hit["body"]
            assert "Force in N, measured at 10 ms" in hit["body"]
            assert hit["heading"] == "Measurements"
    with vault.connect() as db:
        chunks = [
            row[0] for row in db.execute("SELECT body FROM chunks WHERE path='documents/table.md'")
        ]
    for row in rows:
        assert any(row in chunk for chunk in chunks)
    assert path.read_bytes() == original_bytes


def test_oversized_table_segments_repeat_headers_without_losing_or_splitting_rows():
    header = "| ID | Mass (kg) |\n| --- | ---: |"
    rows = [f"| specimen-{index} | {index}.25 |" for index in range(300)]
    chunks = list(
        section_chunks(
            "## Results\n\n" + header + "\n" + "\n".join(rows), "Paper", target=200, maximum=1000
        )
    )
    assert len(chunks) > 1
    assert all(
        heading == "Results" and len(body) <= 1000 and header in body for heading, body in chunks
    )
    recovered = [
        line for _, body in chunks for line in body.splitlines() if line.startswith("| specimen-")
    ]
    assert recovered == rows
    assert all("the complete table remains in the canonical source" in body for _, body in chunks)


def test_headings_inside_code_fences_do_not_change_section_and_paragraphs_stay_whole():
    fence = chr(96) * 3
    paragraph = "Sentence with explicit units m/s. " * 40
    source = f"# Main\n\n{paragraph}\n\n{fence}python\n# Not a document heading\nvalue = 7\n{fence}\n\n## Next\n\nFinal paragraph."
    chunks = list(section_chunks(source, "Fallback", target=300))
    assert any(paragraph in body for heading, body in chunks if heading == "Main")
    assert any("# Not a document heading" in body and heading == "Main" for heading, body in chunks)
    assert chunks[-1] == ("Next", "Final paragraph.")


def test_unrepresentable_table_cell_is_explicitly_refused_instead_of_silent_cut():
    source = "| ID | Value |\n| --- | --- |\n| sample | " + "x" * 1000 + " |"
    with pytest.raises(ValueError, match="never split cells"):
        list(section_chunks(source, "Paper", target=100, maximum=500))


def test_one_column_table_does_not_silently_split_a_single_cell():
    source = "| Value |\n| --- |\n| " + "x" * 1000 + " |"
    with pytest.raises(ValueError, match="never split cells"):
        list(section_chunks(source, "Paper", target=100, maximum=500))


def test_mixed_fence_characters_cannot_close_code_and_drop_its_headings():
    fence = chr(96) * 3
    false_close = chr(96) + "~~"
    source = f"# Actual\n\n{fence}python\n{false_close}\n# after\nvalue = 7\n{fence}"
    chunks = list(section_chunks(source, "Fallback"))
    assert all(heading == "Actual" for heading, _ in chunks)
    assert any("# after" in body for _, body in chunks)


def test_large_code_file_is_indexed_in_complete_line_fragments_without_losing_headings(tmp_path):
    vault = Vault(tmp_path)
    path = vault.root / "documents/large-code.md"
    path.parent.mkdir(parents=True)
    fence = chr(96) * 3
    rows = [f"line_{index:04} = " + repr("scientific value " * 5) for index in range(700)]
    rows.insert(350, "# This is a code comment, not a document section")
    source = "# Large code\n\n" + fence + "python\n" + "\n".join(rows) + "\n" + fence + "\n"
    path.write_text(source, encoding="utf-8")
    original = path.read_bytes()
    vault.index()
    with vault.connect() as db:
        chunks = db.execute(
            "SELECT heading,body FROM chunks WHERE path='documents/large-code.md' ORDER BY rowid"
        ).fetchall()
    assert len(chunks) > 1
    assert all(row["heading"] == "Large code" and len(row["body"]) <= 48000 for row in chunks)
    recovered = [
        line
        for row in chunks
        for line in row["body"].splitlines()
        if line.startswith(("line_", "# This is a code"))
    ]
    assert recovered == rows
    assert path.read_bytes() == original
