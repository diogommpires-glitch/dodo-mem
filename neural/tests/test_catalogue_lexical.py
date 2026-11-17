"""The compact cache preserves lexical retrieval without duplicating ordinary text."""

import hashlib
import sqlite3
from types import SimpleNamespace

from dodo_memory_desktop.whole_index import SCHEMA, Catalogue

from dodo_mem_neural.catalogue_adapter import AuthorisedCatalogue


def test_natural_question_uses_existing_fts_and_excludes_protected_or_offline_sources(tmp_path):
    database = tmp_path / "catalogue.sqlite3"
    db = sqlite3.connect(database)
    db.executescript(SCHEMA)
    db.execute(
        """INSERT INTO nodes(id,root_id,path,path_key,name,kind,category,state,size,mtime_ns,file_id)
        VALUES ('root','root',?,?,'fixture','root','document','online',0,0,'root')""",
        (str(tmp_path), str(tmp_path)),
    )
    for name in ("ordinary.md", "credentials.txt"):
        path = tmp_path / name
        text = "Comet acceleration uses this supporting evidence."
        path.write_text(text)
        info = path.stat()
        db.execute(
            """INSERT INTO nodes(id,parent_id,root_id,path,path_key,name,kind,category,state,digest,size,mtime_ns,file_id)
            VALUES (?,'root','root',?,?,?,'file','document','indexed',?,?,?,?)""",
            (
                name,
                str(path),
                str(path),
                name,
                hashlib.sha256(path.read_bytes()).hexdigest(),
                info.st_size,
                info.st_mtime_ns,
                f"{info.st_dev}:{info.st_ino}",
            ),
        )
        db.execute(
            "INSERT INTO passages(node_id,kind,start,end,text) VALUES (?,'lines',1,1,?)",
            (name, text),
        )
    db.commit()
    catalogue = Catalogue(database)
    adapter = AuthorisedCatalogue(
        catalogue, SimpleNamespace(root=tmp_path / "shared", search=lambda *a, **kw: [])
    )
    question = "Which reference explains the comet acceleration?"
    assert catalogue.search(question) == []
    assert [row["id"] for row in adapter.search(question, scope="full_private")] == ["ordinary.md"]
    assert adapter.read_evidence("ordinary.md")["text"].startswith("Comet acceleration")
    assert adapter.search(question, scope="shared") == []
    (tmp_path / "ordinary.md").write_text("Changed evidence.")
    assert adapter.read_evidence("ordinary.md") is None
    db.execute("UPDATE nodes SET state='offline' WHERE id='root'")
    db.commit()
    assert adapter.search(question, scope="full_private") == []
    db.close()
