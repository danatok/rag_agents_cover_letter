from src import ingest


def make_pdf_bytes(text: str) -> bytes:
    """Build a minimal single-page PDF containing the given text."""
    content = f"BT /F1 24 Tf 72 700 Td ({text}) Tj ET".encode("latin-1")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /Resources << /Font << /F1 4 0 R >> >> "
        b"/MediaBox [0 0 612 792] /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length %d >>\nstream\n" % len(content) + content + b"\nendstream",
    ]
    pdf = b"%PDF-1.4\n"
    offsets = []
    for i, obj in enumerate(objects, start=1):
        offsets.append(len(pdf))
        pdf += b"%d 0 obj\n" % i + obj + b"\nendobj\n"
    xref_offset = len(pdf)
    pdf += b"xref\n0 %d\n" % (len(objects) + 1)
    pdf += b"0000000000 65535 f \n"
    for off in offsets:
        pdf += b"%010d 00000 n \n" % off
    pdf += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF" % (
        len(objects) + 1,
        xref_offset,
    )
    return pdf


def test_load_documents_reads_known_file_with_correct_metadata(tmp_path, monkeypatch):
    (tmp_path / "cv.txt").write_text("Experienced engineer.", encoding="utf-8")
    monkeypatch.setattr(ingest, "DATA_DIR", tmp_path)

    docs = ingest.load_documents()

    assert len(docs) == 1
    doc = docs[0]
    assert doc.page_content == "Experienced engineer."
    assert doc.metadata == {"source": "cv.txt", "type": "cv", "skills": ""}


def test_load_documents_falls_back_to_other_for_unmapped_file(tmp_path, monkeypatch):
    (tmp_path / "random_notes.txt").write_text("Some notes.", encoding="utf-8")
    monkeypatch.setattr(ingest, "DATA_DIR", tmp_path)

    docs = ingest.load_documents()

    assert len(docs) == 1
    assert docs[0].metadata["type"] == "other"
    assert docs[0].metadata["skills"] == ""


def test_load_documents_skips_empty_files(tmp_path, monkeypatch):
    (tmp_path / "empty.txt").write_text("   \n", encoding="utf-8")
    (tmp_path / "cv.txt").write_text("Real content.", encoding="utf-8")
    monkeypatch.setattr(ingest, "DATA_DIR", tmp_path)

    docs = ingest.load_documents()

    assert len(docs) == 1
    assert docs[0].metadata["source"] == "cv.txt"


def test_load_documents_ignores_non_txt_files(tmp_path, monkeypatch):
    (tmp_path / "cv.txt").write_text("Real content.", encoding="utf-8")
    (tmp_path / "notes.md").write_text("Markdown notes.", encoding="utf-8")
    monkeypatch.setattr(ingest, "DATA_DIR", tmp_path)

    docs = ingest.load_documents()

    assert len(docs) == 1
    assert docs[0].metadata["source"] == "cv.txt"


def test_load_documents_returns_empty_list_for_empty_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(ingest, "DATA_DIR", tmp_path)

    docs = ingest.load_documents()

    assert docs == []


def test_infer_type_matches_known_filenames():
    assert ingest.infer_type("cv.txt") == "cv"
    assert ingest.infer_type("cover_letter.txt") == "cover_letter"
    assert ingest.infer_type("cover letter Dana.pdf") == "cover_letter"
    assert ingest.infer_type("project_recommender.txt") == "project"
    assert ingest.infer_type("model monitoring.txt") == "project"
    assert ingest.infer_type("motivational.txt") == "other"


def test_infer_type_is_case_insensitive():
    assert ingest.infer_type("CV.TXT") == "cv"
    assert ingest.infer_type("COVER_LETTER_OLD.TXT") == "cover_letter"


def test_load_pdf_documents_reads_pdf_with_cover_letter_metadata(tmp_path, monkeypatch):
    (tmp_path / "letter.pdf").write_bytes(make_pdf_bytes("Dear Hiring Team"))
    monkeypatch.setattr(ingest, "COVER_LETTERS_DIR", tmp_path)

    docs = ingest.load_pdf_documents()

    assert len(docs) == 1
    doc = docs[0]
    assert "Dear Hiring Team" in doc.page_content
    assert doc.metadata == {"source": "letter.pdf", "type": "cover_letter", "skills": ""}


def test_load_pdf_documents_ignores_non_pdf_files(tmp_path, monkeypatch):
    (tmp_path / "letter.pdf").write_bytes(make_pdf_bytes("Dear Hiring Team"))
    (tmp_path / "notes.txt").write_text("Not a PDF.", encoding="utf-8")
    monkeypatch.setattr(ingest, "COVER_LETTERS_DIR", tmp_path)

    docs = ingest.load_pdf_documents()

    assert len(docs) == 1
    assert docs[0].metadata["source"] == "letter.pdf"


def test_load_pdf_documents_returns_empty_list_for_empty_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(ingest, "COVER_LETTERS_DIR", tmp_path)

    docs = ingest.load_pdf_documents()

    assert docs == []
