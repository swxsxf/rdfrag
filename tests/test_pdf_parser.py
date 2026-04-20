from rdfrag_vkr.config import Settings
from rdfrag_vkr.modules.pdf_parser import PDFParser


def test_parser_reports_pypdf_when_grobid_is_unavailable(tmp_path, monkeypatch):
    settings = Settings(project_root=tmp_path)
    settings.ensure_directories()
    parser = PDFParser(settings)
    monkeypatch.setattr(parser, "is_grobid_available", lambda: False)

    assert parser.is_grobid_available() is False
