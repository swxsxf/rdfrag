from rdfrag_vkr.config import Settings
from rdfrag_vkr.modules.preprocessing import Preprocessor


def test_clean_text_and_chunking(tmp_path):
    settings = Settings(project_root=tmp_path)
    settings.ensure_directories()
    processor = Preprocessor(settings)

    raw_text = "Digital twin-\n based systems are useful.  \n\nBlockchain improves trust."
    cleaned = processor.clean_text(raw_text)
    assert "twin based" not in cleaned
    assert "twinbased" in cleaned

    chunks = processor.chunk_text(
        text=cleaned * 30,
        doc_id="doc-1",
        title="Test Article",
        source_file="test.pdf",
    )
    assert chunks
    assert chunks[0].doc_id == "doc-1"
