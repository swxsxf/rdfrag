from rdfrag_vkr.modules.ner import RuleBasedNER
from rdfrag_vkr.modules.relation_extraction import RuleBasedRelationExtractor
from rdfrag_vkr.schemas import ArticleMetadata, ParsedDocument


def test_rule_based_ner_and_relations():
    document = ParsedDocument(
        metadata=ArticleMetadata(
            doc_id="doc-1",
            source_file="sample.pdf",
            title="Blockchain for digital twins in the digital economy",
            authors=["Alice Example"],
            year=2024,
            page_count=1,
        ),
        text=(
            "This systematic review studies blockchain and digital twin applications. "
            "The dataset is a benchmark corpus and evaluation uses accuracy."
        ),
        pages=[],
    )
    ner = RuleBasedNER()
    entities = ner.extract(document)
    labels = {entity.label.lower() for entity in entities}
    assert "alice example" in labels
    assert "blockchain" in labels
    assert "digital economy" in labels

    relations = RuleBasedRelationExtractor().extract(document, entities)
    predicates = {relation.predicate for relation in relations}
    assert "hasAuthor" in predicates
    assert "hasTopic" in predicates
    assert "mentionsMethod" in predicates
