from chunking_docs.graph.export import export_graph, related_terms
from chunking_docs.models import ChunkKind, DocumentChunk, GraphTriple


def test_export_graph_builds_nodes_and_edges():
    chunk = DocumentChunk(
        chunk_id="chunk-1",
        doc_id="doc",
        page_start=188,
        page_end=188,
        kind=ChunkKind.TEXT,
        text="동북권 발전구상",
    )
    triple = GraphTriple(
        triple_id="triple-1",
        doc_id="doc",
        chunk_id="chunk-1",
        subject="동북권",
        predicate="uses_axis",
        object="중랑천 수변축",
    )

    nodes, edges = export_graph([triple], chunks=[chunk])

    assert {node.label for node in nodes} == {"동북권", "중랑천 수변축"}
    assert edges[0].predicate == "uses_axis"
    assert edges[0].metadata["page_start"] == 188


def test_related_terms_from_triples():
    triples = [
        GraphTriple(
            triple_id="triple-1",
            doc_id="doc",
            chunk_id="chunk-1",
            subject="동북권",
            predicate="uses_axis",
            object="중랑천 수변축",
        )
    ]

    assert "중랑천 수변축" in related_terms(triples, "동북권")
