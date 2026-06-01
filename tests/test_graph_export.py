from chunking_docs.graph.export import export_graph, related_terms, summarize_graph
from chunking_docs.models import ChunkKind, DocumentChunk, GraphTriple


def test_export_graph_builds_nodes_and_edges():
    chunk = DocumentChunk(
        chunk_id="chunk-1",
        doc_id="doc",
        page_start=188,
        page_end=188,
        kind=ChunkKind.TEXT,
        text="north district development concept",
    )
    triple = GraphTriple(
        triple_id="triple-1",
        doc_id="doc",
        chunk_id="chunk-1",
        subject="north district",
        predicate="uses_axis",
        object="riverfront axis",
    )

    nodes, edges = export_graph([triple], chunks=[chunk])

    assert {node.label for node in nodes} == {"north district", "riverfront axis"}
    assert edges[0].predicate == "uses_axis"
    assert edges[0].metadata["triple_id"] == "triple-1"
    assert edges[0].metadata["page_start"] == 188
    node_by_label = {node.label: node for node in nodes}
    assert node_by_label["north district"].metadata["out_degree"] == 1
    assert node_by_label["riverfront axis"].metadata["in_degree"] == 1
    assert node_by_label["north district"].metadata["chunk_ids"] == ["chunk-1"]


def test_summarize_graph_reports_connectivity_and_top_nodes():
    triples = [
        GraphTriple(
            triple_id="triple-1",
            doc_id="doc",
            chunk_id="chunk-1",
            subject="hub",
            predicate="connects_to",
            object="corridor",
        ),
        GraphTriple(
            triple_id="triple-2",
            doc_id="doc",
            chunk_id="chunk-1",
            subject="hub",
            predicate="supports",
            object="district",
        ),
    ]

    nodes, edges = export_graph(triples)
    summary = summarize_graph(nodes, edges)

    assert summary.node_count == 3
    assert summary.edge_count == 2
    assert summary.connected_component_count == 1
    assert summary.largest_component_node_count == 3
    assert summary.max_degree == 2
    assert summary.predicate_counts == {"connects_to": 1, "supports": 1}
    assert summary.top_nodes[0]["label"] == "hub"
    assert summary.top_nodes[0]["degree"] == 2


def test_related_terms_from_triples():
    triples = [
        GraphTriple(
            triple_id="triple-1",
            doc_id="doc",
            chunk_id="chunk-1",
            subject="north district",
            predicate="uses_axis",
            object="riverfront axis",
        )
    ]

    assert "riverfront axis" in related_terms(triples, "north district")
