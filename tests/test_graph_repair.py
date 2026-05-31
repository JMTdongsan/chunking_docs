from chunking_docs.graph.repair import remap_triples_to_available_chunks
from chunking_docs.models import ChunkKind, DocumentChunk, GraphTriple


def test_remap_triples_to_first_available_child_chunk():
    child = DocumentChunk(
        chunk_id="child-1",
        doc_id="doc",
        page_start=1,
        page_end=1,
        kind=ChunkKind.TEXT,
        text="child",
        metadata={"parent_chunk_id": "parent-1"},
    )
    triple = GraphTriple(
        triple_id="triple-1",
        doc_id="doc",
        chunk_id="parent-1",
        subject="north district",
        predicate="uses_axis",
        object="river corridor",
    )

    remapped = remap_triples_to_available_chunks([triple], [child])

    assert remapped[0].chunk_id == "child-1"
    assert remapped[0].qualifiers["original_chunk_id"] == "parent-1"
    assert remapped[0].triple_id != "triple-1"
