from types import SimpleNamespace

from chunking_docs.storage.qdrant_store import QdrantChunkStore


class FakeQdrantClient:
    def query_points(self, **kwargs):
        self.query_kwargs = kwargs
        return SimpleNamespace(
            points=[
                SimpleNamespace(
                    id="point-1",
                    score=0.75,
                    payload={
                        "chunk_id": "chunk-1",
                        "doc_id": "doc",
                        "page_start": 1,
                        "page_end": 1,
                        "text": "retrieved text",
                    },
                )
            ]
        )


def test_qdrant_query_vector_maps_query_response():
    store = object.__new__(QdrantChunkStore)
    store.collection_name = "collection"
    store.client = FakeQdrantClient()

    hits = store.query_vector([1.0, 0.0], vector_name="text_dense", top_k=3)

    assert store.client.query_kwargs["collection_name"] == "collection"
    assert store.client.query_kwargs["using"] == "text_dense"
    assert store.client.query_kwargs["limit"] == 3
    assert hits[0].point_id == "point-1"
    assert hits[0].chunk_id == "chunk-1"
    assert hits[0].payload["text"] == "retrieved text"
