from types import SimpleNamespace

from chunking_docs.storage.qdrant_store import QdrantChunkStore


class FakeQdrantClient:
    def __init__(self):
        self.collections = []
        self.payload_schema = {}
        self.created_collection = None
        self.created_payload_indexes = []

    def get_collections(self):
        return SimpleNamespace(collections=[SimpleNamespace(name=name) for name in self.collections])

    def get_collection(self, **kwargs):
        return SimpleNamespace(payload_schema=self.payload_schema)

    def create_collection(self, **kwargs):
        self.created_collection = kwargs
        self.collections.append(kwargs["collection_name"])

    def create_payload_index(self, **kwargs):
        self.created_payload_indexes.append(kwargs)

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


def test_qdrant_ensure_collection_creates_payload_indexes():
    store = object.__new__(QdrantChunkStore)
    store.collection_name = "collection"
    store.client = FakeQdrantClient()
    store._vector_params = lambda size, distance: {"size": size, "distance": distance}
    store._distance = SimpleNamespace(COSINE="cosine")
    store._payload_schema_type = SimpleNamespace(KEYWORD="keyword", INTEGER="integer")

    store.ensure_collection(
        {"text_dense": 3},
        payload_indexes=[
            {"field": "doc_id", "schema": "keyword"},
            {"field": "page_no", "schema": "integer"},
            "asset_id",
        ],
    )

    assert store.client.created_collection["collection_name"] == "collection"
    assert store.client.created_collection["vectors_config"] == {
        "text_dense": {"size": 3, "distance": "cosine"}
    }
    assert [
        (call["field_name"], call["field_schema"])
        for call in store.client.created_payload_indexes
    ] == [
        ("doc_id", "keyword"),
        ("page_no", "integer"),
        ("asset_id", "keyword"),
    ]


def test_qdrant_ensure_collection_skips_existing_payload_index():
    store = object.__new__(QdrantChunkStore)
    store.collection_name = "collection"
    store.client = FakeQdrantClient()
    store.client.collections = ["collection"]
    store.client.payload_schema = {"doc_id": "keyword"}
    store._vector_params = lambda size, distance: {"size": size, "distance": distance}
    store._distance = SimpleNamespace(COSINE="cosine")
    store._payload_schema_type = SimpleNamespace(KEYWORD="keyword", INTEGER="integer")

    store.ensure_collection(
        {"text_dense": 3},
        payload_indexes=[
            {"field": "doc_id", "schema": "keyword"},
            {"field": "page_no", "schema": "integer"},
        ],
    )

    assert store.client.created_collection is None
    assert [
        (call["field_name"], call["field_schema"])
        for call in store.client.created_payload_indexes
    ] == [("page_no", "integer")]


def test_qdrant_query_vector_builds_extended_payload_filter():
    store = object.__new__(QdrantChunkStore)
    store.collection_name = "collection"
    store.client = FakeQdrantClient()
    store._filter = lambda must: SimpleNamespace(must=must)
    store._field_condition = (
        lambda key, match=None, range=None: SimpleNamespace(key=key, match=match, range=range)
    )
    store._match_value = lambda value: SimpleNamespace(value=value)
    store._match_any = lambda any: SimpleNamespace(any=any)
    store._range = lambda **kwargs: SimpleNamespace(**kwargs)

    store.query_vector(
        [1.0, 0.0],
        must_payload={
            "doc_id": "doc",
            "kind": ["map", "figure"],
            "page_start": {"lte": 12},
            "page_end": {"gte": 12},
        },
    )

    conditions = store.client.query_kwargs["query_filter"].must
    assert [(condition.key, getattr(condition.match, "value", None)) for condition in conditions[:1]] == [
        ("doc_id", "doc")
    ]
    assert conditions[1].key == "kind"
    assert conditions[1].match.any == ["map", "figure"]
    assert conditions[2].range.lte == 12
    assert conditions[3].range.gte == 12
