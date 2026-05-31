from types import SimpleNamespace

from typer.testing import CliRunner

from chunking_docs.cli import app
from chunking_docs.storage.qdrant_store import QdrantChunkStore


class FakeQdrantClient:
    def __init__(self):
        self.collections = []
        self.payload_schema = {}
        self.vector_config = {}
        self.created_collection = None
        self.created_payload_indexes = []

    def get_collections(self):
        return SimpleNamespace(collections=[SimpleNamespace(name=name) for name in self.collections])

    def get_collection(self, **kwargs):
        return SimpleNamespace(
            payload_schema=self.payload_schema,
            config=SimpleNamespace(params=SimpleNamespace(vectors=self.vector_config)),
        )

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


def test_qdrant_collection_contract_passes_matching_collection():
    store = object.__new__(QdrantChunkStore)
    store.collection_name = "collection"
    store.client = FakeQdrantClient()
    store.client.collections = ["collection"]
    store.client.vector_config = {
        "text_dense": SimpleNamespace(size=3),
        "caption_dense": SimpleNamespace(size=4),
    }
    store.client.payload_schema = {"doc_id": "keyword", "page_no": "integer"}
    store._payload_schema_type = SimpleNamespace(KEYWORD="keyword", INTEGER="integer")

    report = store.check_collection_contract(
        {"text_dense": 3, "caption_dense": 4},
        payload_indexes=[
            {"field": "doc_id", "schema": "keyword"},
            {"field": "page_no", "schema": "integer"},
        ],
    )

    assert report.passed is True
    assert report.exists is True
    assert report.actual_vectors == {"caption_dense": 4, "text_dense": 3}
    assert report.missing_payload_indexes == []


def test_qdrant_collection_contract_flags_mismatch_and_missing_indexes():
    store = object.__new__(QdrantChunkStore)
    store.collection_name = "collection"
    store.client = FakeQdrantClient()
    store.client.collections = ["collection"]
    store.client.vector_config = {"text_dense": SimpleNamespace(size=2)}
    store.client.payload_schema = {"doc_id": "keyword"}
    store._payload_schema_type = SimpleNamespace(KEYWORD="keyword", INTEGER="integer")

    report = store.check_collection_contract(
        {"text_dense": 3, "caption_dense": 4},
        payload_indexes=[
            {"field": "doc_id", "schema": "keyword"},
            {"field": "page_no", "schema": "integer"},
        ],
    )

    assert report.passed is False
    assert report.missing_vectors == ["caption_dense"]
    assert report.mismatched_vectors == {"text_dense": {"expected": 3, "actual": 2}}
    assert report.missing_payload_indexes == ["page_no"]
    assert set(report.failed_checks) == {
        "missing_vectors",
        "vector_size_mismatch",
        "missing_payload_indexes",
    }


def test_qdrant_collection_contract_allows_missing_collection():
    store = object.__new__(QdrantChunkStore)
    store.collection_name = "collection"
    store.client = FakeQdrantClient()
    store._payload_schema_type = SimpleNamespace(KEYWORD="keyword", INTEGER="integer")

    report = store.check_collection_contract({"text_dense": 3}, allow_missing=True)

    assert report.passed is True
    assert report.exists is False


def test_qdrant_check_collection_cli_writes_report(tmp_path, monkeypatch):
    package_dir = tmp_path / "package"
    package_dir.mkdir()
    (package_dir / "qdrant_collection.json").write_text(
        """
        {
          "collection": "document_chunks",
          "named_vectors": {"text_dense": {"size": 3}},
          "payload_indexes": [{"field": "doc_id", "schema": "keyword"}]
        }
        """,
        encoding="utf-8",
    )
    output = tmp_path / "contract.json"

    class FakeStore:
        def __init__(self, **kwargs):
            self.collection_name = kwargs["collection_name"]

        def check_collection_contract(self, named_vectors, payload_indexes=None, allow_missing=False):
            return QdrantChunkStore.check_collection_contract(
                fake_store(named_vectors, payload_indexes),
                named_vectors,
                payload_indexes=payload_indexes,
                allow_missing=allow_missing,
            )

    monkeypatch.setattr("chunking_docs.storage.qdrant_store.QdrantChunkStore", FakeStore)

    result = CliRunner().invoke(
        app,
        [
            "qdrant-check-collection",
            "--package-dir",
            str(package_dir),
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.output
    assert '"passed": true' in output.read_text(encoding="utf-8")


def fake_store(named_vectors, payload_indexes):
    store = object.__new__(QdrantChunkStore)
    store.collection_name = "document_chunks"
    store.client = FakeQdrantClient()
    store.client.collections = ["document_chunks"]
    store.client.vector_config = {
        name: SimpleNamespace(size=size)
        for name, size in named_vectors.items()
    }
    store.client.payload_schema = {
        (index["field"] if isinstance(index, dict) else index): "keyword"
        for index in payload_indexes or []
    }
    store._payload_schema_type = SimpleNamespace(KEYWORD="keyword", INTEGER="integer")
    return store


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
