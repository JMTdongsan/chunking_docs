import json

from chunking_docs.models import AssetKind, ChunkKind, DocumentChunk, VisualAsset
from chunking_docs.pipeline import rebuild_search_artifacts, write_embedding_artifacts
from chunking_docs.storage.records import EmbeddingRecord


class FakeTextEmbedder:
    def __init__(self, embedding_dim: int):
        self.embedding_dim = embedding_dim

    def embed_texts(self, texts):
        return [[float(index + 1)] * self.embedding_dim for index, _ in enumerate(texts)]


class FakeImageEmbedder:
    embedding_dim = 5

    def embed_images(self, image_paths):
        return [[0.5] * self.embedding_dim for _ in image_paths]


def test_write_embedding_artifacts_writes_selected_vectors_and_config(tmp_path):
    chunk = DocumentChunk(
        chunk_id="chunk-1",
        doc_id="doc",
        page_start=1,
        page_end=1,
        kind=ChunkKind.TEXT,
        text="urban development strategy",
    )
    asset_path = tmp_path / "page.png"
    asset_path.write_bytes(b"fake image")
    asset = VisualAsset(
        asset_id="asset-1",
        doc_id="doc",
        page_no=1,
        kind=AssetKind.MAP,
        path=asset_path,
        caption="north district development map",
    )

    (tmp_path / "qdrant_image_records.jsonl").write_text("stale\n", encoding="utf-8")
    result = write_embedding_artifacts(
        output_dir=tmp_path,
        chunks=[chunk],
        assets=[asset],
        text_embedder=FakeTextEmbedder(3),
        caption_embedder=FakeTextEmbedder(4),
        image_embedder=None,
        vector_notes={"text_dense": "text model", "caption_dense": "caption model"},
        vector_metadata={
            "text_dense": {
                "backend": "fake-text",
                "model": "fake-text-model",
                "device": "cpu",
            },
            "caption_dense": {
                "backend": "fake-caption",
                "model": "fake-caption-model",
                "device": "cpu",
            },
        },
    )

    assert result["records"] == {"text_dense": 1, "caption_dense": 1}
    assert result["embedding_manifest"] == str(tmp_path / "embedding_manifest.json")
    assert not (tmp_path / "qdrant_image_records.jsonl").exists()

    text_records = [
        EmbeddingRecord.model_validate_json(line)
        for line in (tmp_path / "qdrant_text_records.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    caption_records = [
        EmbeddingRecord.model_validate_json(line)
        for line in (tmp_path / "qdrant_caption_records.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(text_records[0].vector) == 3
    assert len(caption_records[0].vector) == 4

    config = json.loads((tmp_path / "qdrant_collection.json").read_text(encoding="utf-8"))
    assert config["named_vectors"]["text_dense"]["size"] == 3
    assert config["named_vectors"]["text_dense"]["note"] == "text model"
    assert config["named_vectors"]["caption_dense"]["size"] == 4
    assert {"field": "doc_id", "schema": "keyword"} in config["payload_indexes"]
    assert {"field": "page_no", "schema": "integer"} in config["payload_indexes"]
    assert "image_dense" not in config["named_vectors"]
    manifest = json.loads((tmp_path / "embedding_manifest.json").read_text(encoding="utf-8"))
    assert manifest["collection"] == "document_chunks"
    assert manifest["vectors"]["text_dense"]["file"] == "qdrant_text_records.jsonl"
    assert manifest["vectors"]["text_dense"]["record_count"] == 1
    assert manifest["vectors"]["text_dense"]["dimension"] == 3
    assert manifest["vectors"]["text_dense"]["note"] == "text model"
    assert manifest["vectors"]["text_dense"]["embedding"] == {
        "backend": "fake-text",
        "model": "fake-text-model",
        "device": "cpu",
        "batch_size": 32,
    }
    assert manifest["vectors"]["text_dense"]["exists"] is True
    assert len(manifest["vectors"]["text_dense"]["sha256"]) == 64
    assert manifest["vectors"]["caption_dense"]["dimension"] == 4
    assert manifest["vectors"]["caption_dense"]["embedding"]["backend"] == "fake-caption"
    assert manifest["vectors"]["caption_dense"]["embedding"]["batch_size"] == 32
    assert "image_dense" not in manifest["vectors"]


def test_write_embedding_artifacts_supports_image_vectors(tmp_path):
    asset_path = tmp_path / "page.png"
    asset_path.write_bytes(b"fake image")
    asset = VisualAsset(
        asset_id="asset-1",
        doc_id="doc",
        page_no=1,
        kind=AssetKind.PAGE_IMAGE,
        path=asset_path,
    )

    result = write_embedding_artifacts(
        output_dir=tmp_path,
        chunks=[],
        assets=[asset],
        image_embedder=FakeImageEmbedder(),
    )

    assert result["records"] == {"image_dense": 1}
    manifest = json.loads((tmp_path / "embedding_manifest.json").read_text(encoding="utf-8"))
    assert manifest["vectors"]["image_dense"]["record_count"] == 1
    assert manifest["vectors"]["image_dense"]["dimension"] == 5
    image_records = [
        EmbeddingRecord.model_validate_json(line)
        for line in (tmp_path / "qdrant_image_records.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert image_records[0].vector_name == "image_dense"
    assert len(image_records[0].vector) == 5


def test_caption_embedding_records_include_structured_visual_metadata(tmp_path):
    asset = VisualAsset(
        asset_id="asset-1",
        doc_id="doc",
        page_no=2,
        kind=AssetKind.FIGURE,
        metadata={
            "visual_elements": ["route legend"],
            "objects": [{"label": "station marker", "attributes": ["red circle"]}],
        },
    )

    result = write_embedding_artifacts(
        output_dir=tmp_path,
        chunks=[],
        assets=[asset],
        caption_embedder=FakeTextEmbedder(4),
    )

    assert result["records"] == {"caption_dense": 1}
    caption_records = [
        EmbeddingRecord.model_validate_json(line)
        for line in (tmp_path / "qdrant_caption_records.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert caption_records[0].payload["text"] == (
        "Visual elements: route legend\nObjects: station marker: red circle"
    )


def test_rebuild_search_artifacts_refreshes_bm25_without_overwriting_embeddings(tmp_path):
    (tmp_path / "qdrant_collection.json").write_text(
        json.dumps(
            {
                "collection": "custom_documents",
                "named_vectors": {"text_dense": {"size": 1024, "distance": "Cosine"}},
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "embedding_manifest.json").write_text(
        json.dumps(
            {
                "collection": "custom_documents",
                "vectors": {"text_dense": {"dimension": 1024}},
            }
        ),
        encoding="utf-8",
    )
    chunk = DocumentChunk(
        chunk_id="chunk-1",
        doc_id="doc",
        page_start=1,
        page_end=1,
        kind=ChunkKind.TEXT,
        text="renewal strategy",
    )

    rebuild_search_artifacts(tmp_path, [chunk])

    config = json.loads((tmp_path / "qdrant_collection.json").read_text(encoding="utf-8"))
    manifest = json.loads((tmp_path / "embedding_manifest.json").read_text(encoding="utf-8"))
    assert (tmp_path / "bm25_tokens.json").exists()
    assert config["named_vectors"]["text_dense"]["size"] == 1024
    assert manifest["vectors"]["text_dense"]["dimension"] == 1024


def test_rebuild_search_artifacts_can_rebuild_hashing_embeddings(tmp_path):
    (tmp_path / "qdrant_collection.json").write_text(
        json.dumps({"collection": "custom_documents", "named_vectors": {}}),
        encoding="utf-8",
    )
    chunk = DocumentChunk(
        chunk_id="chunk-1",
        doc_id="doc",
        page_start=1,
        page_end=1,
        kind=ChunkKind.TEXT,
        text="renewal strategy",
    )

    rebuild_search_artifacts(tmp_path, [chunk], rebuild_embeddings=True)

    config = json.loads((tmp_path / "qdrant_collection.json").read_text(encoding="utf-8"))
    manifest = json.loads((tmp_path / "embedding_manifest.json").read_text(encoding="utf-8"))
    assert config["collection"] == "custom_documents"
    assert manifest["collection"] == "custom_documents"
    assert manifest["vectors"]["text_dense"]["embedding"]["backend"] == "hashing"
