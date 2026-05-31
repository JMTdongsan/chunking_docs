import json

from chunking_docs.models import AssetKind, ChunkKind, DocumentChunk, VisualAsset
from chunking_docs.pipeline import write_embedding_artifacts
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
        text="서울 도시기본계획",
    )
    asset_path = tmp_path / "page.png"
    asset_path.write_bytes(b"fake image")
    asset = VisualAsset(
        asset_id="asset-1",
        doc_id="doc",
        page_no=1,
        kind=AssetKind.MAP,
        path=asset_path,
        caption="동북권 발전구상 지도",
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
    )

    assert result["records"] == {"text_dense": 1, "caption_dense": 1}
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
    assert "image_dense" not in config["named_vectors"]


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
    image_records = [
        EmbeddingRecord.model_validate_json(line)
        for line in (tmp_path / "qdrant_image_records.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert image_records[0].vector_name == "image_dense"
    assert len(image_records[0].vector) == 5
