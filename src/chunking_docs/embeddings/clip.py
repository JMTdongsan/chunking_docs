from __future__ import annotations

from pathlib import Path

from PIL import Image


class TransformersCLIPTextEmbedder:
    def __init__(
        self,
        model_name: str = "openai/clip-vit-large-patch14",
        device: str = "cuda",
        normalize_embeddings: bool = True,
    ):
        try:
            import torch
            from transformers import AutoModel, AutoProcessor
        except ImportError as exc:
            raise RuntimeError("Install chunking-docs[vision] to use TransformersCLIPTextEmbedder") from exc

        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device)
        self.model.eval()
        self.device = device
        self.normalize_embeddings = normalize_embeddings
        self.embedding_dim = int(getattr(self.model.config, "projection_dim", 0) or 768)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        inputs = self.processor(text=texts, return_tensors="pt", padding=True, truncation=True).to(
            self.device
        )
        with self.torch.no_grad():
            if hasattr(self.model, "get_text_features"):
                vectors = self.model.get_text_features(**inputs)
            else:
                outputs = self.model(**inputs)
                vectors = outputs.pooler_output
            if self.normalize_embeddings:
                vectors = vectors / vectors.norm(dim=-1, keepdim=True)
        return vectors.detach().cpu().float().tolist()


class TransformersImageEmbedder:
    def __init__(
        self,
        model_name: str = "openai/clip-vit-large-patch14",
        device: str = "cuda",
        normalize_embeddings: bool = True,
    ):
        try:
            import torch
            from transformers import AutoModel, AutoProcessor
        except ImportError as exc:
            raise RuntimeError("Install chunking-docs[vision] to use TransformersImageEmbedder") from exc

        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device)
        self.model.eval()
        self.device = device
        self.normalize_embeddings = normalize_embeddings
        self.embedding_dim = int(getattr(self.model.config, "projection_dim", 0) or 768)

    def embed_images(self, image_paths: list[Path]) -> list[list[float]]:
        images = [Image.open(path).convert("RGB") for path in image_paths]
        inputs = self.processor(images=images, return_tensors="pt", padding=True).to(self.device)
        with self.torch.no_grad():
            if hasattr(self.model, "get_image_features"):
                vectors = self.model.get_image_features(**inputs)
            else:
                outputs = self.model(**inputs)
                vectors = outputs.pooler_output
            if self.normalize_embeddings:
                vectors = vectors / vectors.norm(dim=-1, keepdim=True)
        return vectors.detach().cpu().float().tolist()
