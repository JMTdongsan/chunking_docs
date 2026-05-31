from __future__ import annotations

from pathlib import Path
from typing import Protocol


class OCRBackend(Protocol):
    def recognize(self, image_path: Path, language: str = "ko") -> str:
        """Return OCR text for the given page or visual asset image."""


class VLMBackend(Protocol):
    def summarize(self, image_path: Path, prompt: str) -> str:
        """Return a vision-language summary for the given image."""


class NullOCRBackend:
    def recognize(self, image_path: Path, language: str = "ko") -> str:
        return ""


class NullVLMBackend:
    def summarize(self, image_path: Path, prompt: str) -> str:
        return ""
