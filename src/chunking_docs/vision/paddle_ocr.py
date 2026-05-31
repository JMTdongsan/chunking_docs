from __future__ import annotations

from pathlib import Path
from typing import Any


class PaddleOCRBackend:
    def __init__(
        self,
        lang: str = "korean",
        device: str = "",
        engine: str = "",
        min_confidence: float = 0.0,
        use_doc_orientation_classify: bool = False,
        use_doc_unwarping: bool = False,
        use_textline_orientation: bool = False,
        use_angle_cls: bool = True,
        use_gpu: bool | None = None,
        pipeline: Any | None = None,
    ):
        self.lang = lang
        self.device = device
        self.engine = engine
        self.min_confidence = min_confidence
        self.use_doc_orientation_classify = use_doc_orientation_classify
        self.use_doc_unwarping = use_doc_unwarping
        self.use_textline_orientation = use_textline_orientation
        self.use_angle_cls = use_angle_cls
        self.use_gpu = use_gpu

        self.pipeline = pipeline or self._build_pipeline()
        self.api = "predict" if hasattr(self.pipeline, "predict") else "ocr"

    def _build_pipeline(self):
        try:
            from paddleocr import PaddleOCR
        except ImportError as exc:  # pragma: no cover - depends on optional package.
            raise RuntimeError("Install chunking-docs[ocr] to use PaddleOCR") from exc

        kwargs: dict[str, Any] = {
            "lang": self.lang,
            "use_doc_orientation_classify": self.use_doc_orientation_classify,
            "use_doc_unwarping": self.use_doc_unwarping,
            "use_textline_orientation": self.use_textline_orientation,
        }
        if self.device:
            kwargs["device"] = self.device
        if self.engine:
            kwargs["engine"] = self.engine
        try:
            return PaddleOCR(**kwargs)
        except TypeError:
            legacy_kwargs: dict[str, Any] = {
                "lang": self.lang,
                "use_angle_cls": self.use_angle_cls,
            }
            if self.use_gpu is not None:
                legacy_kwargs["use_gpu"] = self.use_gpu
            return PaddleOCR(**legacy_kwargs)

    def metadata(self) -> dict:
        return {
            "provider": "paddleocr",
            "lang": self.lang,
            "device": self.device,
            "engine": self.engine,
            "min_confidence": self.min_confidence,
            "api": self.api,
            "use_doc_orientation_classify": self.use_doc_orientation_classify,
            "use_doc_unwarping": self.use_doc_unwarping,
            "use_textline_orientation": self.use_textline_orientation,
            "use_angle_cls": self.use_angle_cls,
            "use_gpu": self.use_gpu,
        }

    def recognize(self, image_path: Path, language: str = "") -> str:
        if hasattr(self.pipeline, "predict"):
            result = self.pipeline.predict(str(image_path))
        else:
            result = self.pipeline.ocr(str(image_path), cls=self.use_angle_cls)
        lines = paddle_result_text_lines(result, min_confidence=self.min_confidence)
        return "\n".join(lines)


def paddle_result_text_lines(result: Any, min_confidence: float = 0.0) -> list[str]:
    lines: list[str] = []
    collect_paddle_text_lines(result, lines, min_confidence=min_confidence)
    return lines


def collect_paddle_text_lines(result: Any, lines: list[str], min_confidence: float = 0.0) -> None:
    result = unwrap_result_object(result)
    if result is None:
        return
    if isinstance(result, dict):
        collect_mapping_text_lines(result, lines, min_confidence=min_confidence)
        return
    if is_legacy_text_line(result):
        text, score = legacy_text_line_payload(result)
        add_text_line(lines, text, score, min_confidence=min_confidence)
        return
    if isinstance(result, list | tuple):
        for item in result:
            collect_paddle_text_lines(item, lines, min_confidence=min_confidence)


def collect_mapping_text_lines(
    result: dict[str, Any],
    lines: list[str],
    min_confidence: float = 0.0,
) -> None:
    if isinstance(result.get("res"), dict):
        collect_mapping_text_lines(result["res"], lines, min_confidence=min_confidence)
        return

    texts = result.get("rec_texts")
    if isinstance(texts, list | tuple):
        scores = result.get("rec_scores") or []
        for index, text in enumerate(texts):
            score = scores[index] if index < len(scores) else None
            add_text_line(lines, text, score, min_confidence=min_confidence)
        return

    for key in ("text", "transcription"):
        if key in result:
            add_text_line(lines, result.get(key), result.get("score"), min_confidence=min_confidence)


def unwrap_result_object(result: Any) -> Any:
    if isinstance(result, dict | list | tuple) or result is None:
        return result
    if hasattr(result, "json"):
        payload = result.json
        payload = payload() if callable(payload) else payload
        if isinstance(payload, dict | list | tuple):
            return payload
    if hasattr(result, "to_dict"):
        payload = result.to_dict()
        if isinstance(payload, dict | list | tuple):
            return payload
    return result


def is_legacy_text_line(result: Any) -> bool:
    if not isinstance(result, list | tuple) or len(result) < 2:
        return False
    payload = result[1]
    return isinstance(payload, list | tuple) and payload and isinstance(payload[0], str)


def legacy_text_line_payload(result: list | tuple) -> tuple[str, float | None]:
    payload = result[1]
    text = str(payload[0])
    score = payload[1] if len(payload) > 1 else None
    return text, score


def add_text_line(
    lines: list[str],
    text: Any,
    score: Any = None,
    min_confidence: float = 0.0,
) -> None:
    stripped = str(text or "").strip()
    if not stripped:
        return
    if score is not None and confidence_value(score) < min_confidence:
        return
    lines.append(stripped)


def confidence_value(score: Any) -> float:
    try:
        return float(score)
    except (TypeError, ValueError):
        return 0.0
