from pathlib import Path

from chunking_docs.vision.paddle_ocr import PaddleOCRBackend, paddle_result_text_lines


class PredictPipeline:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def predict(self, image_path):
        self.calls.append(image_path)
        return self.result


class LegacyPipeline:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def ocr(self, image_path, cls=True):
        self.calls.append((image_path, cls))
        return self.result


def test_paddle_result_text_lines_parses_predict_result():
    result = [
        {
            "res": {
                "rec_texts": ["First line", "  ", "Low confidence"],
                "rec_scores": [0.95, 0.9, 0.2],
            }
        }
    ]

    assert paddle_result_text_lines(result, min_confidence=0.5) == ["First line"]


def test_paddle_backend_recognize_uses_predict_api():
    pipeline = PredictPipeline(
        [
            {
                "rec_texts": ["Alpha", "Beta"],
                "rec_scores": [0.9, 0.8],
            }
        ]
    )
    backend = PaddleOCRBackend(lang="korean", min_confidence=0.5, pipeline=pipeline)

    text = backend.recognize(Path("page.png"))

    assert text == "Alpha\nBeta"
    assert pipeline.calls == ["page.png"]
    assert backend.metadata()["api"] == "predict"


def test_paddle_backend_recognize_uses_legacy_ocr_api():
    pipeline = LegacyPipeline(
        [
            [
                [[[0, 0], [1, 0], [1, 1], [0, 1]], ("Alpha", 0.9)],
                [[[0, 2], [1, 2], [1, 3], [0, 3]], ("Low confidence", 0.1)],
            ]
        ]
    )
    backend = PaddleOCRBackend(
        lang="korean",
        min_confidence=0.5,
        use_angle_cls=False,
        pipeline=pipeline,
    )

    text = backend.recognize(Path("page.png"))

    assert text == "Alpha"
    assert pipeline.calls == [("page.png", False)]
    assert backend.metadata()["api"] == "ocr"
