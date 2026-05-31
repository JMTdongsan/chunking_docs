from pathlib import Path

from chunking_docs.models import AssetKind, VisualAsset
from chunking_docs.vision.jobs import completed_annotations, plan_visual_jobs, run_visual_jobs


class FakeOCR:
    def recognize(self, image_path: Path, language: str = "kor+eng"):
        return f"ocr:{image_path.name}:{language}"


class FakeVLM:
    def summarize(self, image_path: Path, prompt: str):
        return f"vlm:{image_path.name}:{prompt[:4]}"


def test_plan_visual_jobs_prioritizes_maps_and_missing_annotations(tmp_path):
    map_path = tmp_path / "map.png"
    page_path = tmp_path / "page.png"
    map_path.write_bytes(b"map")
    page_path.write_bytes(b"page")
    assets = [
        VisualAsset(
            asset_id="page",
            doc_id="doc",
            page_no=1,
            kind=AssetKind.PAGE_IMAGE,
            path=page_path,
            metadata={"requires_ocr": True, "requires_vlm": True},
        ),
        VisualAsset(
            asset_id="map",
            doc_id="doc",
            page_no=2,
            kind=AssetKind.MAP,
            path=map_path,
            metadata={"requires_ocr": True, "requires_vlm": True},
        ),
    ]

    jobs = plan_visual_jobs(assets)

    assert [job.asset_id for job in jobs] == ["map", "page"]
    assert jobs[0].operations == ["ocr", "vlm"]


def test_run_visual_jobs_returns_asset_annotations(tmp_path):
    image_path = tmp_path / "map.png"
    image_path.write_bytes(b"map")
    assets = [
        VisualAsset(
            asset_id="map",
            doc_id="doc",
            page_no=2,
            kind=AssetKind.MAP,
            path=image_path,
            metadata={"requires_ocr": True, "requires_vlm": True},
        )
    ]
    jobs = plan_visual_jobs(assets)

    results = run_visual_jobs(jobs, assets, ocr_backend=FakeOCR(), vlm_backend=FakeVLM())
    annotations = completed_annotations(results)

    assert results[0].status == "completed"
    assert annotations[0].asset_id == "map"
    assert annotations[0].ocr_text.startswith("ocr:")
    assert annotations[0].vlm_summary.startswith("vlm:")
