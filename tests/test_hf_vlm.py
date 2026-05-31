import pytest

from chunking_docs.vision.hf_vlm import (
    get_vlm_model_profile,
    hf_vlm_model_loaders,
    load_hf_vlm_model,
)


class FakeVision2Seq:
    calls = 0

    @classmethod
    def from_pretrained(cls, model_name, **kwargs):
        cls.calls += 1
        raise ValueError("unsupported")


class FakeImageTextToText:
    calls = 0

    @classmethod
    def from_pretrained(cls, model_name, **kwargs):
        cls.calls += 1
        return {"model_name": model_name, "kwargs": kwargs}


class FakeTransformers:
    AutoModelForVision2Seq = FakeVision2Seq
    AutoModelForImageTextToText = FakeImageTextToText


def test_get_vlm_model_profile_normalizes_name():
    profile = get_vlm_model_profile("qwen2-5-vl-7b")

    assert profile.model_name == "Qwen/Qwen2.5-VL-7B-Instruct"
    assert profile.model_class == "image-text-to-text"


def test_hf_vlm_model_loaders_selects_model_class():
    loaders = hf_vlm_model_loaders(FakeTransformers, "image_text_to_text")

    assert loaders == [FakeImageTextToText]


def test_load_hf_vlm_model_auto_falls_back_to_next_loader():
    FakeVision2Seq.calls = 0
    FakeImageTextToText.calls = 0

    model = load_hf_vlm_model(
        transformers_module=FakeTransformers,
        model_name="model-id",
        model_kwargs={"device_map": "auto"},
        model_class="auto",
    )

    assert model == {"model_name": "model-id", "kwargs": {"device_map": "auto"}}
    assert FakeVision2Seq.calls == 1
    assert FakeImageTextToText.calls == 1


def test_hf_vlm_model_loaders_rejects_unknown_model_class():
    with pytest.raises(ValueError, match="model_class must be one of"):
        hf_vlm_model_loaders(FakeTransformers, "unknown")
