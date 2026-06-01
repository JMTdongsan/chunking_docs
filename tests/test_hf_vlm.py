import pytest

from chunking_docs.vision.hf_vlm import (
    build_hf_quantization_config,
    effective_vlm_quantization,
    get_vlm_model_profile,
    hf_vlm_dependency_error_message,
    hf_vlm_model_loaders,
    load_hf_vlm_model,
    normalize_vlm_quantization,
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

    class BitsAndBytesConfig:
        def __init__(self, **kwargs):
            self.kwargs = kwargs


class FakeTorch:
    bfloat16 = "bf16"
    float16 = "fp16"
    float32 = "fp32"


def test_get_vlm_model_profile_normalizes_name():
    profile = get_vlm_model_profile("qwen2-5-vl-7b")

    assert profile.model_name == "Qwen/Qwen2.5-VL-7B-Instruct"
    assert profile.model_class == "image-text-to-text"


def test_get_vlm_model_profile_supports_quantized_32b_profile():
    profile = get_vlm_model_profile("qwen2-5-vl-32b-bnb4")

    assert profile.model_name == "Qwen/Qwen2.5-VL-32B-Instruct"
    assert profile.min_gpu_memory_mib == 30720
    assert profile.quantization == "bitsandbytes_4bit"
    assert effective_vlm_quantization(profile) == "bitsandbytes_4bit"


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


def test_vlm_quantization_aliases_and_config():
    assert normalize_vlm_quantization("bnb4") == "bitsandbytes_4bit"
    assert normalize_vlm_quantization("none") == ""

    config = build_hf_quantization_config(
        FakeTransformers,
        FakeTorch,
        "bitsandbytes_4bit",
        "bfloat16",
    )

    assert config.kwargs == {
        "load_in_4bit": True,
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_compute_dtype": "bf16",
        "bnb_4bit_use_double_quant": True,
    }


def test_hf_vlm_dependency_error_message_points_to_vision_extra():
    message = hf_vlm_dependency_error_message()

    assert "chunking-docs[vision]" in message
    assert "doctor --require-vision" in message
