from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image
from pydantic import BaseModel, Field


class VLMModelProfile(BaseModel):
    name: str
    model_name: str
    model_class: str = "auto"
    device_map: str = "auto"
    torch_dtype: str = "bfloat16"
    max_new_tokens: int = 768
    attn_implementation: str = ""
    notes: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


VLM_MODEL_PROFILES = {
    "qwen2_5_vl_7b": VLMModelProfile(
        name="qwen2_5_vl_7b",
        model_name="Qwen/Qwen2.5-VL-7B-Instruct",
        model_class="image-text-to-text",
        notes="General-purpose instruction VLM profile for 24GB+ local GPUs.",
    ),
    "qwen2_vl_7b": VLMModelProfile(
        name="qwen2_vl_7b",
        model_name="Qwen/Qwen2-VL-7B-Instruct",
        model_class="vision2seq",
        notes="General-purpose instruction VLM profile for 24GB+ local GPUs.",
    ),
    "llava_next_7b": VLMModelProfile(
        name="llava_next_7b",
        model_name="llava-hf/llava-v1.6-mistral-7b-hf",
        model_class="vision2seq",
        notes="LLaVA-NeXT profile for visual summaries and chart/map descriptions.",
    ),
    "idefics2_8b": VLMModelProfile(
        name="idefics2_8b",
        model_name="HuggingFaceM4/idefics2-8b",
        model_class="vision2seq",
        notes="Idefics2 profile for comparing another open VLM family.",
    ),
    "phi3_5_vision": VLMModelProfile(
        name="phi3_5_vision",
        model_name="microsoft/Phi-3.5-vision-instruct",
        model_class="causal-lm",
        notes="Compact VLM profile; trust_remote_code is required.",
    ),
}


def get_vlm_model_profile(name: str) -> VLMModelProfile:
    normalized = name.strip().lower().replace("-", "_")
    if normalized not in VLM_MODEL_PROFILES:
        supported = ", ".join(sorted(VLM_MODEL_PROFILES))
        raise ValueError(f"Unsupported VLM profile '{name}'. Supported profiles: {supported}")
    return VLM_MODEL_PROFILES[normalized]


class HuggingFaceVLMBackend:
    """Generic transformers VLM backend for local GPU experiments."""

    def __init__(
        self,
        model_name: str,
        device_map: str = "auto",
        torch_dtype: str = "auto",
        max_new_tokens: int = 768,
        attn_implementation: str = "",
        model_class: str = "auto",
        profile: str = "",
    ):
        try:
            import torch
            import transformers
            from transformers import AutoProcessor
        except ImportError as exc:
            raise RuntimeError("Install chunking-docs[vision] to use HuggingFaceVLMBackend") from exc

        self.model_name = model_name
        self.device_map = device_map
        self.torch_dtype = torch_dtype
        self.max_new_tokens = max_new_tokens
        self.attn_implementation = attn_implementation
        self.model_class = model_class
        self.profile = profile

        dtype = torch_dtype
        if torch_dtype == "bfloat16":
            dtype = torch.bfloat16
        elif torch_dtype == "float16":
            dtype = torch.float16
        elif torch_dtype == "float32":
            dtype = torch.float32

        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        model_kwargs = {
            "device_map": device_map,
            "torch_dtype": dtype,
            "trust_remote_code": True,
        }
        if attn_implementation:
            model_kwargs["attn_implementation"] = attn_implementation
        self.model = load_hf_vlm_model(
            transformers_module=transformers,
            model_name=model_name,
            model_kwargs=model_kwargs,
            model_class=model_class,
        )

    def metadata(self) -> dict:
        return {
            "provider": "huggingface",
            "model_name": self.model_name,
            "profile": self.profile,
            "model_class": self.model_class,
            "device_map": self.device_map,
            "torch_dtype": self.torch_dtype,
            "max_new_tokens": self.max_new_tokens,
            "attn_implementation": self.attn_implementation,
        }

    def summarize(self, image_path: Path, prompt: str) -> str:
        image = Image.open(image_path).convert("RGB")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        if hasattr(self.processor, "apply_chat_template"):
            text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            inputs = self.processor(images=image, text=text, return_tensors="pt").to(self.model.device)
        else:
            inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.model.device)

        outputs = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        if "input_ids" in inputs and outputs.shape[-1] > inputs["input_ids"].shape[-1]:
            outputs = outputs[:, inputs["input_ids"].shape[-1] :]
        decoded = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
        return decoded.strip()


def load_hf_vlm_model(
    transformers_module,
    model_name: str,
    model_kwargs: dict[str, Any],
    model_class: str = "auto",
):
    loaders = hf_vlm_model_loaders(transformers_module, model_class)
    last_error = None
    for loader in loaders:
        if loader is None:
            continue
        try:
            return loader.from_pretrained(model_name, **model_kwargs)
        except (OSError, ValueError, TypeError, AttributeError) as exc:
            last_error = exc
            if model_class != "auto":
                raise
    if last_error is not None:
        raise RuntimeError(f"Unable to load Hugging Face VLM model '{model_name}'") from last_error
    raise RuntimeError(f"No supported Hugging Face VLM loader is available for model_class={model_class!r}")


def hf_vlm_model_loaders(transformers_module, model_class: str):
    normalized = model_class.strip().lower().replace("_", "-")
    loader_names = {
        "vision2seq": ["AutoModelForVision2Seq"],
        "image-text-to-text": ["AutoModelForImageTextToText"],
        "causal-lm": ["AutoModelForCausalLM"],
        "auto": ["AutoModelForVision2Seq", "AutoModelForImageTextToText", "AutoModelForCausalLM"],
    }
    if normalized not in loader_names:
        supported = ", ".join(sorted(loader_names))
        raise ValueError(f"model_class must be one of: {supported}")
    return [getattr(transformers_module, name, None) for name in loader_names[normalized]]
