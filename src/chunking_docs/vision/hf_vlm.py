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
    min_gpu_memory_mib: int | None = None
    quantization: str = ""
    attn_implementation: str = ""
    notes: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


VLM_MODEL_PROFILES = {
    "qwen2_5_vl_7b": VLMModelProfile(
        name="qwen2_5_vl_7b",
        model_name="Qwen/Qwen2.5-VL-7B-Instruct",
        model_class="image-text-to-text",
        min_gpu_memory_mib=24576,
        notes="General-purpose instruction VLM profile for 24GB+ local GPUs.",
    ),
    "qwen2_vl_7b": VLMModelProfile(
        name="qwen2_vl_7b",
        model_name="Qwen/Qwen2-VL-7B-Instruct",
        model_class="vision2seq",
        min_gpu_memory_mib=24576,
        notes="General-purpose instruction VLM profile for 24GB+ local GPUs.",
    ),
    "qwen2_5_vl_32b_bnb4": VLMModelProfile(
        name="qwen2_5_vl_32b_bnb4",
        model_name="Qwen/Qwen2.5-VL-32B-Instruct",
        model_class="image-text-to-text",
        min_gpu_memory_mib=30720,
        quantization="bitsandbytes_4bit",
        notes=(
            "Larger Qwen2.5-VL profile for 32GB-class local GPUs using "
            "bitsandbytes 4-bit NF4 quantization."
        ),
    ),
    "llava_next_7b": VLMModelProfile(
        name="llava_next_7b",
        model_name="llava-hf/llava-v1.6-mistral-7b-hf",
        model_class="vision2seq",
        min_gpu_memory_mib=24576,
        notes="LLaVA-NeXT profile for visual summaries and chart/map descriptions.",
    ),
    "idefics2_8b": VLMModelProfile(
        name="idefics2_8b",
        model_name="HuggingFaceM4/idefics2-8b",
        model_class="vision2seq",
        min_gpu_memory_mib=24576,
        notes="Idefics2 profile for comparing another open VLM family.",
    ),
    "phi3_5_vision": VLMModelProfile(
        name="phi3_5_vision",
        model_name="microsoft/Phi-3.5-vision-instruct",
        model_class="causal-lm",
        min_gpu_memory_mib=12288,
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
        quantization: str = "",
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
        self.quantization = normalize_vlm_quantization(quantization)
        self.model_class = model_class
        self.profile = profile

        dtype = torch_dtype_value(torch, torch_dtype)

        try:
            self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        except ImportError as exc:
            raise RuntimeError(hf_vlm_dependency_error_message()) from exc
        model_kwargs = {
            "device_map": device_map,
            "torch_dtype": dtype,
            "trust_remote_code": True,
        }
        if attn_implementation:
            model_kwargs["attn_implementation"] = attn_implementation
        if self.quantization:
            model_kwargs["quantization_config"] = build_hf_quantization_config(
                transformers,
                torch,
                self.quantization,
                torch_dtype,
            )
        try:
            self.model = load_hf_vlm_model(
                transformers_module=transformers,
                model_name=model_name,
                model_kwargs=model_kwargs,
                model_class=model_class,
            )
        except ImportError as exc:
            raise RuntimeError(hf_vlm_dependency_error_message()) from exc

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
            "quantization": self.quantization,
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


def hf_vlm_dependency_error_message() -> str:
    return (
        "Hugging Face VLM runtime dependencies are incomplete. "
        "Install chunking-docs[vision] and verify `chunking-docs doctor --require-vision` passes."
    )


def normalize_vlm_quantization(value: str) -> str:
    normalized = value.strip().lower().replace("-", "_")
    aliases = {
        "": "",
        "none": "",
        "no": "",
        "false": "",
        "bitsandbytes_4bit": "bitsandbytes_4bit",
        "bnb_4bit": "bitsandbytes_4bit",
        "bnb4": "bitsandbytes_4bit",
        "4bit": "bitsandbytes_4bit",
        "bitsandbytes_8bit": "bitsandbytes_8bit",
        "bnb_8bit": "bitsandbytes_8bit",
        "bnb8": "bitsandbytes_8bit",
        "8bit": "bitsandbytes_8bit",
    }
    if normalized not in aliases:
        supported = ", ".join(["none", "bitsandbytes_4bit", "bitsandbytes_8bit"])
        raise ValueError(f"VLM quantization must be one of: {supported}")
    return aliases[normalized]


def effective_vlm_quantization(profile: VLMModelProfile, override: str = "auto") -> str:
    normalized_override = override.strip().lower()
    if normalized_override == "auto":
        return normalize_vlm_quantization(profile.quantization)
    return normalize_vlm_quantization(override)


def vlm_quantization_requires_bitsandbytes(quantization: str) -> bool:
    return normalize_vlm_quantization(quantization) in {
        "bitsandbytes_4bit",
        "bitsandbytes_8bit",
    }


def torch_dtype_value(torch_module, torch_dtype: str):
    if torch_dtype == "bfloat16":
        return torch_module.bfloat16
    if torch_dtype == "float16":
        return torch_module.float16
    if torch_dtype == "float32":
        return torch_module.float32
    return torch_dtype


def build_hf_quantization_config(
    transformers_module,
    torch_module,
    quantization: str,
    torch_dtype: str,
):
    normalized = normalize_vlm_quantization(quantization)
    if not normalized:
        return None
    config_cls = getattr(transformers_module, "BitsAndBytesConfig", None)
    if config_cls is None:
        raise RuntimeError(
            "Transformers BitsAndBytesConfig is unavailable. "
            "Install chunking-docs[vision-quantized] for quantized VLM profiles."
        )
    if normalized == "bitsandbytes_4bit":
        return config_cls(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch_dtype_value(torch_module, torch_dtype),
            bnb_4bit_use_double_quant=True,
        )
    if normalized == "bitsandbytes_8bit":
        return config_cls(load_in_8bit=True)
    raise ValueError(f"Unsupported VLM quantization: {quantization}")


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
