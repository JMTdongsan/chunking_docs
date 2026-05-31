from __future__ import annotations

from pathlib import Path

from PIL import Image


class HuggingFaceVLMBackend:
    """Generic transformers VLM backend for local GPU experiments."""

    def __init__(
        self,
        model_name: str,
        device_map: str = "auto",
        torch_dtype: str = "auto",
        max_new_tokens: int = 768,
        attn_implementation: str = "",
    ):
        try:
            import torch
            from transformers import AutoModelForVision2Seq, AutoProcessor
        except ImportError as exc:
            raise RuntimeError("Install chunking-docs[vision] to use HuggingFaceVLMBackend") from exc

        self.model_name = model_name
        self.device_map = device_map
        self.torch_dtype = torch_dtype
        self.max_new_tokens = max_new_tokens
        self.attn_implementation = attn_implementation

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
        self.model = AutoModelForVision2Seq.from_pretrained(model_name, **model_kwargs)

    def metadata(self) -> dict:
        return {
            "provider": "huggingface",
            "model_name": self.model_name,
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
        decoded = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
        return decoded.strip()
