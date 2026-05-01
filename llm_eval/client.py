"""Local LLM client. Default model: MamayLM-Gemma-3-12B-IT-v1.0.

Backend is auto-detected:
- CUDA → 4-bit nf4 (bitsandbytes) with bf16 compute. Fits on a 24 GB GPU.
- Apple MPS → fp16; bitsandbytes is not supported on MPS so we run unquantized.
- CPU → fp32 fallback (only useful for tests).

The chat template comes from the tokenizer (``apply_chat_template``); we
never hand-format Gemma's turn separators.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch


DEFAULT_MODEL = "INSAIT-Institute/MamayLM-Gemma-3-12B-IT-v1.0"


@dataclass
class LLMConfig:
    model_id: str = DEFAULT_MODEL
    backend: str = "auto"            # "auto" | "cuda_4bit" | "cuda_bf16" | "mps_fp16" | "cpu"
    dtype: str = "auto"               # "auto" | "bfloat16" | "float16" | "float32"
    max_new_tokens: int = 16
    do_sample: bool = False           # greedy by default for determinism
    temperature: float = 0.0
    top_p: float = 1.0
    seed: int = 1914
    trust_remote_code: bool = False
    # Qwen3 chat template ships with a built-in chain-of-thought "thinking"
    # mode. Set ``False`` to suppress it (returns label tokens directly,
    # comparable to Gemma/Mamay/Lapa). Templates that don't reference this
    # variable ignore it silently.
    enable_thinking: Optional[bool] = None
    extra: dict = field(default_factory=dict)


def _resolve_backend(cfg: LLMConfig) -> str:
    if cfg.backend != "auto":
        return cfg.backend
    if torch.cuda.is_available():
        # 24 GB consumer GPUs benefit from 4-bit; 48 GB+ would prefer bf16
        # but bf16 still works in 4-bit mode (compute dtype is bf16), so we
        # default everyone to 4-bit and let the user override via the config.
        return "cuda_4bit"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps_fp16"
    return "cpu"


def _resolve_dtype(name: str) -> torch.dtype:
    return {
        "auto": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


class MamayClient:
    """Single-process, single-GPU instruction-tuned LLM wrapper."""

    def __init__(self, cfg: LLMConfig):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.cfg = cfg
        self.backend = _resolve_backend(cfg)

        torch.manual_seed(cfg.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg.seed)

        print(f"[llm] loading tokenizer: {cfg.model_id}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.model_id, trust_remote_code=cfg.trust_remote_code
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        print(f"[llm] loading model: {cfg.model_id} (backend={self.backend})")
        load_kwargs: dict = {"trust_remote_code": cfg.trust_remote_code}

        if self.backend == "cuda_4bit":
            from transformers import BitsAndBytesConfig
            # MamayLM ships as Gemma3ForConditionalGeneration (multimodal config)
            # but only the language_model is fine-tuned for Ukrainian. The
            # vision_tower's weights produce NaN logits when 4-bit-quantized,
            # so we keep it (plus the projector and embed/lm_head) unquantized.
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                llm_int8_skip_modules=[
                    "vision_tower", "multi_modal_projector",
                    "embed_tokens", "lm_head",
                ],
            )
            load_kwargs["device_map"] = "auto"
            load_kwargs["torch_dtype"] = torch.bfloat16
        elif self.backend == "cuda_bf16":
            load_kwargs["torch_dtype"] = torch.bfloat16
            load_kwargs["device_map"] = "auto"
        elif self.backend == "mps_fp16":
            load_kwargs["torch_dtype"] = torch.float16
            # device_map="auto" routes to MPS when CUDA is unavailable but to be
            # explicit and avoid offload weirdness, place the whole model on mps.
            load_kwargs["device_map"] = {"": "mps"}
        elif self.backend == "cpu":
            load_kwargs["torch_dtype"] = torch.float32
        else:
            raise ValueError(f"unknown backend: {self.backend}")

        self.model = AutoModelForCausalLM.from_pretrained(cfg.model_id, **load_kwargs)
        self.model.eval()
        self.device = next(self.model.parameters()).device
        print(f"[llm] loaded; device={self.device}, dtype={next(self.model.parameters()).dtype}")

    @torch.no_grad()
    def generate(self, messages: list[dict], max_new_tokens: Optional[int] = None) -> str:
        # Tokenize via the chat template directly. Re-tokenizing the templated
        # string (tokenize=False → tokenizer(...)) double-emits BOS on Gemma,
        # which silently produces empty completions.
        template_kwargs: dict = {}
        if self.cfg.enable_thinking is not None:
            template_kwargs["enable_thinking"] = self.cfg.enable_thinking
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            **template_kwargs,
        ).to(self.device)
        attention_mask = torch.ones_like(input_ids)
        gen_kwargs = {
            "max_new_tokens": max_new_tokens or self.cfg.max_new_tokens,
            "do_sample": self.cfg.do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if self.cfg.do_sample:
            gen_kwargs["temperature"] = self.cfg.temperature
            gen_kwargs["top_p"] = self.cfg.top_p
        out = self.model.generate(
            input_ids=input_ids, attention_mask=attention_mask, **gen_kwargs
        )
        new_tokens = out[0, input_ids.shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)


def load_config(path: str | None) -> LLMConfig:
    if path is None:
        return LLMConfig()
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    fields = {f for f in LLMConfig.__dataclass_fields__}
    known = {k: v for k, v in raw.items() if k in fields}
    extra = {k: v for k, v in raw.items() if k not in fields}
    cfg = LLMConfig(**known)
    cfg.extra = extra
    return cfg
