"""Backends de inferência — só o que muda entre rodadas: o modelo.

Cada backend recebe títulos JÁ pré-normalizados (``pre_normalize_title`` é
aplicado a montante por ``core.normalize_many``), monta o prompt do seu modelo,
chama o modelo e devolve o JSON cru por título. Toda a limpeza/match key vem
de ``normalization.py`` (compartilhada), nunca daqui.

Imports pesados (``google.generativeai``, ``unsloth``) são lazy: importar este
módulo não puxa GPU nem o SDK do Gemini.
"""
from __future__ import annotations

import logging
import os
import time
from abc import ABC, abstractmethod

from .normalization import (
    GEMINI_USER_TEMPLATE,
    QWEN_SYSTEM_PROMPT,
    QWEN_USER_TEMPLATE,
    SYSTEM_PROMPT,
    parse_llm_json,
)

logger = logging.getLogger(__name__)


class Backend(ABC):
    """Contrato único atrás de ``normalize()``.

    ``generate`` recebe títulos pré-normalizados e devolve uma lista de dicts
    (JSON cru do modelo), um por título de entrada e na mesma ordem. Faltas/
    erros devem virar ``{}`` para a linha correspondente (nunca pular).
    """

    name: str = "base"

    @abstractmethod
    def generate(self, titles_pre: list[str]) -> list[dict]:
        ...


# ═══════════════════════════════════════════════════════════════════
#  Gemini (google-generativeai) — batch indexado numa única chamada
# ═══════════════════════════════════════════════════════════════════

class GeminiBackend(Backend):
    name = "gemini"

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str | None = None,
        temperature: float = 0.0,
        batch_size: int = 5,
        sleep_between_batches: float = 2.0,
        max_output_tokens: int = 2048,
    ):
        self.api_key = (api_key or os.getenv("GOOGLE_API_KEY") or "").strip()
        self.model_name = model_name or os.getenv("MODEL_NAME", "gemini-3.1-flash-lite-preview")
        self.temperature = temperature
        self.batch_size = batch_size
        self.sleep_between_batches = sleep_between_batches
        self.max_output_tokens = max_output_tokens
        self._client = None

    def _ensure_client(self):
        if self._client is not None:
            return
        import google.generativeai as genai  # lazy

        if not self.api_key:
            raise RuntimeError(
                "GOOGLE_API_KEY ausente. Defina no ambiente ou passe api_key= ao GeminiBackend."
            )
        genai.configure(api_key=self.api_key)
        self._client = genai.GenerativeModel(
            model_name=self.model_name,
            system_instruction=SYSTEM_PROMPT,
            generation_config=genai.GenerationConfig(
                response_mime_type="application/json",
                temperature=self.temperature,
                max_output_tokens=self.max_output_tokens,
            ),
        )

    def _call_batch(self, titles: list[str], start_idx: int) -> list[dict]:
        """Uma chamada com `len(titles)` títulos indexados. Reproduz extract_batch_llm."""
        lines = [
            GEMINI_USER_TEMPLATE.format(idx=start_idx + i, categoria="", titulo=t)
            for i, t in enumerate(titles)
        ]
        user_prompt = "\n".join(lines)
        try:
            response = self._client.generate_content(user_prompt)
            return parse_llm_json(response.text)
        except ValueError as e:
            logger.warning("Resposta bloqueada para batch: %s", e)
            return []
        except Exception as e:  # noqa: BLE001 — faithful ao notebook
            logger.error("Erro LLM batch: %s", e)
            return []

    def generate(self, titles_pre: list[str]) -> list[dict]:
        self._ensure_client()
        n = len(titles_pre)
        out: list[dict] = [{} for _ in range(n)]

        for batch_start in range(0, n, self.batch_size):
            batch_end = min(batch_start + self.batch_size, n)
            batch_titles = titles_pre[batch_start:batch_end]
            raw_list = self._call_batch(batch_titles, start_idx=batch_start)

            for local_i in range(batch_end - batch_start):
                global_i = batch_start + local_i
                matched = None
                for r in raw_list:
                    if isinstance(r, dict) and r.get("idx") == global_i:
                        matched = r
                        break
                if matched is None and local_i < len(raw_list):
                    matched = raw_list[local_i] if isinstance(raw_list[local_i], dict) else {}
                out[global_i] = matched or {}

            if self.sleep_between_batches and batch_end < n:
                time.sleep(self.sleep_between_batches)

        return out


# ═══════════════════════════════════════════════════════════════════
#  Qwen2.5 fine-tuned (Unsloth, 4-bit) — um título por prompt
# ═══════════════════════════════════════════════════════════════════

class QwenBackend(Backend):
    name = "qwen"

    def __init__(
        self,
        adapter_dir: str | None = None,
        model_name: str = "unsloth/Qwen2.5-3B-Instruct",
        max_seq_length: int = 1536,
        load_in_4bit: bool = True,
        batch_size: int = 8,
        max_new_tokens: int = 256,
    ):
        # adapter_dir=None → base zero-shot; caso contrário, carrega o LoRA salvo
        # (Unsloth detecta o adapter_config.json e monta base+adapter).
        self.adapter_dir = adapter_dir
        self.model_name = model_name
        self.max_seq_length = max_seq_length
        self.load_in_4bit = load_in_4bit
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self._model = None
        self._tokenizer = None

    def _ensure_model(self):
        if self._model is not None:
            return
        from unsloth import FastLanguageModel  # lazy (GPU)

        load_target = self.adapter_dir or self.model_name
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=load_target,
            max_seq_length=self.max_seq_length,
            dtype=None,
            load_in_4bit=self.load_in_4bit,
        )
        FastLanguageModel.for_inference(model)
        self._model = model
        self._tokenizer = tokenizer

    def _build_prompt(self, title: str) -> str:
        messages = [
            {"role": "system", "content": QWEN_SYSTEM_PROMPT},
            {"role": "user", "content": QWEN_USER_TEMPLATE.format(idx=0, titulo=title)},
        ]
        return self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def generate(self, titles_pre: list[str]) -> list[dict]:
        self._ensure_model()
        import torch  # lazy

        out: list[dict] = []
        tok = self._tokenizer
        with torch.inference_mode():
            for i in range(0, len(titles_pre), self.batch_size):
                chunk = titles_pre[i:i + self.batch_size]
                prompts = [self._build_prompt(t) for t in chunk]
                enc = tok(
                    prompts, return_tensors="pt", padding=True,
                    truncation=True, max_length=self.max_seq_length,
                ).to("cuda")
                generated = self._model.generate(
                    **enc,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    temperature=0.0,
                    pad_token_id=tok.eos_token_id,
                )
                for j in range(len(chunk)):
                    new_tokens = generated[j][enc.input_ids.shape[1]:]
                    raw = tok.decode(new_tokens, skip_special_tokens=True)
                    parsed = parse_llm_json(raw)
                    out.append(parsed[0] if parsed else {})
        return out


# ═══════════════════════════════════════════════════════════════════
#  Factory — seleção de backend por rodada
# ═══════════════════════════════════════════════════════════════════

_BACKENDS = {
    "gemini": GeminiBackend,
    "qwen": QwenBackend,
}


def get_backend(name: str, **cfg) -> Backend:
    """Instancia um backend por nome. `name` ∈ {"gemini", "qwen"}.

    Rodar todos os modelos = loop por cima de `get_backend(...)`; a unidade base
    é um backend por vez.
    """
    key = name.strip().lower()
    if key not in _BACKENDS:
        raise ValueError(f"Backend desconhecido: {name!r}. Opções: {sorted(_BACKENDS)}")
    return _BACKENDS[key](**cfg)
