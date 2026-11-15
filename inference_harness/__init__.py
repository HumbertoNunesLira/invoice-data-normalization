"""Harness de inferência: roda um backend de normalização (gemini ou qwen)
sobre os dados de avaliação e gera os CSVs no schema exato que o benchmark
já consome.

Uso típico (uma rodada = um backend):

    from inference_harness import get_backend, build_consistency_csv, build_pairwise_csv

    backend = get_backend("qwen", adapter_dir="outputs/.../final_adapter")  # ou "gemini"
    build_consistency_csv(backend)   # -> output/normalized.csv
    build_pairwise_csv(backend)      # -> output/pairs_normalized.csv

A camada simbólica (pré-normalização, limpeza, match key) é compartilhada por
todos os backends — ver `inference_harness.normalization`.
"""
from .backends import Backend, GeminiBackend, QwenBackend, get_backend
from .core import normalize, normalize_many
from .runner import build_consistency_csv, build_pairwise_csv

__all__ = [
    "Backend",
    "GeminiBackend",
    "QwenBackend",
    "get_backend",
    "normalize",
    "normalize_many",
    "build_consistency_csv",
    "build_pairwise_csv",
]
