"""Interface única de normalização: `normalize(title) -> dict`.

Aplica a pipeline canônica (pré-normalização → modelo → limpeza → match key)
com QUALQUER backend atrás dela. Pré e pós são sempre as funções de
``normalization.py``; só o backend (modelo) varia entre rodadas.
"""
from __future__ import annotations

from .backends import Backend
from .normalization import (
    build_canonical_name,
    build_match_key,
    clean_llm_output,
    pre_normalize_title,
)


def normalize_many(titles: list[str], *, backend: Backend) -> list[dict]:
    """Normaliza vários títulos de uma vez (deixa o backend lotear/bater GPU).

    Retorna, por título e na ordem de entrada, um dict:
        {
          "raw":       <JSON cru do modelo>,
          "cleaned":   <dict após camada simbólica>,
          "match_key": <string estável — coluna `normalized` do benchmark>,
          "canonical": <string display — coluna `display_name`>,
        }
    """
    titles_pre = [pre_normalize_title(t) for t in titles]
    raws = backend.generate(titles_pre)
    if len(raws) != len(titles):
        raise RuntimeError(
            f"Backend {backend.name!r} devolveu {len(raws)} resultados para "
            f"{len(titles)} títulos — esperado 1:1 na mesma ordem."
        )

    results = []
    for raw in raws:
        cleaned = clean_llm_output(raw)
        results.append({
            "raw": raw,
            "cleaned": cleaned,
            "match_key": build_match_key(cleaned),
            "canonical": build_canonical_name(cleaned),
        })
    return results


def normalize(title: str, *, backend: Backend) -> dict:
    """Normaliza um único título. Conveniência sobre ``normalize_many``."""
    return normalize_many([title], backend=backend)[0]
